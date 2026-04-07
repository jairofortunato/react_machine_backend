import os
import base64
import json
import subprocess
import tempfile
import httpx
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class ProcessRequest(BaseModel):
    instagram_url: str


class ProfileRequest(BaseModel):
    username: str
    max_posts: int = 12


@app.post("/api/process-video")
async def process_video(req: ProcessRequest):
    if "instagram.com" not in req.instagram_url:
        raise HTTPException(status_code=400, detail="Link inválido do Instagram.")

    with tempfile.TemporaryDirectory(prefix="react-machine-") as tmp_dir:
        tmp = Path(tmp_dir)
        audio_path = tmp / "audio.m4a"
        frame_path = tmp / "frame.jpg"

        # 1. Download video + thumbnail + metadata
        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "--write-thumbnail",
                    "--write-info-json",
                    "--output",
                    str(tmp / "video.%(ext)s"),
                    "--no-playlist",
                    req.instagram_url,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao baixar o vídeo. Verifique se o link está correto. {e.stderr[:200]}",
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=500, detail="Timeout ao baixar o vídeo.")

        # Find files
        files = list(tmp.iterdir())
        video_file = next(
            (f for f in files if f.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov")),
            None,
        )
        thumbnail_file = next(
            (f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")),
            None,
        )
        info_file = next((f for f in files if f.name.endswith(".info.json")), None)

        if not video_file:
            raise HTTPException(status_code=500, detail="Não foi possível baixar o vídeo.")

        # 2. Parse metadata
        video_stats = {
            "likes": None,
            "comments": None,
            "shares": None,
            "date": None,
            "description": None,
        }

        if info_file:
            info = json.loads(info_file.read_text(encoding="utf-8"))
            raw_date = info.get("upload_date")  # YYYYMMDD
            video_stats = {
                "likes": info.get("like_count"),
                "comments": info.get("comment_count"),
                "shares": info.get("repost_count") or info.get("share_count"),
                "date": (
                    f"{raw_date[6:8]}/{raw_date[4:6]}/{raw_date[:4]}"
                    if raw_date
                    else None
                ),
                "description": info.get("description"),
            }

        # 3. Extract audio with ffmpeg
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(video_file), "-vn", "-acodec", "aac", "-y", str(audio_path)],
                capture_output=True,
                timeout=60,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            raise HTTPException(status_code=500, detail="Erro ao extrair áudio do vídeo.")

        # 4. Extract frame at ~1 second
        frame_base64 = None
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(video_file), "-ss", "1", "-vframes", "1", "-y", str(frame_path)],
                capture_output=True,
                timeout=30,
                check=True,
            )
            frame_base64 = base64.b64encode(frame_path.read_bytes()).decode("utf-8")
        except Exception:
            pass  # Frame extraction may fail for very short videos

        # 5. Read thumbnail as base64
        thumbnail_base64 = None
        if thumbnail_file:
            thumbnail_base64 = base64.b64encode(thumbnail_file.read_bytes()).decode("utf-8")

        # 6. Transcribe with Whisper
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="pt",
                )
            transcript = transcription.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro na transcrição: {str(e)[:200]}")

        return {
            "transcript": transcript,
            "frameImage": frame_base64,
            "thumbnailImage": thumbnail_base64,
            "videoStats": video_stats,
        }


@app.post("/api/profile-posts")
async def profile_posts(req: ProfileRequest):
    username = req.username.strip().lstrip("@")
    if not username:
        raise HTTPException(status_code=400, detail="Username é obrigatório.")

    # Use Instagram's public web API to fetch profile posts
    url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "X-IG-App-ID": "936619743392459",
        "X-Requested-With": "XMLHttpRequest",
    }

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client_http:
        try:
            resp = await client_http.get(url, headers=headers)
        except httpx.TimeoutException:
            raise HTTPException(status_code=500, detail="Timeout ao buscar perfil.")

        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Perfil não encontrado.")
        if resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao buscar perfil (status {resp.status_code}).",
            )

        try:
            data = resp.json()
        except Exception:
            raise HTTPException(status_code=500, detail="Resposta inválida do Instagram.")

    user_data = data.get("data", {}).get("user")
    if not user_data:
        raise HTTPException(status_code=404, detail="Perfil não encontrado ou privado.")

    edges = (
        user_data.get("edge_owner_to_timeline_media", {}).get("edges", [])
    )

    posts = []
    for edge in edges[: req.max_posts]:
        node = edge.get("node", {})
        # Only include video posts (reels/videos)
        if not node.get("is_video", False):
            continue

        shortcode = node.get("shortcode", "")
        timestamp = node.get("taken_at_timestamp")
        date_str = None
        if timestamp:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            date_str = dt.strftime("%d/%m/%Y")

        caption_edges = node.get("edge_media_to_caption", {}).get("edges", [])
        caption = caption_edges[0]["node"]["text"] if caption_edges else ""

        posts.append({
            "id": shortcode,
            "url": f"https://www.instagram.com/reel/{shortcode}/",
            "title": "",
            "description": caption,
            "thumbnail": node.get("thumbnail_src") or node.get("display_url"),
            "likes": node.get("edge_media_preview_like", {}).get("count"),
            "comments": node.get("edge_media_to_comment", {}).get("count"),
            "views": node.get("video_view_count"),
            "date": date_str,
        })

    return {"username": username, "posts": posts}


@app.get("/health")
async def health():
    return {"status": "ok"}
