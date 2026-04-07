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

    # Use yt-dlp to scrape individual post info from a profile
    # First try the graphql approach via yt-dlp with cookies workaround
    profile_url = f"https://www.instagram.com/{username}/reels/"

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                "--playlist-end",
                str(req.max_posts),
                "--extractor-args",
                "instagram:compatible_formats=dash",
                profile_url,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Timeout ao buscar perfil.")

    # If yt-dlp fails, try scraping the page HTML for shortcodes
    posts = []

    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                info = json.loads(line)
                raw_date = info.get("upload_date")
                post_id = info.get("id", "")
                posts.append({
                    "id": post_id,
                    "url": info.get("url") or info.get("webpage_url") or f"https://www.instagram.com/reel/{post_id}/",
                    "title": info.get("title", ""),
                    "description": info.get("description", ""),
                    "thumbnail": info.get("thumbnail") or (info.get("thumbnails", [{}]) or [{}])[-1].get("url"),
                    "likes": info.get("like_count"),
                    "comments": info.get("comment_count"),
                    "views": info.get("view_count"),
                    "date": (
                        f"{raw_date[6:8]}/{raw_date[4:6]}/{raw_date[:4]}"
                        if raw_date
                        else None
                    ),
                })
            except json.JSONDecodeError:
                continue

    # Fallback: scrape shortcodes from HTML page
    if not posts:
        import re
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as http:
            try:
                resp = await http.get(
                    f"https://www.instagram.com/{username}/",
                    headers={
                        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
                    },
                )
                if resp.status_code == 200:
                    # Extract shortcodes from the HTML
                    shortcodes = re.findall(r'/reel/([A-Za-z0-9_-]+)/', resp.text)
                    # Also try /p/ pattern
                    shortcodes += re.findall(r'/p/([A-Za-z0-9_-]+)/', resp.text)
                    # Deduplicate while preserving order
                    seen = set()
                    unique_codes = []
                    for sc in shortcodes:
                        if sc not in seen:
                            seen.add(sc)
                            unique_codes.append(sc)

                    for sc in unique_codes[: req.max_posts]:
                        posts.append({
                            "id": sc,
                            "url": f"https://www.instagram.com/reel/{sc}/",
                            "title": "",
                            "description": "",
                            "thumbnail": None,
                            "likes": None,
                            "comments": None,
                            "views": None,
                            "date": None,
                        })
            except Exception:
                pass

    if not posts:
        raise HTTPException(
            status_code=500,
            detail="Não foi possível buscar posts. O perfil pode ser privado ou o Instagram bloqueou a requisição.",
        )

    return {"username": username, "posts": posts}


@app.get("/health")
async def health():
    return {"status": "ok"}
