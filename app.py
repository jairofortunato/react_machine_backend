import os
import base64
import json
import subprocess
import tempfile
import httpx
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
    max_id: str = ""


@app.post("/api/process-video")
async def process_video(req: ProcessRequest):
    if not req.instagram_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Link inválido.")

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
            err_str = str(e).lower()
            if "quota" in err_str or "billing" in err_str or "rate" in err_str or "insufficient" in err_str:
                raise HTTPException(
                    status_code=500,
                    detail="Os créditos do serviço de transcrição acabaram. Entre em contato com o Jairo pelo WhatsApp (48) 99926-3038 para resolver.",
                )
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

    rapidapi_key = os.environ.get("RAPIDAPI_KEY")
    if not rapidapi_key:
        raise HTTPException(status_code=500, detail="RAPIDAPI_KEY não configurada.")

    async with httpx.AsyncClient(timeout=20) as http:
        try:
            resp = await http.post(
                "https://instagram120.p.rapidapi.com/api/instagram/posts",
                headers={
                    "Content-Type": "application/json",
                    "x-rapidapi-key": rapidapi_key,
                    "x-rapidapi-host": "instagram120.p.rapidapi.com",
                },
                json={"username": username, "maxId": req.max_id},
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=500, detail="Timeout ao buscar perfil.")

        if resp.status_code == 429 or resp.status_code == 403:
            raise HTTPException(
                status_code=500,
                detail="Os créditos da busca de perfis acabaram. Entre em contato com o Jairo pelo WhatsApp (48) 99926-3038 para resolver.",
            )
        if resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao buscar perfil (status {resp.status_code}).",
            )

        try:
            data = resp.json()
        except Exception:
            raise HTTPException(status_code=500, detail="Resposta inválida da API.")

    edges = data.get("result", {}).get("edges", [])
    if not edges:
        raise HTTPException(
            status_code=404,
            detail="Nenhum post encontrado. O perfil pode ser privado.",
        )

    posts = []
    for edge in edges:
        node = edge.get("node", {})
        # media_type: 1=photo, 2=video, 8=carousel
        # Include videos (2) and carousels (8) that may contain videos
        media_type = node.get("media_type")
        is_video = media_type == 2 or node.get("video_versions") is not None

        shortcode = node.get("code", "")
        caption_data = node.get("caption") or {}
        caption = caption_data.get("text", "") if isinstance(caption_data, dict) else ""
        taken_at = node.get("taken_at")

        date_str = None
        if taken_at:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(taken_at, tz=timezone.utc)
            date_str = dt.strftime("%d/%m/%Y")

        thumbnail = None
        candidates = node.get("image_versions2", {}).get("candidates", [])
        if candidates:
            thumbnail = candidates[0].get("url")

        posts.append({
            "id": shortcode,
            "url": f"https://www.instagram.com/reel/{shortcode}/" if is_video else f"https://www.instagram.com/p/{shortcode}/",
            "title": "",
            "description": caption,
            "thumbnail": thumbnail,
            "likes": node.get("like_count"),
            "comments": node.get("comment_count"),
            "views": node.get("video_view_count") if is_video else None,
            "date": date_str,
            "is_video": is_video,
        })

        if len(posts) >= req.max_posts:
            break

    # Pagination cursor
    page_info = data.get("result", {}).get("page_info", {})
    next_max_id = page_info.get("end_cursor", "")
    has_more = page_info.get("has_next_page", False)

    return {"username": username, "posts": posts, "nextMaxId": next_max_id if has_more else None}


class DownloadRequest(BaseModel):
    url: str


@app.post("/api/download-video")
async def download_video(req: DownloadRequest):
    if not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Link inválido.")

    tmp_dir = tempfile.mkdtemp(prefix="react-machine-dl-")
    tmp = Path(tmp_dir)
    output_template = str(tmp / "video.%(ext)s")

    try:
        subprocess.run(
            [
                "yt-dlp",
                "--output", output_template,
                "--no-playlist",
                "--merge-output-format", "mp4",
                req.url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao baixar o vídeo. {e.stderr[:200]}",
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Timeout ao baixar o vídeo.")

    files = list(tmp.iterdir())
    video_file = next(
        (f for f in files if f.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov")),
        None,
    )

    if not video_file:
        raise HTTPException(status_code=500, detail="Não foi possível baixar o vídeo.")

    return FileResponse(
        path=str(video_file),
        media_type="video/mp4",
        filename=video_file.name,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
