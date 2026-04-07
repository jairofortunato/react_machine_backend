import os
import base64
import json
import subprocess
import tempfile
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


@app.get("/health")
async def health():
    return {"status": "ok"}
