import asyncio
import os
import sqlite3
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

load_dotenv()

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "onca.db"))
RAPID_HOST = "instagram120.p.rapidapi.com"
WEEK = 7 * 86400

DEFAULT_ACCOUNTS = [
    "andiara.ferreiraadv",
    "brunodias.missao",
    "felipebarcellos.sc",
    "jairo99x",
    "tucofariassc",
    "cleiton_siqueira1",
    "marcodosuldomundo",
    "tessarisc",
    "carolformigoni.sc",
    "correasc14",
    "edupercio",
    "precandidatonilsonvicenti",
    "vagner.visoli",
    "margarethpratts",
    "paulojosuesc",
    "rodrigues_coronel",
    "santiagocesarsc",
    "rafaeldemarco.sc",
    "jordambrito.sc",
    "diegos.vieira",
    "eduardocezarmariano",
    "mauriciocoutoadv",
    "evertonjfmelo",
    "jonathanborges.sc",
    "edu_silva378",
    "romulopericles.sc",
    "gabrieloliveira_sc",
    "rafachagas",
    "rodrigokoe_sc",
    "gabrielmartinstj",
    "joaoluizgon",
    "lucasilvasc",
    "fiscalizatijucas",
    "larissakamers",
    "olucasfsilveira",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS accounts(
            username TEXT PRIMARY KEY,
            full_name TEXT,
            biography TEXT,
            is_private INTEGER DEFAULT 0,
            avatar BLOB,
            last_refreshed REAL,
            last_error TEXT
        );
        CREATE TABLE IF NOT EXISTS snapshots(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            followers INTEGER,
            following INTEGER,
            media_count INTEGER,
            taken_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_snapshots ON snapshots(username, taken_at);
        CREATE TABLE IF NOT EXISTS posts(
            pk TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            code TEXT,
            taken_at REAL,
            like_count INTEGER,
            comment_count INTEGER,
            view_count INTEGER,
            media_type INTEGER,
            caption TEXT,
            fetched_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_posts ON posts(username, taken_at);
        """
    )
    for u in DEFAULT_ACCOUNTS:
        conn.execute("INSERT OR IGNORE INTO accounts(username) VALUES(?)", (u,))
    conn.commit()
    conn.close()


init_db()


class RefreshRequest(BaseModel):
    usernames: list[str] | None = None


class AccountRequest(BaseModel):
    username: str


async def rapid_post(http: httpx.AsyncClient, path: str, payload: dict, attempts: int = 4) -> dict:
    key = os.environ.get("RAPIDAPI_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="RAPIDAPI_KEY não configurada.")

    last_status = None
    for i in range(attempts):
        try:
            resp = await http.post(
                f"https://{RAPID_HOST}{path}",
                headers={
                    "Content-Type": "application/json",
                    "x-rapidapi-key": key,
                    "x-rapidapi-host": RAPID_HOST,
                },
                json=payload,
            )
        except httpx.HTTPError:
            if i == attempts - 1:
                raise HTTPException(status_code=502, detail="Falha de rede ao consultar a API do Instagram.")
            await asyncio.sleep(2 * (i + 1))
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                raise HTTPException(status_code=502, detail="Resposta inválida da API do Instagram.")
        if resp.status_code in (403, 429):
            raise HTTPException(
                status_code=502,
                detail="Os créditos da API do Instagram acabaram. Entre em contato com o Jairo pelo WhatsApp (48) 99926-3038 para resolver.",
            )
        # A API devolve 500 com "not found" para conta sem reels/posts — é definitivo, não adianta repetir.
        try:
            body = resp.json()
        except Exception:
            body = {}
        if isinstance(body, dict) and "not found" in str(body.get("message", "")).lower():
            return {}
        # Outros 5xx costumam ser rate limit disfarçado — espera e tenta de novo.
        last_status = resp.status_code
        await asyncio.sleep(2 * (i + 1))

    raise HTTPException(
        status_code=502,
        detail=f"Erro na API do Instagram (status {last_status}).",
    )


async def _rapid_post_retry(http: httpx.AsyncClient, path: str, payload: dict, attempts: int = 3) -> dict:
    """A API às vezes retorna 200 com resultado vazio sob carga — tenta de novo antes de aceitar."""
    last: dict = {}
    for i in range(attempts):
        try:
            last = await rapid_post(http, path, payload)
        except HTTPException as e:
            if "créditos" in str(e.detail) or i == attempts - 1:
                raise
            await asyncio.sleep(1.5 * (i + 1))
            continue
        if not last or (last.get("result") or {}).get("edges"):
            return last
        await asyncio.sleep(1.5 * (i + 1))
    return last


def _caption_text(node: dict) -> str:
    cap = node.get("caption") or {}
    return (cap.get("text") or "")[:500]


def _caption_created_at(node: dict):
    cap = node.get("caption") or {}
    return cap.get("created_at")


async def refresh_account(http: httpx.AsyncClient, username: str) -> dict:
    now = time.time()

    info = await rapid_post(http, "/api/instagram/userInfo", {"username": username})
    result = info.get("result") or []
    user = (result[0] or {}).get("user") if result else None
    if not user:
        raise HTTPException(status_code=404, detail=f"Perfil @{username} não encontrado.")

    followers = user.get("follower_count")
    following = user.get("following_count")
    media_count = user.get("media_count")
    full_name = user.get("full_name") or ""
    biography = user.get("biography") or ""
    is_private = 1 if user.get("is_private") else 0

    avatar = None
    pic_url = user.get("profile_pic_url")
    if pic_url:
        try:
            pic = await http.get(pic_url)
            if pic.status_code == 200:
                avatar = pic.content
        except httpx.HTTPError:
            pass

    posts_nodes: dict[str, dict] = {}
    if not is_private:
        data = await _rapid_post_retry(http, "/api/instagram/posts", {"username": username, "maxId": ""})
        for edge in (data.get("result") or {}).get("edges") or []:
            node = edge.get("node") or {}
            pk = node.get("pk") or node.get("id")
            if pk:
                posts_nodes[str(pk)] = {
                    "pk": str(pk),
                    "code": node.get("code"),
                    "taken_at": node.get("taken_at"),
                    "like_count": node.get("like_count"),
                    "comment_count": node.get("comment_count"),
                    "view_count": node.get("play_count") or node.get("view_count"),
                    "media_type": node.get("media_type"),
                    "caption": _caption_text(node),
                }

        # Reels carregam play_count (views), que o feed de posts não expõe.
        data = await _rapid_post_retry(http, "/api/instagram/reels", {"username": username, "maxId": ""})
        for edge in (data.get("result") or {}).get("edges") or []:
            media = (edge.get("node") or {}).get("media") or {}
            pk = media.get("pk") or media.get("id")
            if not pk:
                continue
            pk = str(pk).split("_")[0]
            views = media.get("play_count") or media.get("ig_play_count") or media.get("view_count")
            if pk in posts_nodes:
                if views:
                    posts_nodes[pk]["view_count"] = views
            else:
                posts_nodes[pk] = {
                    "pk": pk,
                    "code": media.get("code"),
                    "taken_at": media.get("taken_at") or _caption_created_at(media),
                    "like_count": media.get("like_count"),
                    "comment_count": media.get("comment_count"),
                    "view_count": views,
                    "media_type": media.get("media_type"),
                    "caption": _caption_text(media),
                }

    conn = db()
    conn.execute(
        """
        INSERT INTO accounts(username, full_name, biography, is_private, avatar, last_refreshed, last_error)
        VALUES(?,?,?,?,?,?,NULL)
        ON CONFLICT(username) DO UPDATE SET
            full_name=excluded.full_name,
            biography=excluded.biography,
            is_private=excluded.is_private,
            avatar=COALESCE(excluded.avatar, accounts.avatar),
            last_refreshed=excluded.last_refreshed,
            last_error=NULL
        """,
        (username, full_name, biography, is_private, avatar, now),
    )
    conn.execute(
        "INSERT INTO snapshots(username, followers, following, media_count, taken_at) VALUES(?,?,?,?,?)",
        (username, followers, following, media_count, now),
    )
    for p in posts_nodes.values():
        conn.execute(
            """
            INSERT INTO posts(pk, username, code, taken_at, like_count, comment_count, view_count, media_type, caption, fetched_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(pk) DO UPDATE SET
                like_count=excluded.like_count,
                comment_count=excluded.comment_count,
                view_count=COALESCE(excluded.view_count, posts.view_count),
                taken_at=COALESCE(excluded.taken_at, posts.taken_at),
                caption=excluded.caption,
                fetched_at=excluded.fetched_at
            """,
            (
                p["pk"], username, p["code"], p["taken_at"], p["like_count"],
                p["comment_count"], p["view_count"], p["media_type"], p["caption"], now,
            ),
        )
    conn.commit()
    conn.close()

    return {"username": username, "followers": followers, "posts": len(posts_nodes)}


def _avg(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def compute_metrics(conn: sqlite3.Connection, username: str) -> dict:
    now = time.time()
    snaps = conn.execute(
        "SELECT followers, following, media_count, taken_at FROM snapshots WHERE username=? ORDER BY taken_at",
        (username,),
    ).fetchall()
    latest = snaps[-1] if snaps else None
    followers = latest["followers"] if latest else None

    # Crescimento semanal: compara o snapshot mais recente com o mais próximo de 7 dias atrás
    # e normaliza linearmente para uma janela de 7 dias.
    growth_abs = growth_pct = None
    growth_window_days = None
    if len(snaps) >= 2 and latest and latest["followers"] is not None:
        target = latest["taken_at"] - WEEK
        candidates = [s for s in snaps[:-1] if s["followers"] is not None]
        if candidates:
            old = min(candidates, key=lambda s: abs(s["taken_at"] - target))
            days = (latest["taken_at"] - old["taken_at"]) / 86400
            if days >= 0.04:  # pelo menos ~1h de intervalo
                delta = latest["followers"] - old["followers"]
                growth_abs = round(delta * (7 / days), 1)
                if old["followers"]:
                    growth_pct = round(delta / old["followers"] * 100 * (7 / days), 3)
                growth_window_days = round(days, 2)

    posts = conn.execute(
        "SELECT taken_at, like_count, comment_count, view_count, media_type, code, caption "
        "FROM posts WHERE username=? AND taken_at IS NOT NULL ORDER BY taken_at DESC LIMIT 24",
        (username,),
    ).fetchall()

    # Frequência: usa o intervalo entre o post mais novo e o mais antigo conhecidos,
    # o que não satura para quem posta muito (a API só retorna ~12 posts por página).
    posts_per_week = None
    if len(posts) >= 2:
        span_days = (posts[0]["taken_at"] - posts[-1]["taken_at"]) / 86400
        if span_days >= 0.5:
            posts_per_week = round((len(posts) - 1) / span_days * 7, 2)
    elif len(posts) == 1:
        posts_per_week = round(1 / max((now - posts[0]["taken_at"]) / 86400, 7) * 7, 2)

    recent = posts[:12]
    avg_likes = _avg([p["like_count"] for p in recent])
    avg_comments = _avg([p["comment_count"] for p in recent])
    avg_views = _avg([p["view_count"] for p in recent])
    max_likes = max((p["like_count"] or 0 for p in recent), default=None)
    max_views = max((p["view_count"] or 0 for p in recent), default=None)

    # Produção semanal: soma do que foi publicado nos últimos 28 dias ÷ 4.
    # Junta alcance e constância — quem posta um viral por trimestre não
    # pontua mais do que quem entrega views toda semana. Sem posts na
    # janela = 0 (não postar derruba o desempenho de propósito).
    window_posts = [p for p in posts if p["taken_at"] >= now - 28 * 86400]
    if posts:
        weekly_likes = round(sum(p["like_count"] or 0 for p in window_posts) / 4, 1)
        if window_posts and not any(p["view_count"] for p in window_posts):
            weekly_views = None  # postou, mas só fotos — não dá para medir views
        else:
            weekly_views = round(sum(p["view_count"] or 0 for p in window_posts) / 4, 1)
    else:
        weekly_likes = weekly_views = None
    weekly_views_per_follower = (
        round(weekly_views / followers, 3) if followers and weekly_views is not None else None
    )

    engagement_pct = None
    views_per_follower = None
    if followers:
        if avg_likes is not None or avg_comments is not None:
            engagement_pct = round(((avg_likes or 0) + (avg_comments or 0)) / followers * 100, 3)
        if avg_views is not None:
            views_per_follower = round(avg_views / followers, 3)

    return {
        "username": username,
        "followers": followers,
        "following": latest["following"] if latest else None,
        "media_count": latest["media_count"] if latest else None,
        "posts_per_week": posts_per_week,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "avg_views": avg_views,
        "max_likes": max_likes,
        "max_views": max_views,
        "engagement_pct": engagement_pct,
        "views_per_follower": views_per_follower,
        "weekly_likes": weekly_likes,
        "weekly_views": weekly_views,
        "weekly_views_per_follower": weekly_views_per_follower,
        "posts_last_28d": len(window_posts),
        "weekly_growth_abs": growth_abs,
        "weekly_growth_pct": growth_pct,
        "growth_window_days": growth_window_days,
        "last_post_at": posts[0]["taken_at"] if posts else None,
        "snapshots": len(snaps),
    }


@app.get("/api/accounts")
def list_accounts():
    conn = db()
    rows = conn.execute(
        "SELECT username, full_name, is_private, last_refreshed, last_error, avatar IS NOT NULL AS has_avatar "
        "FROM accounts ORDER BY username"
    ).fetchall()
    conn.close()
    return {"accounts": [dict(r) for r in rows]}


@app.post("/api/accounts")
def add_account(req: AccountRequest):
    username = req.username.strip().lstrip("@").lower()
    if not username:
        raise HTTPException(status_code=400, detail="Username é obrigatório.")
    conn = db()
    conn.execute("INSERT OR IGNORE INTO accounts(username) VALUES(?)", (username,))
    conn.commit()
    conn.close()
    return {"ok": True, "username": username}


@app.delete("/api/accounts/{username}")
def remove_account(username: str):
    conn = db()
    conn.execute("DELETE FROM accounts WHERE username=?", (username,))
    conn.execute("DELETE FROM snapshots WHERE username=?", (username,))
    conn.execute("DELETE FROM posts WHERE username=?", (username,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/api/refresh")
async def refresh(req: RefreshRequest):
    conn = db()
    if req.usernames:
        usernames = [u.strip().lstrip("@") for u in req.usernames if u.strip()]
    else:
        usernames = [r["username"] for r in conn.execute("SELECT username FROM accounts").fetchall()]
    conn.close()

    results = []

    # Sequencial de propósito: a API derruba chamadas em paralelo com erro 500.
    async with httpx.AsyncClient(timeout=30) as http:
        for username in usernames:
            try:
                r = await refresh_account(http, username)
                results.append({**r, "ok": True})
            except HTTPException as e:
                _record_error(username, str(e.detail))
                results.append({"username": username, "ok": False, "error": str(e.detail)})
            except Exception as e:
                _record_error(username, str(e))
                results.append({"username": username, "ok": False, "error": str(e)})
            await asyncio.sleep(0.7)

    ok = [r for r in results if r["ok"]]
    return {"refreshed": len(ok), "failed": len(results) - len(ok), "results": results}


def _record_error(username: str, error: str):
    conn = db()
    conn.execute("UPDATE accounts SET last_error=? WHERE username=?", (error[:300], username))
    conn.commit()
    conn.close()


@app.get("/api/dashboard")
def dashboard():
    conn = db()
    accounts = conn.execute(
        "SELECT username, full_name, is_private, last_refreshed, last_error FROM accounts"
    ).fetchall()
    items = []
    for a in accounts:
        m = compute_metrics(conn, a["username"])
        m["full_name"] = a["full_name"]
        m["is_private"] = bool(a["is_private"])
        m["last_refreshed"] = a["last_refreshed"]
        m["last_error"] = a["last_error"]
        items.append(m)
    conn.close()

    with_data = [i for i in items if i["followers"] is not None]
    totals = {
        "accounts": len(items),
        "with_data": len(with_data),
        "total_followers": sum(i["followers"] for i in with_data) if with_data else 0,
        "last_refreshed": max((i["last_refreshed"] or 0 for i in items), default=None),
    }
    return {"totals": totals, "accounts": items}


@app.get("/api/account/{username}")
def account_detail(username: str):
    conn = db()
    acc = conn.execute(
        "SELECT username, full_name, biography, is_private, last_refreshed, last_error FROM accounts WHERE username=?",
        (username,),
    ).fetchone()
    if not acc:
        conn.close()
        raise HTTPException(status_code=404, detail="Conta não encontrada.")

    metrics = compute_metrics(conn, username)
    snaps = conn.execute(
        "SELECT followers, following, media_count, taken_at FROM snapshots WHERE username=? ORDER BY taken_at",
        (username,),
    ).fetchall()
    posts = conn.execute(
        "SELECT pk, code, taken_at, like_count, comment_count, view_count, media_type, caption "
        "FROM posts WHERE username=? AND taken_at IS NOT NULL ORDER BY taken_at DESC LIMIT 24",
        (username,),
    ).fetchall()
    conn.close()

    return {
        "profile": dict(acc),
        "metrics": metrics,
        "history": [dict(s) for s in snaps],
        "posts": [dict(p) for p in posts],
    }


@app.get("/api/avatar/{username}")
def avatar(username: str):
    conn = db()
    row = conn.execute("SELECT avatar FROM accounts WHERE username=?", (username,)).fetchone()
    conn.close()
    if not row or not row["avatar"]:
        raise HTTPException(status_code=404, detail="Sem avatar.")
    return Response(content=row["avatar"], media_type="image/jpeg")
