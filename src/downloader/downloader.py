"""
downloader.py — Download YouTube videos via yt-dlp and ingest them into Milvus.

Each successful ingest is recorded in docs/video_registry.md, after which the
local video file is deleted to keep disk usage bounded.

Usage:
    python run.py download <youtube_url> [--platform youtube]
    python run.py download --query "cooking tips" [--max 5] [--platform youtube]
"""

import os
import sys
import subprocess
from typing import Iterable

from src.log_setup import get_logger
from src.registry import append as registry_append, loaded_urls as registry_urls

logger       = get_logger("downloader")
DOWNLOAD_DIR = "./videos"


def download_url(url: str) -> tuple[str, str, float | None] | None:
    """Download a single YouTube URL. Returns (file_path, title, duration_sec) or None."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--output", f"{DOWNLOAD_DIR}/%(title)s.%(ext)s",
        "--format", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--print", "%(title)s",
        "--print", "%(duration)s",
        "--print", "after_move:filepath",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("yt-dlp failed for %s: %s", url, result.stderr.strip())
        print(f"[downloader] Failed to download {url}:\n{result.stderr}")
        return None

    lines = [ln for ln in result.stdout.strip().splitlines() if ln]
    if len(lines) < 3:
        logger.error("Unexpected yt-dlp output for %s: %r", url, result.stdout)
        return None

    title       = lines[0]
    try:
        duration = float(lines[1])
    except ValueError:
        duration = None
    file_path   = lines[-1]
    return file_path, title, duration


def _normalize(videos: Iterable) -> list[dict]:
    out: list[dict] = []
    for v in videos:
        if isinstance(v, str):
            out.append({"url": v, "title": ""})
        elif isinstance(v, dict) and "url" in v:
            out.append({"url": v["url"], "title": v.get("title", "")})
    return out


def download_and_ingest(videos, platform: str = "youtube") -> list[str]:
    """Download, ingest, log to registry, and delete local file. Returns ingested file_ids.

    URLs already present in `docs/video_registry.md` are skipped before download
    to avoid duplicate ingestion of the same source video across runs.
    """
    items = _normalize(videos)

    seen = registry_urls()
    if seen:
        before = len(items)
        items  = [it for it in items if it["url"] not in seen]
        skipped = before - len(items)
        if skipped:
            logger.info("Skipped %d already-ingested URL(s) per registry", skipped)
            print(f"[downloader] Skipping {skipped} URL(s) already in registry.")

    if not items:
        logger.info("All requested URLs already ingested; nothing new to download.")
        print("[downloader] Nothing new to download — all URLs already in registry.")
        return []

    downloaded = []  # (url, title, duration, file_path)

    # Download all videos before connecting to Milvus — gRPC can't fork subprocesses.
    for item in items:
        url = item["url"]
        logger.info("Downloading %s", url)
        print(f"\n[downloader] Downloading: {url}")
        info = download_url(url)
        if not info:
            continue
        file_path, fetched_title, duration = info
        title = item["title"] or fetched_title
        logger.info("Downloaded %s -> %s (duration=%s)", url, file_path, duration)
        print(f"[downloader] Saved to: {file_path}")
        downloaded.append((url, title, duration, file_path))

    if not downloaded:
        logger.warning("No videos downloaded.")
        print("[downloader] No videos downloaded.")
        return []

    from pymilvus import connections
    from src.ingest.ingest import ingest_video
    connections.connect(uri="./milvus_local.db")

    file_ids: list[str] = []
    for url, title, duration, file_path in downloaded:
        logger.info("Ingesting %s", file_path)
        print(f"\n[downloader] Ingesting: {file_path}")
        try:
            file_id = ingest_video(file_path, platform=platform)
        except Exception as e:
            logger.exception("Ingest failed for %s: %s", file_path, e)
            print(f"[downloader] Ingest failed: {e}")
            continue

        file_ids.append(file_id)
        registry_append(file_id=file_id, url=url, title=title, duration=duration)
        logger.info("Registry row written for file_id=%s", file_id)

        try:
            os.remove(file_path)
            logger.info("Removed local file %s", file_path)
            print(f"[downloader] Removed local file: {file_path}")
        except OSError as e:
            logger.warning("Could not remove %s: %s", file_path, e)
            print(f"[downloader] Could not remove {file_path}: {e}")

    return file_ids


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(1)

    platform = "youtube"
    if "--platform" in args:
        platform = args[args.index("--platform") + 1]

    if "--query" in args:
        from src.downloader.URL_generation import search_top_videos
        query       = args[args.index("--query") + 1]
        max_results = int(args[args.index("--max") + 1]) if "--max" in args else 5
        print(f"[downloader] Searching YouTube for: {query!r}")
        videos = search_top_videos(query, max_results=max_results)
    else:
        videos = [args[0]]

    download_and_ingest(videos, platform=platform)
