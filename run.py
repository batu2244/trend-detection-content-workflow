"""
python run.py ingest      <video.mp4> [--prompt "..."] [--platform youtube]
python run.py ingest_all  [--prompt "..."] [--platform youtube]
python run.py summary     <file_id>
python run.py query       "your question" [--file_id <id>]
python run.py trend       <niche>
python run.py clusters    [<niche>]   # bare form runs every niche in the database
python run.py download    <youtube_url> [--platform youtube]
python run.py download    --query "cooking tips" [--max 5] [--platform youtube]
python run.py --trendpilot --topic "cooking" [--max 5]
"""

import sys
from collections import Counter
from dotenv import load_dotenv
from pymilvus import connections

from src.log_setup import setup_logging, get_logger

load_dotenv()
setup_logging()
logger = get_logger("run")

MILVUS_URI = "./milvus_local.db"   # milvus-lite local file (no Docker needed)


def connect():
    connections.connect(uri=MILVUS_URI)


def _arg(args, flag, default=None, cast=str):
    if flag in args:
        return cast(args[args.index(flag) + 1])
    return default


def cmd_ingest(args):
    if not args:
        print("Usage: python run.py ingest <video.mp4> [--prompt 'your prompt']")
        sys.exit(1)

    video    = args[0]
    prompt   = _arg(args, "--prompt",   "Describe all visible objects, people, actions and events in detail.")
    platform = _arg(args, "--platform", "unknown")

    from src.ingest.ingest import ingest_video
    connect()
    file_id = ingest_video(video, vlm_prompt=prompt, platform=platform)
    print(f"\nSave this file_id: {file_id}")
#------------------------------------------------------------------------------------------------
def cmd_summary(args):
    if not args:
        print("Usage: python run.py summary <file_id>")
        sys.exit(1)
    from src.summarize.summarize import summarize_video
    connect()
    print(summarize_video(args[0]))
#------------------------------------------------------------------------------------------------
def cmd_query(args):
    if not args:
        print("Usage: python run.py query \"your question\" [--file_id <id>]")
        sys.exit(1)

    question = args[0]
    file_id  = _arg(args, "--file_id")

    from src.query.query import run_query
    connect()
    run_query(question, file_id=file_id)
#------------------------------------------------------------------------------------------------
def cmd_trend(args):
    if not args:
        print("Usage: python run.py trend <niche>  e.g. python run.py trend cooking")
        sys.exit(1)
    from src.trend.trend import run_trend
    connect()
    run_trend(args[0])
#------------------------------------------------------------------------------------------------
def cmd_clusters(args):
    """`clusters <niche>` for a single niche; bare `clusters` runs every stored niche."""
    from src.trend.trend import run_clusters, run_clusters_all
    connect()
    if args:
        run_clusters(args[0])
    else:
        run_clusters_all()
#------------------------------------------------------------------------------------------------
def cmd_ingest_all(args):
    import glob
    videos_dir = "videos"
    prompt     = _arg(args, "--prompt",   "Describe all visible objects, people, actions and events in detail.")
    platform   = _arg(args, "--platform", "unknown")

    patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
    videos   = []
    for p in patterns:
        videos.extend(glob.glob(f"{videos_dir}/{p}"))

    if not videos:
        print(f"No video files found in '{videos_dir}/'")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in '{videos_dir}/'")
    from src.ingest.ingest import ingest_video
    connect()
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Ingesting: {video}")
        try:
            file_id = ingest_video(video, vlm_prompt=prompt, platform=platform)
            print(f"Done: file_id={file_id}")
        except Exception as e:
            print(f"Failed: {e}, skipping.")
#------------------------------------------------------------------------------------------------
def cmd_download(args):
    if not args:
        print("Usage: python run.py download <youtube_url> [--platform youtube]")
        print("       python run.py download --query \"cooking tips\" [--max 5] [--platform youtube]")
        sys.exit(1)

    platform = _arg(args, "--platform", "youtube")

    from src.downloader.URL_generation import search_top_videos
    from src.downloader.downloader import download_and_ingest

    if "--query" in args:
        query       = args[args.index("--query") + 1]
        max_results = _arg(args, "--max", 5, int)
        print(f"[download] Searching YouTube for: {query!r}")
        videos = search_top_videos(query, max_results=max_results)
    else:
        videos = [args[0]]

    download_and_ingest(videos, platform=platform)
#------------------------------------------------------------------------------------------------
def cmd_trendpilot(args):
    """End-to-end: search YouTube shorts → download → ingest+summarize → registry+cleanup → trend."""
    topic = _arg(args, "--topic")
    max_n = _arg(args, "--max", 5, int)
    if not topic:
        print('Usage: python run.py --trendpilot --topic "cooking" [--max 5]')
        sys.exit(1)

    logger.info("Trendpilot start topic=%r max=%d", topic, max_n)
    print(f"\n[trendpilot] Topic: {topic!r}  max: {max_n}")

    from src.downloader.URL_generation import search_top_videos
    from src.downloader.downloader import download_and_ingest
    from src.trend.trend import run_trend
    from db import ensure_summaries_collection

    videos = search_top_videos(topic, max_results=max_n)
    if not videos:
        print("[trendpilot] No videos found.")
        return

    file_ids = download_and_ingest(videos, platform="youtube")

    if file_ids:
        # Resolve dominant niche from the just-ingested videos.
        col = ensure_summaries_collection()
        col.load()
        quoted = ", ".join(f'"{fid}"' for fid in file_ids)
        rows   = col.query(
            expr=f"file_id in [{quoted}]",
            output_fields=["niche"],
            limit=len(file_ids),
        )
        niches = [r["niche"] for r in rows if r.get("niche")]
        niche  = Counter(niches).most_common(1)[0][0] if niches else topic
        logger.info("Trendpilot dominant niche=%r (from %d ingested)", niche, len(file_ids))
        print(f"\n[trendpilot] Dominant niche detected: {niche!r}")
    else:
        # All search results were duplicates — run trend over what's already stored.
        # download_and_ingest never opened a Milvus connection on the early-return path,
        # so we have to connect ourselves before run_trend touches Milvus.
        connect()
        niche = topic
        logger.info("Trendpilot no new ingests; running trend on topic=%r", niche)
        print(f"\n[trendpilot] No new videos ingested. Running trend on existing rows for niche={niche!r}.")

    run_trend(niche)
    logger.info("Trendpilot done topic=%r ingested=%d", topic, len(file_ids))
#------------------------------------------------------------------------------------------------

COMMANDS = {
    "ingest":       cmd_ingest,
    "ingest_all":   cmd_ingest_all,
    "summary":      cmd_summary,
    "query":        cmd_query,
    "trend":        cmd_trend,
    "clusters":     cmd_clusters,
    "download":     cmd_download,
    "--trendpilot": cmd_trendpilot,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0)
    COMMANDS[sys.argv[1]](sys.argv[2:])
