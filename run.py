"""
python run.py ingest     <video.mp4> [--prompt "..."] [--platform youtube]
python run.py ingest_all [--prompt "..."] [--platform youtube]
python run.py summary    <file_id>
python run.py query      "your question" [--file_id <id>]
python run.py trend      <niche>
python run.py export     [--out custom_output.csv]
"""

import sys
from dotenv import load_dotenv
from pymilvus import connections

load_dotenv()

MILVUS_URI = "./milvus_local.db"   # milvus-lite local file (no Docker needed)


def connect():
    connections.connect(uri=MILVUS_URI)

def cmd_ingest(args):
    if not args:
        print("Usage: python run.py ingest <video.mp4> [--prompt 'your prompt']")
        sys.exit(1)

    video    = args[0]
    prompt   = "Describe all visible objects, people, actions and events in detail."
    platform = "unknown"
    if "--prompt" in args:
        idx    = args.index("--prompt")
        prompt = args[idx + 1]
    if "--platform" in args:
        idx      = args.index("--platform")
        platform = args[idx + 1]

    from ingest import ingest_video
    connect()
    file_id = ingest_video(video, vlm_prompt=prompt, platform=platform)
    print(f"\nSave this file_id: {file_id}")
#------------------------------------------------------------------------------------------------
def cmd_summary(args):
    if not args:
        print("Usage: python run.py summary <file_id>")
        sys.exit(1)
    from summarize import summarize_video
    connect()
    print(summarize_video(args[0]))
#------------------------------------------------------------------------------------------------
def cmd_query(args):
    if not args:
        print("Usage: python run.py query \"your question\" [--file_id <id>]")
        sys.exit(1)

    question = args[0]
    file_id  = None
    if "--file_id" in args:
        file_id = args[args.index("--file_id") + 1]

    from query import run_query
    connect()
    run_query(question, file_id=file_id)
#------------------------------------------------------------------------------------------------
def cmd_trend(args):
    if not args:
        print("Usage: python run.py trend <niche>  e.g. python run.py trend cooking")
        sys.exit(1)
    from trend import run_trend
    connect()
    run_trend(args[0])
#------------------------------------------------------------------------------------------------
def cmd_ingest_all(args):
    import glob
    videos_dir = "videos"
    prompt     = "Describe all visible objects, people, actions and events in detail."
    platform   = "unknown"
    if "--prompt" in args:
        idx    = args.index("--prompt")
        prompt = args[idx + 1]
    if "--platform" in args:
        idx      = args.index("--platform")
        platform = args[idx + 1]

    patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
    videos   = []
    for p in patterns:
        videos.extend(glob.glob(f"{videos_dir}/{p}"))

    if not videos:
        print(f"No video files found in '{videos_dir}/'")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in '{videos_dir}/'")
    from ingest import ingest_video
    connect()
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Ingesting: {video}")
        try:
            file_id = ingest_video(video, vlm_prompt=prompt, platform=platform)
            print(f"Done: file_id={file_id}")
        except Exception as e:
            print(f"Failed: {e}, skipping.")
#------------------------------------------------------------------------------------------------
def cmd_export(args):
    out = "video_database.csv"
    if "--out" in args:
        out = args[args.index("--out") + 1]
    from export import export_all
    connect()
    export_all(csv_path=out)
#------------------------------------------------------------------------------------------------

COMMANDS = {
    "ingest":     cmd_ingest,
    "ingest_all": cmd_ingest_all,
    "summary":    cmd_summary,
    "query":      cmd_query,
    "trend":      cmd_trend,
    "export":     cmd_export,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0)
    COMMANDS[sys.argv[1]](sys.argv[2:])
