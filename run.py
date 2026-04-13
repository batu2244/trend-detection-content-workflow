"""
python run.py ingest  video.mp4
python run.py ingest  video.mp4 --prompt "Focus on people and actions"
python run.py summary <file_id>
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

    video = args[0]
    prompt = "Describe all visible objects, people, actions and events in detail."
    if "--prompt" in args:
        idx    = args.index("--prompt")
        prompt = args[idx + 1]

    from ingest import ingest_video
    connect()
    file_id = ingest_video(video, vlm_prompt=prompt)
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

COMMANDS = {
    "ingest":  cmd_ingest,
    "summary": cmd_summary,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0)
    COMMANDS[sys.argv[1]](sys.argv[2:])
