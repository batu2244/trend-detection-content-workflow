"""
summarize.py — retrieve captions for a file_id and produce a
GPT-4o-mini summary of the full video.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import connections, Collection

load_dotenv()

MILVUS_URI      = "./milvus_local.db"
COLLECTION_NAME = "video_captions"
LLM_MODEL       = "gpt-4o-mini"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def connect_milvus():
    connections.connect(uri=MILVUS_URI)


def fetch_captions(file_id: str) -> list[dict]:
    col = Collection(COLLECTION_NAME)
    col.load()
    results = col.query(
        expr=f'file_id == "{file_id}"',
        output_fields=["chunk_index", "start_sec", "end_sec", "caption"],
        limit=500,
    )
    return sorted(results, key=lambda x: x["chunk_index"])


def summarize_video(
    file_id: str,
    focus: str = "general",
    output_format: str = "3-5 sentences covering main subject, key actions, and outcome",
) -> str:
    """
    Pull all captions for file_id and ask GPT-4o-mini to produce a final summary.
    """
    chunks = fetch_captions(file_id)
    if not chunks:
        raise ValueError(f"No captions found for file_id: {file_id}")

    # Build timestamped caption list
    caption_block = "\n".join(
        f"[{c['start_sec']:.1f}s-{c['end_sec']:.1f}s]: {c['caption']}"
        for c in chunks
    )

    prompt = f"""You are analyzing a video. Below are timestamped descriptions of each segment.

Focus area: {focus}

Segment descriptions:
{caption_block}

Produce a summary in this format: {output_format}.
Be specific about what happens, when, and who is involved."""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def detect_niche_topic(summary: str) -> tuple[str, str]:
    """
    Ask GPT-4o-mini to infer the niche and specific topic from a video summary.
    Returns (niche, topic) e.g. ("cooking", "sourdough bread recipe").
    """
    import json as _json
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Analyze this video summary and return a JSON object with exactly two keys:\n"
                '- "niche": the broad content category (e.g. "cooking", "fitness", "finance", "tech", "beauty", "travel")\n'
                '- "topic": the specific subject (e.g. "sourdough bread recipe", "home workout for beginners")\n\n'
                f"Summary: {summary}\n\n"
                "Return only valid JSON, nothing else."
            ),
        }],
        max_tokens=100,
        temperature=0,
    )
    try:
        data = _json.loads(resp.choices[0].message.content.strip())
        return data.get("niche", "unknown"), data.get("topic", "unknown")
    except Exception:
        return "unknown", "unknown"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python summarize.py <file_id>")
        sys.exit(1)

    connect_milvus()
    fid = sys.argv[1]
    print(f"\nSummary:\n{summarize_video(fid)}")
