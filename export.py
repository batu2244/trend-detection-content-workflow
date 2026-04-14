"""
export.py — Export all ingested videos to a CSV with full transcripts,
GPT-4o-mini summaries, and summary embeddings (for smart search).

CSV columns:
    file_id             unique video identifier
    file_path           original video file path
    full_transcript     all Whisper segments concatenated in order
    summary             GPT-4o-mini summary of the full video
    summary_embedding   JSON-encoded 1536-dim float vector of the summary
"""

import os
import csv
import json
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import Collection
from tqdm import tqdm

from summarize import summarize_video, detect_niche_topic
from db import insert_summary, ensure_summaries_collection

load_dotenv()

COLLECTION_NAME = "video_captions"
EMBED_MODEL     = "text-embedding-3-small"
CSV_PATH        = "video_database.csv"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def fetch_all_file_ids() -> list[str]:
    col = Collection(COLLECTION_NAME)
    col.load()
    results = col.query(
        expr='file_id != ""',
        output_fields=["file_id"],
        limit=10000,
    )
    return sorted({r["file_id"] for r in results})


def fetch_chunks(file_id: str) -> list[dict]:
    col = Collection(COLLECTION_NAME)
    col.load()
    results = col.query(
        expr=f'file_id == "{file_id}"',
        output_fields=["chunk_index", "start_sec", "end_sec", "transcript", "file_path"],
        limit=500,
    )
    return sorted(results, key=lambda x: x["chunk_index"])


def build_full_transcript(chunks: list[dict]) -> str:
    parts = [c["transcript"].strip() for c in chunks if c["transcript"].strip()]
    return " ".join(parts)


def embed_text(text: str) -> list[float]:
    return client.embeddings.create(
        model=EMBED_MODEL, input=text
    ).data[0].embedding


def fetch_already_summarized() -> set[str]:
    """Return file_ids already present in video_summaries."""
    col = ensure_summaries_collection()
    col.load()
    results = col.query(expr='file_id != ""', output_fields=["file_id"], limit=10000)
    return {r["file_id"] for r in results}


def export_all(csv_path: str = CSV_PATH) -> None:
    file_ids   = fetch_all_file_ids()
    if not file_ids:
        print("No ingested videos found in Milvus.")
        return

    already_done = fetch_already_summarized()
    new_ids      = [fid for fid in file_ids if fid not in already_done]

    print(f"Found {len(file_ids)} video(s): {len(already_done)} already summarized, {len(new_ids)} to process.\n")

    rows = []
    for file_id in tqdm(file_ids, desc="Videos"):
        chunks = fetch_chunks(file_id)
        file_path       = chunks[0]["file_path"] if chunks else ""
        full_transcript = build_full_transcript(chunks)

        if file_id in already_done:
            # Pull existing summary from video_summaries instead of regenerating
            col = ensure_summaries_collection()
            col.load()
            existing = col.query(
                expr=f'file_id == "{file_id}"',
                output_fields=["summary", "niche", "topic", "summary_embedding"],
                limit=1,
            )
            if existing:
                summary           = existing[0]["summary"]
                niche             = existing[0]["niche"]
                topic             = existing[0]["topic"]
                summary_embedding = existing[0]["summary_embedding"]
                rows.append({
                    "file_id":           file_id,
                    "file_path":         file_path,
                    "full_transcript":   full_transcript,
                    "niche":             niche,
                    "topic":             topic,
                    "summary":           summary,
                    "summary_embedding": json.dumps(summary_embedding),
                })
                continue

        summary           = summarize_video(file_id)
        niche, topic      = detect_niche_topic(summary)
        summary_embedding = embed_text(summary)

        insert_summary(
            file_id=file_id,
            file_path=file_path,
            full_transcript=full_transcript,
            summary=summary,
            summary_embedding=summary_embedding,
            niche=niche,
            topic=topic,
        )

        rows.append({
            "file_id":           file_id,
            "file_path":         file_path,
            "full_transcript":   full_transcript,
            "niche":             niche,
            "topic":             topic,
            "summary":           summary,
            "summary_embedding": json.dumps(summary_embedding),
        })

    fieldnames = ["file_id", "file_path", "full_transcript", "niche", "topic", "summary", "summary_embedding"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExported {len(rows)} row(s) → {csv_path}")
