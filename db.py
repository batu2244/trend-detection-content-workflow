"""
db.py — Milvus collection definitions and shared helpers.

Collections:
    video_captions   chunk-level captions (already exists, owned by ingest.py)
    video_summaries  video-level summaries for semantic search and trend detection
"""

import time
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

SUMMARIES_COLLECTION = "video_summaries"
EMBED_DIM            = 1536


def ensure_summaries_collection() -> Collection:
    if utility.has_collection(SUMMARIES_COLLECTION):
        return Collection(SUMMARIES_COLLECTION)

    fields = [
        FieldSchema("id",                DataType.INT64,        is_primary=True, auto_id=True),
        FieldSchema("file_id",           DataType.VARCHAR,      max_length=64),
        FieldSchema("file_path",         DataType.VARCHAR,      max_length=512),
        FieldSchema("full_transcript",   DataType.VARCHAR,      max_length=65535),
        FieldSchema("summary",           DataType.VARCHAR,      max_length=4096),
        FieldSchema("platform",          DataType.VARCHAR,      max_length=64),
        FieldSchema("niche",             DataType.VARCHAR,      max_length=256),
        FieldSchema("topic",             DataType.VARCHAR,      max_length=256),
        FieldSchema("ingested_at",       DataType.INT64),
        FieldSchema("summary_embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ]
    schema = CollectionSchema(fields, description="Video-level summaries for search and trend detection")
    col = Collection(SUMMARIES_COLLECTION, schema)
    col.create_index(
        "summary_embedding",
        {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
    )
    return col


def insert_summary(
    file_id:           str,
    file_path:         str,
    full_transcript:   str,
    summary:           str,
    summary_embedding: list[float],
    platform:          str = "unknown",
    niche:             str = "unknown",
    topic:             str = "unknown",
    ingested_at:       int = None,
) -> None:
    """Insert (or replace) a video-level summary record in video_summaries."""
    col = ensure_summaries_collection()
    col.load()

    # Delete existing record for this file_id (upsert behaviour)
    existing = col.query(expr=f'file_id == "{file_id}"', output_fields=["id"], limit=1)
    if existing:
        ids = [r["id"] for r in existing]
        col.delete(f"id in {ids}")

    col.insert([
        [file_id],
        [file_path],
        [full_transcript[:65535]],
        [summary[:4096]],
        [platform],
        [niche],
        [topic],
        [ingested_at or int(time.time())],
        [summary_embedding],
    ])
    col.flush()
