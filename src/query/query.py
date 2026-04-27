"""
query.py — Ask questions about ingested videos using semantic search + GPT-4o-mini.

Pipeline:
  1. Embed the user's question with text-embedding-3-small
  2. Vector search video_captions for the most relevant chunks
  3. Feed chunks to GPT-4o-mini to generate an answer
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import Collection

from src.log_setup import get_logger

load_dotenv()
logger = get_logger("query")

CAPTIONS_COLLECTION  = "video_captions"
SUMMARIES_COLLECTION = "video_summaries"
EMBED_MODEL          = "text-embedding-3-small"
LLM_MODEL            = "gpt-4o-mini"
TOP_K                = 5

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed_question(question: str) -> list[float]:
    return client.embeddings.create(
        model=EMBED_MODEL, input=question
    ).data[0].embedding


def search_all_videos(embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    """Search video_summaries — one result per video, spreads across all videos."""
    col = Collection(SUMMARIES_COLLECTION)
    col.load()

    results = col.search(
        data=[embedding],
        anns_field="summary_embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["file_id", "summary", "niche", "topic"],
    )

    chunks = []
    for hit in results[0]:
        chunks.append({
            "file_id": hit.entity.get("file_id"),
            "caption": hit.entity.get("summary"),
            "topic":   hit.entity.get("topic"),
            "niche":   hit.entity.get("niche"),
            "score":   hit.score,
        })
    return chunks


def search_single_video(embedding: list[float], file_id: str, top_k: int = TOP_K) -> list[dict]:
    """Search video_captions for detailed chunk-level results within one video."""
    col = Collection(CAPTIONS_COLLECTION)
    col.load()

    results = col.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        expr=f'file_id == "{file_id}"',
        output_fields=["file_id", "start_sec", "end_sec", "caption", "transcript"],
    )

    chunks = []
    for hit in results[0]:
        chunks.append({
            "file_id":    hit.entity.get("file_id"),
            "start_sec":  hit.entity.get("start_sec"),
            "end_sec":    hit.entity.get("end_sec"),
            "caption":    hit.entity.get("caption"),
            "transcript": hit.entity.get("transcript"),
            "score":      hit.score,
        })
    return chunks


def answer_question(question: str, chunks: list[dict], scoped: bool = False) -> str:
    if scoped:
        context = "\n\n".join(
            f"[{c['start_sec']:.1f}s-{c['end_sec']:.1f}s]\n"
            f"Visual: {c['caption']}\nAudio: {c['transcript']}"
            for c in chunks
        )
    else:
        context = "\n\n".join(
            f"Video {c['file_id']} (topic: {c.get('topic', '')}):\n{c['caption']}"
            for c in chunks
        )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a video analysis assistant. Answer the user's question "
                    "using only the provided video content. Reference video IDs or timestamps "
                    "when relevant. If the answer is not in the content, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Video content:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_tokens=600,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def run_query(question: str, file_id: str = None) -> None:
    logger.info("Query question=%r file_id=%s", question, file_id)
    print(f"\n[query] Question: {question}")

    embedding = embed_question(question)

    if file_id:
        print(f"[query] Scoped to file_id: {file_id}")
        chunks = search_single_video(embedding, file_id=file_id)
        scoped = True
    else:
        chunks = search_all_videos(embedding)
        scoped = False

    if not chunks:
        print("No relevant content found.")
        return

    answer = answer_question(question, chunks, scoped=scoped)
    print(f"\nAnswer:\n{answer}")
