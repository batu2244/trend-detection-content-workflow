"""
trend.py — Trend detection and content brief generation.

Pipeline:
  1. Fetch all video summaries for a given niche from video_summaries
  2. Cluster summary embeddings with HDBSCAN (or skip if too few videos)
  3. Feed clusters to GPT-4o-mini to identify trending topics
  4. Generate a content brief (title, hook, key points, format)
"""

import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import Collection

from db import ensure_summaries_collection

from src.log_setup import get_logger

load_dotenv()
logger = get_logger("trend")

LLM_MODEL   = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def fetch_summaries(niche: str) -> list[dict]:
    """Fetch all video summaries for a given niche."""
    col = ensure_summaries_collection()
    col.load()
    results = col.query(
        expr=f'niche == "{niche}"',
        output_fields=["file_id", "topic", "summary", "summary_embedding"],
        limit=1000,
    )
    return results


def cluster_summaries(summaries: list[dict]) -> list[list[dict]]:
    """
    Cluster summaries by embedding similarity using HDBSCAN.
    Falls back to a single cluster if too few videos.
    """
    if len(summaries) < 4:
        return [summaries]

    import hdbscan
    embeddings = np.array([s["summary_embedding"] for s in summaries])
    labels = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean").fit_predict(embeddings)

    clusters = {}
    for label, summary in zip(labels, summaries):
        clusters.setdefault(label, []).append(summary)

    # -1 = noise points, put them in their own group
    return list(clusters.values())


def detect_trends(niche: str, clusters: list[list[dict]]) -> list[dict]:
    """Ask GPT-4o-mini to identify trending topics from each cluster."""
    trends = []
    for cluster in clusters:
        summaries_text = "\n".join(
            f"- Topic: {s['topic']}\n  Summary: {s['summary']}" for s in cluster
        )
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": (
                    f"You are analyzing a group of {niche} videos that share a common theme.\n\n"
                    f"Videos:\n{summaries_text}\n\n"
                    "Identify the trending topic in this group. Return a JSON object with:\n"
                    '- "trend": the trending topic (1 short phrase)\n'
                    '- "why": why this is trending (1-2 sentences)\n'
                    "Return only valid JSON."
                ),
            }],
            max_tokens=200,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            trends.append(json.loads(raw))
        except Exception as e:
            logger.warning("detect_trends JSON parse failed: %s | raw=%r", e, raw)
            trends.append({"trend": "unknown", "why": ""})
    return trends


def generate_brief(niche: str, trend: str, why: str) -> dict:
    """Generate a content brief for a given trend."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": (
                f"You are a content strategist for {niche} videos.\n\n"
                f"Trending topic: {trend}\n"
                f"Why it's trending: {why}\n\n"
                "Generate a content brief as a JSON object with:\n"
                '- "title": a compelling video title\n'
                '- "hook": opening line to grab attention (1 sentence)\n'
                '- "key_points": list of 3-5 main points to cover\n'
                '- "format": recommended video format and length\n'
                "Return only valid JSON."
            ),
        }],
        max_tokens=400,
        temperature=0.7,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning("generate_brief JSON parse failed: %s | raw=%r", e, raw)
        return {}


def run_trend(niche: str) -> None:
    logger.info("Trend run niche=%r", niche)
    print(f"\n[trend] Fetching summaries for niche: {niche!r}")
    summaries = fetch_summaries(niche)
    if not summaries:
        logger.warning("No videos found for niche=%r", niche)
        print(f"No videos found for niche '{niche}'. Ingest some videos first.")
        return

    print(f"[trend] Found {len(summaries)} video(s). Clustering...")
    clusters = cluster_summaries(summaries)
    logger.info("Clustered %d summaries into %d cluster(s)", len(summaries), len(clusters))
    print(f"[trend] {len(clusters)} cluster(s) found. Detecting trends...")
    trends = detect_trends(niche, clusters)

    for i, trend in enumerate(trends, 1):
        trend_name = trend.get("trend", "unknown")
        why        = trend.get("why", "")

        print(f"\n{'='*60}")
        print(f"Trend #{i}: {trend_name}")
        print(f"Why:       {why}")

        brief = generate_brief(niche, trend_name, why)
        if brief:
            print(f"\nContent Brief:")
            print(f"  Title:      {brief.get('title', '')}")
            print(f"  Hook:       {brief.get('hook', '')}")
            print(f"  Key points: {brief.get('key_points', [])}")
            print(f"  Format:     {brief.get('format', '')}")

    print(f"\n{'='*60}")


def fetch_all_niches() -> list[str]:
    """Return distinct niches currently stored in video_summaries."""
    col = ensure_summaries_collection()
    col.load()
    rows = col.query(expr='niche != ""', output_fields=["niche"], limit=10000)
    return sorted({r["niche"] for r in rows if r.get("niche")})


def run_clusters_all() -> None:
    """Run `run_clusters` once per niche present in the database."""
    niches = fetch_all_niches()
    if not niches:
        print("No videos have been ingested yet.")
        return
    print(f"[clusters] Niches in database: {niches}")
    for niche in niches:
        print(f"\n{'#'*60}\n# Niche: {niche}\n{'#'*60}")
        run_clusters(niche)


def run_clusters(niche: str) -> None:
    """Print clusters + their member video summaries for a niche.

    Skips the content-brief step from `run_trend`. Useful for inspecting how
    summaries are grouped before deciding what to write about.
    """
    logger.info("Clusters run niche=%r", niche)
    print(f"\n[clusters] Fetching summaries for niche: {niche!r}")
    summaries = fetch_summaries(niche)
    if not summaries:
        logger.warning("No videos found for niche=%r", niche)
        print(f"No videos found for niche '{niche}'. Ingest some videos first.")
        return

    print(f"[clusters] Found {len(summaries)} video(s). Clustering...")
    clusters = cluster_summaries(summaries)
    logger.info("Clustered %d summaries into %d cluster(s)", len(summaries), len(clusters))
    print(f"[clusters] {len(clusters)} cluster(s) found. Labeling...")
    trends = detect_trends(niche, clusters)

    for i, (trend, cluster) in enumerate(zip(trends, clusters), 1):
        label = trend.get("trend", "unknown")
        why   = trend.get("why", "")

        print(f"\n{'='*60}")
        print(f"Cluster #{i}  ({len(cluster)} video{'s' if len(cluster) != 1 else ''})")
        print(f"  Trend: {label}")
        if why:
            print(f"  Why:   {why}")

        for s in cluster:
            print(f"\n  - file_id: {s.get('file_id', '')}")
            print(f"    topic:   {s.get('topic', '')}")
            print(f"    summary: {s.get('summary', '')}")

    print(f"\n{'='*60}")
