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

load_dotenv()

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
        try:
            data = json.loads(resp.choices[0].message.content.strip())
            trends.append(data)
        except Exception:
            trends.append({"trend": "unknown", "why": ""})
    return trends


def generate_brief(niche: str, trend: str, why: str) -> dict:
    """Generate a content brief for a given trend."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
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
    try:
        return json.loads(resp.choices[0].message.content.strip())
    except Exception:
        return {}


def run_trend(niche: str) -> None:
    print(f"\n[trend] Fetching summaries for niche: {niche!r}")
    summaries = fetch_summaries(niche)
    if not summaries:
        print(f"No videos found for niche '{niche}'. Ingest some videos first.")
        return

    print(f"[trend] Found {len(summaries)} video(s). Clustering...")
    clusters = cluster_summaries(summaries)
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
