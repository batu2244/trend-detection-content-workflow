"""
ingest.py - Video ingestion pipeline with ffmpeg audio + vision captioning.
Pipeline:
  1. ffmpeg to extract audio (WAV) from video
  2. OpenAI Whisper to transcribe audio with word-level timestamps
  3. ffmpeg to extract one keyframe per chunk interval
  4. GPT-4o vision to caption each frame
  5. Merge transcript segments with visual captions per chunk
  6. OpenAI text-embedding-3-small to embed merged captions
  7. Milvus to store embeddings + metadata
"""

import os
import uuid
import base64
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, utility,
)

load_dotenv()

# config 

MILVUS_URI      = "./milvus_local.db"
COLLECTION_NAME = "video_captions"
EMBED_MODEL     = "text-embedding-3-small"  # OpenAI embedding, no local model needed
EMBED_DIM       = 1536
VLM_MODEL       = "gpt-4o"
WHISPER_MODEL   = "whisper-1"
LLM_MODEL       = "gpt-4o-mini"
CHUNK_SEC       = 5                          # seconds per chunk (default)
SAMPLE_RATE     = 16000                      # whisper wants 16 kHz mono

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Milvus schema 

def ensure_collection() -> Collection:
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema("id",          DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema("file_id",     DataType.VARCHAR,       max_length=64),
        FieldSchema("file_path",   DataType.VARCHAR,       max_length=512),
        FieldSchema("chunk_index", DataType.INT64),
        FieldSchema("start_sec",   DataType.FLOAT),
        FieldSchema("end_sec",     DataType.FLOAT),
        FieldSchema("caption",     DataType.VARCHAR,       max_length=4096),
        FieldSchema("transcript",  DataType.VARCHAR,       max_length=4096),
        FieldSchema("embedding",   DataType.FLOAT_VECTOR,  dim=EMBED_DIM),
    ]
    schema = CollectionSchema(fields, description="Video chunk captions + audio")
    col    = Collection(COLLECTION_NAME, schema)
    col.create_index(
        "embedding",
        {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
    )
    return col


# ffmpeg helpers 

def _run(cmd: list[str], desc: str = "") -> None:
    """Run a subprocess command, raising on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg error{' (' + desc + ')' if desc else ''}:\n{result.stderr}"
        )


def extract_audio(video_path: str, out_wav: str) -> None:
    """Extract mono 16 kHz WAV from video using ffmpeg."""
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                        # no video
        "-ac", "1",                   # mono
        "-ar", str(SAMPLE_RATE),      # 16 kHz
        "-acodec", "pcm_s16le",       # PCM 16-bit
        out_wav,
    ], "extract audio")


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error:\n{result.stderr}")
    return float(result.stdout.strip())


def extract_frame(video_path: str, timestamp: float, out_jpg: str) -> None:
    """Extract a single frame at `timestamp` seconds from the video."""
    _run([
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "3",                  # JPEG quality (lower = better)
        out_jpg,
    ], f"extract frame at {timestamp:.1f}s")


# Whisper transcription 

def transcribe_audio(wav_path: str) -> list[dict]:
    """
    Send WAV to Whisper and return verbose_json segments:
    [{"start": float, "end": float, "text": str}, ...]
    """
    with open(wav_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    segments = []
    for seg in response.segments:
        segments.append({
            "start": seg.start,
            "end":   seg.end,
            "text":  seg.text.strip(),
        })
    return segments


def segments_for_chunk(
    segments: list[dict], start: float, end: float
) -> str:
    """Collect transcript text that overlaps with [start, end)."""
    texts = []
    for seg in segments:
        # Include segment if it overlaps with this chunk window
        if seg["end"] > start and seg["start"] < end:
            texts.append(seg["text"])
    return " ".join(texts).strip()


# GPT-4o vision captioning 

def caption_frame(jpg_path: str, prompt: str, transcript_hint: str = "") -> str:
    """Caption a JPEG frame with GPT-4o vision."""
    with open(jpg_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    system = "You are a video analysis assistant. Be specific and concise."
    user_text = prompt
    if transcript_hint:
        user_text += f"\n\nAudio transcript for this segment: \"{transcript_hint}\""

    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",       "text": user_text},
                {"type": "image_url",  "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def merge_caption_transcript(visual: str, transcript: str) -> str:
    """Merge visual caption and audio transcript into a single rich description."""
    if not transcript:
        return visual
    if not visual:
        return transcript
    return f"{visual} [Audio: {transcript}]"


#===============================================================
# main ingestion 

def ingest_video(
    video_path: str,
    chunk_sec: int = CHUNK_SEC,
    vlm_prompt: str = "Describe all visible objects, people, actions and events in detail.",
    platform: str = "unknown",
) -> str:
    """
    Ingest a video file into Milvus.

    Returns:
        file_id (str): unique identifier for this video, needed for summary/QA.
    """
    video_path = str(Path(video_path).resolve())
    file_id    = str(uuid.uuid4())[:8]

    print(f"\n[ingest] file_id : {file_id}")
    print(f"[ingest] video   : {video_path}")
    print(f"[ingest] chunk   : {chunk_sec}s")

    # Get video duration
    duration = get_video_duration(video_path)
    n_chunks = max(1, int(np.ceil(duration / chunk_sec)))
    print(f"[ingest] duration: {duration:.1f}s  →  {n_chunks} chunks")

    with tempfile.TemporaryDirectory() as tmpdir:

        # Extract audio with ffmpeg
        wav_path = os.path.join(tmpdir, "audio.wav")
        print("[ingest] Extracting audio with ffmpeg...")
        extract_audio(video_path, wav_path)

        # Transcribe audio with Whisper
        print("[ingest] Transcribing audio with Whisper...")
        try:
            segments = transcribe_audio(wav_path)
            print(f"[ingest] Whisper returned {len(segments)} segments")
        except Exception as e:
            print(f"[ingest] Whisper failed ({e}), continuing without transcript")
            segments = []

        # Ensure collection exists (connection already established by caller)
        col = ensure_collection()

        # Process each chunk
        records = {
            "file_id":     [],
            "file_path":   [],
            "chunk_index": [],
            "start_sec":   [],
            "end_sec":     [],
            "caption":     [],
            "transcript":  [],
            "embedding":   [],
        }

        for i in tqdm(range(n_chunks), desc="Chunks", unit="chunk"):
            start = i * chunk_sec
            end   = min(start + chunk_sec, duration)
            mid   = (start + end) / 2.0

            # Extract frame at chunk midpoint
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
            extract_frame(video_path, mid, frame_path)

            # Get transcript for this time window
            transcript_text = segments_for_chunk(segments, start, end)

            # Caption the frame (with transcript hint)
            visual_caption = caption_frame(frame_path, vlm_prompt, transcript_text)

            # Merge into final caption
            caption = merge_caption_transcript(visual_caption, transcript_text)

            # Embedding
            embedding = client.embeddings.create(
                model=EMBED_MODEL, input=caption
            ).data[0].embedding

            records["file_id"].append(file_id)
            records["file_path"].append(video_path)
            records["chunk_index"].append(i)
            records["start_sec"].append(float(start))
            records["end_sec"].append(float(end))
            records["caption"].append(caption[:4096])
            records["transcript"].append(transcript_text[:4096])
            records["embedding"].append(embedding)

        # Insert into Milvus
        col.insert([
            records["file_id"],
            records["file_path"],
            records["chunk_index"],
            records["start_sec"],
            records["end_sec"],
            records["caption"],
            records["transcript"],
            records["embedding"],
        ])
        col.flush()

    print(f"\n[ingest] Done. Inserted {n_chunks} chunks for file_id={file_id}")

    # Build full transcript from all chunks
    full_transcript = " ".join(
        t for t in records["transcript"] if t.strip()
    )

    # Generate summary, detect niche/topic, embed, store in video_summaries
    from summarize import summarize_video, detect_niche_topic
    from db import insert_summary

    print("[ingest] Generating summary...")
    summary = summarize_video(file_id)

    print("[ingest] Detecting niche and topic...")
    niche, topic = detect_niche_topic(summary)
    print(f"[ingest] niche={niche!r}  topic={topic!r}")

    print("[ingest] Embedding summary...")
    summary_embedding = client.embeddings.create(
        model=EMBED_MODEL, input=summary
    ).data[0].embedding

    insert_summary(
        file_id=file_id,
        file_path=video_path,
        full_transcript=full_transcript,
        summary=summary,
        summary_embedding=summary_embedding,
        platform=platform,
        niche=niche,
        topic=topic,
    )
    print(f"[ingest] Summary stored in video_summaries for file_id={file_id}")

    return file_id



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <video.mp4> [chunk_sec] [--prompt 'text']")
        sys.exit(1)

    video  = sys.argv[1]
    chunk  = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else CHUNK_SEC
    prompt = "Describe all visible objects, people, actions and events in detail."

    if "--prompt" in sys.argv:
        idx    = sys.argv.index("--prompt")
        prompt = sys.argv[idx + 1]

    fid = ingest_video(video, chunk_sec=chunk, vlm_prompt=prompt)
    print(f"\nSave this file_id: {fid}")
