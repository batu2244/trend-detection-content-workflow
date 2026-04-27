# TODO Tracker

| Task | Description | Status |
|------|-------------|--------|
| Project restructure | Move flat scripts (ingest.py, summarize.py, query.py, trend.py) into `src/` package with submodules | Completed |
| CLI entrypoint | Unified `run.py` dispatcher for `ingest`, `ingest_all`, `summary`, `query`, `trend`, `download`, `--trendpilot` | Completed |
| Milvus schema (captions) | `video_captions` collection with chunk-level embeddings (IVF_FLAT, COSINE) | Completed |
| Milvus schema (summaries) | `video_summaries` collection with summary embeddings, niche, topic, platform (`db.py`) | Completed |
| Audio extraction | ffmpeg mono/16kHz WAV extraction | Completed |
| Transcription | OpenAI Whisper with segment-level timestamps | Completed |
| Frame extraction | ffmpeg keyframe-per-chunk at chunk midpoint | Completed |
| Vision captioning | GPT-4o captioning of frames with transcript hint | Completed |
| Caption embedding | `text-embedding-3-small` embeddings into `video_captions` | Completed |
| Video summarization | GPT-4o-mini summary over timestamped chunk captions | Completed |
| Niche/topic detection | GPT-4o-mini JSON classification of niche + specific topic | Completed |
| Summary embedding storage | Summary + embedding persisted to `video_summaries` during ingest | Completed |
| Batch ingestion | `ingest_all` over `videos/` directory with common video extensions | Completed |
| Q&A over videos | Semantic search across all videos or scoped to a `file_id` + GPT-4o-mini answer | Completed |
| YouTube URL search | `URL_generation.search_top_videos` for query-based discovery | Completed |
| YouTube downloader | yt-dlp download + auto-ingest pipeline | Completed |
| Trend clustering | HDBSCAN clustering of summary embeddings per niche (fallback for <4 videos) | Completed |
| Trend detection | GPT-4o-mini identifies trending topic + reason per cluster | Completed |
| Content brief generation | GPT-4o-mini generates title, hook, key points, format per trend | Completed |
| Activity/todo docs | `docs/log.md` + `docs/todo.md` scaffolding | Completed |
| Logging | Structured logging to `yt_digest.log` via `src/log_setup.py` across pipeline stages | Completed |
| Video registry md | Append-only `docs/video_registry.md` row per ingested video: `file_id`, url, title, duration, timestamp | Completed |
| Shorts-only download | Search restricted to short videos via YouTube API `videoDuration=short` parameter | Completed |
| Auto-cleanup after ingest | Local video file deleted once embeddings are written to Milvus and the registry md row is logged | Completed |
| Trendpilot orchestrator | `python run.py --trendpilot --topic <str> [--max <n>]` runs search → download (shorts) → ingest+summarize → registry+cleanup → trend analysis on dominant detected niche | Completed |
