import os
import sys
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from src.log_setup import get_logger

load_dotenv()

API_KEY = os.environ["YOUTUBE_DATA_API_KEY"]
logger  = get_logger("downloader.search")


def log(msg):
    print(msg)
    logger.info(msg)


def get_published_after(days):
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def search_top_videos(query, max_results=10):
    log(f"Searching YouTube for: '{query}'")

    params = urllib.parse.urlencode({
        "part": "snippet",
        "q": query,
        "order": "viewCount",
        "maxResults": max_results,
        "type": "video",
        "videoDuration": "short",   # YouTube API "short" = duration < 4 min — restricts results to short-form videos
        "publishedAfter": get_published_after(21),   # 3 weeks ago
        "relevanceLanguage": "en",   # ranking hint only — YouTube has no hard language filter on search.list
        "key": API_KEY
    })

    url = f"https://www.googleapis.com/youtube/v3/search?{params}"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        error_body = json.loads(e.read().decode())
        log(f"API Error: {error_body['error']['message']}")
        sys.exit(1)

    items = data.get("items", [])
    if not items:
        log("No results found.")
        sys.exit(0)

    videos = []
    log(f"Found {len(items)} videos:")
    for i, item in enumerate(items, 1):
        video_id = item["id"]["videoId"]
        title    = item["snippet"]["title"]
        channel  = item["snippet"]["channelTitle"]
        url      = f"https://www.youtube.com/watch?v={video_id}"
        videos.append({"video_id": video_id, "title": title, "channel": channel, "url": url})
        log(f"  {i}. {title} — {channel}")
        log(f"     {url}")

    return videos


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.downloader.URL_generation \"your search query\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    for v in search_top_videos(query):
        print(v["url"])
