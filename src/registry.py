"""Append-only markdown registry of ingested videos at docs/video_registry.md."""

from datetime import datetime
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "docs" / "video_registry.md"

_HEADER = (
    "# Ingested Video Registry\n\n"
    "| file_id | url | title | duration_sec | timestamp |\n"
    "|---------|-----|-------|--------------|-----------|\n"
)


def _safe(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def _ensure_file() -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists() or REGISTRY_PATH.stat().st_size == 0:
        REGISTRY_PATH.write_text(_HEADER)


def append(file_id: str, url: str = "", title: str = "", duration: float | None = None) -> None:
    _ensure_file()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dur = f"{duration:.1f}" if isinstance(duration, (int, float)) else ""
    line = f"| {_safe(file_id)} | {_safe(url)} | {_safe(title)} | {dur} | {ts} |\n"
    with REGISTRY_PATH.open("a") as f:
        f.write(line)


def loaded_urls() -> set[str]:
    """Return the set of source URLs already recorded in the registry."""
    if not REGISTRY_PATH.exists():
        return set()
    urls: set[str] = set()
    for raw in REGISTRY_PATH.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        if cells[0] == "file_id" or cells[0].startswith("---"):
            continue
        url = cells[1]
        if url:
            urls.add(url)
    return urls
