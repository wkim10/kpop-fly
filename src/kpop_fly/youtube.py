"""Audio from a YouTube link, cached in ~/.cache/kpop-fly/youtube/<video id>.opus."""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Callable

from .brain import cache_dir

ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def normalize(link: str) -> str:
    """A full URL, a youtu.be/shorts link, or a bare 11-character video id."""
    return f"https://www.youtube.com/watch?v={link}" if ID.match(link) else link


def fetch_audio(link: str, log: Callable[[str], None] = print) -> tuple[Path, dict]:
    """Download (or reuse) a video's audio track. Returns the .opus path and basic video info."""
    from yt_dlp import YoutubeDL

    from .choreo import save_audio

    url = normalize(link)
    common = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True}
    with YoutubeDL(common) as y:
        info = y.extract_info(url, download=False)
    meta = {k: info.get(k) for k in ("id", "title", "channel", "duration", "webpage_url")}
    folder = cache_dir() / "youtube"
    audio, info_file = folder / f"{meta['id']}.opus", folder / f"{meta['id']}.json"
    if audio.exists():
        log(f"using cached audio for {meta['title']!r}")
        return audio, meta

    log(f"downloading audio: {meta['title']!r} ({meta['channel']})")
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kpop-fly-") as tmp:
        with YoutubeDL({**common, "format": "ba[acodec=opus]/ba", "outtmpl": str(Path(tmp) / "audio.%(ext)s")}) as y:
            src = Path(y.prepare_filename(y.extract_info(url, download=True)))
        save_audio(src, audio)
    info_file.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return audio, meta
