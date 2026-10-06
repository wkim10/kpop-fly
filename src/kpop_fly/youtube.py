"""Audio from a YouTube link, cached in ~/.cache/kpop-fly/youtube/<video id>.opus."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from .brain import cache_dir

ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def normalize(link: str) -> str:
    """A full URL, a youtu.be/shorts link, or a bare 11-character video id."""
    return f"https://www.youtube.com/watch?v={link}" if ID.match(link) else link


def ydl_options(**extra) -> dict:
    """Common yt-dlp options. YouTube now serves some formats only to clients that solve a
    JavaScript challenge (otherwise: HTTP 403); yt-dlp uses Deno by default, so point it at Node
    or Bun when Deno isn't installed."""
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True}
    if not shutil.which("deno"):
        for runtime in ("node", "bun"):
            if path := shutil.which(runtime):
                opts["js_runtimes"] = {runtime: {"path": path}}
                break
    return opts | extra


def download(url: str, fmt: str, outtmpl: str) -> tuple[dict, Path]:
    """Download one format; a failure becomes a readable SystemExit instead of a traceback."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    try:
        with YoutubeDL(ydl_options(format=fmt, outtmpl=outtmpl)) as y:
            info = y.extract_info(url, download=True)
            return info, Path(y.prepare_filename(info))
    except DownloadError as e:
        hint = "" if shutil.which("deno") or shutil.which("node") else (
            "\nYouTube may need a JavaScript runtime: `brew install deno` (or install Node).")
        raise SystemExit(f"couldn't download {url}: {e}{hint}\nIf this keeps happening, update yt-dlp: "
                         "`uv lock --upgrade-package yt-dlp && uv sync`") from None


def fetch_audio(link: str, log: Callable[[str], None] = print) -> tuple[Path, dict]:
    """Download (or reuse) a video's audio track. Returns the .opus path and basic video info."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    from .choreo import save_audio

    url = normalize(link)
    try:
        with YoutubeDL(ydl_options()) as y:
            info = y.extract_info(url, download=False)
    except DownloadError as e:
        raise SystemExit(f"couldn't open {url}: {e}") from None
    meta = {k: info.get(k) for k in ("id", "title", "channel", "duration", "webpage_url")}
    folder = cache_dir() / "youtube"
    audio, info_file = folder / f"{meta['id']}.opus", folder / f"{meta['id']}.json"
    if audio.exists():
        log(f"using cached audio for {meta['title']!r}")
        return audio, meta

    log(f"downloading audio: {meta['title']!r} ({meta['channel']})")
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kpop-fly-") as tmp:
        _, src = download(url, "ba[acodec=opus]/ba", str(Path(tmp) / "audio.%(ext)s"))
        save_audio(src, audio)
    info_file.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return audio, meta
