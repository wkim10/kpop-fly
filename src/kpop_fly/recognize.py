"""Song recognition with AudD (https://audd.io): a short clip in, artist / title / position out.

The token is read from the AUDD_API_TOKEN environment variable and never logged.
"""
from __future__ import annotations

import io
import json
import os
import re
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

import numpy as np

AUDD_URL = "https://api.audd.io/"
TOKEN_ENV = "AUDD_API_TOKEN"


class AudDError(RuntimeError):
    pass


@dataclass
class Recognition:
    artist: str
    title: str
    timecode: float | None                  # seconds into AudD's recording of the song where the clip starts
    genres: list[str] = field(default_factory=list)
    song_link: str | None = None

    @property
    def name(self) -> str:
        return f"{self.artist} - {self.title}"

    @property
    def is_kpop(self) -> bool:
        return any(re.sub(r"[^a-z]", "", g.lower()) == "kpop" for g in self.genres)


def token() -> str | None:
    return os.environ.get(TOKEN_ENV) or None


def parse_timecode(text: str | None) -> float | None:
    """AudD timecodes look like "01:23" (or "1:02:03")."""
    if not text:
        return None
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return None
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def parse(response: dict) -> Recognition | None:
    """AudD's JSON -> Recognition, None for "not recognized"; raises AudDError on API errors."""
    if response.get("status") != "success":
        err = response.get("error") or {}
        raise AudDError(f"AudD error #{err.get('error_code', '?')}: {err.get('error_message', 'unknown error')}")
    result = response.get("result")
    if not result:
        return None
    apple = result.get("apple_music") or {}
    return Recognition(artist=result.get("artist", ""), title=result.get("title", ""),
                       timecode=parse_timecode(result.get("timecode")),
                       genres=list(apple.get("genreNames") or []), song_link=result.get("song_link"))


def _multipart(fields: dict[str, str], file_name: str, file_bytes: bytes, file_type: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
               f"Content-Type: {file_type}\r\n\r\n".encode())
    body.write(file_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    return body.getvalue(), f"multipart/form-data; boundary={boundary}"


def recognize(samples: np.ndarray, sr: int, api_token: str, timeout: float = 20.0) -> Recognition | None:
    """Send a mono clip (~10 s) to AudD. Network and API failures raise AudDError."""
    import soundfile as sf
    wav = io.BytesIO()
    sf.write(wav, np.asarray(samples, np.float32), sr, format="WAV", subtype="PCM_16")
    body, content_type = _multipart({"api_token": api_token, "return": "apple_music"}, "clip.wav", wav.getvalue(),
                                    "audio/wav")
    request = urllib.request.Request(AUDD_URL, data=body, headers={"Content-Type": content_type}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return parse(json.load(r))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise AudDError(f"couldn't reach AudD: {e}") from None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def saved_dance(rec: Recognition, songs: dict[str, dict]) -> str | None:
    """Which saved dance (slug -> meta with artist and title) a recognition is, if any. Titles must
    match exactly once case and punctuation are dropped ("Cheer Up" = "CHEER UP"); the saved artist
    must appear in AudD's (which may list features). Versions don't match on purpose: "FANCY
    (Japanese ver.)" has different vocals and can have different timing."""
    title, artist = _norm(rec.title), _norm(rec.artist)
    for slug, meta in songs.items():
        if _norm(meta.get("title", "")) == title and _norm(meta.get("artist", "")) in artist:
            return slug
    return None
