"""Clip studio: YouTube URLs -> reusable highlight clips and still frames.

The pipeline behind the *Clip studio* tab:

    download_video()    yt-dlp fetches the source video (up to 1080p, mp4)
    fetch_transcript()  manual or automatic subtitles as "[mm:ss] text" lines
    suggest_segments()  an OpenRouter model picks self-contained highlight windows
    cut_clip()          ffmpeg stream-copies each segment into a reusable mp4
    extract_frames()    ffmpeg dumps still images from the cut clips

Everything lands in the session's ``clip_library`` and can be reused as B-roll
from the *Video* tab. yt-dlp is an optional extra (``pip install
"ai-voiceover-studio[clip]"``); ffmpeg is required for cutting and frames.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .errors import ConfigError

USER_AGENT = "ai-voiceover-studio (+https://github.com/yanghhu/ai-voiceover-studio)"
SUBTITLE_LANGS = ("it", "en", "es", "fr", "de", "pt")
MAX_TRANSCRIPT_CHARS = 12000

CLIP_SYSTEM = (
    "You pick the most reusable highlight segments from a video transcript. Respond with ONLY "
    "a JSON array, no markdown fences, no commentary:\n"
    '[{{"start": <seconds number>, "end": <seconds number>, '
    '"title": "short clip title", "reason": "why other videos can reuse it"}}]\n'
    "Pick {n} segments of about {clip} seconds each. They must not overlap, must stay inside "
    "the video, and must be self-contained: they will be reused as B-roll in other videos, so "
    "avoid segments that depend on earlier context."
)


def _import_yt_dlp():
    try:
        import yt_dlp  # type: ignore
    except ImportError:
        raise ConfigError("The clip studio needs yt-dlp: "
                          "pip install \"ai-voiceover-studio[clip]\"") from None
    return yt_dlp


def _cookie_opts(cookies: Optional[str] = None) -> Dict[str, Any]:
    """yt-dlp cookie options from an explicit file, else from the environment."""
    cookie_file = (cookies or "").strip() or os.environ.get("VOICEOVER_YTDLP_COOKIES", "")
    if cookie_file:
        return {"cookiefile": cookie_file}
    browser = (os.environ.get("VOICEOVER_YTDLP_COOKIES_FROM_BROWSER", "") or "").strip()
    if browser:
        return {"cookiesfrombrowser": (browser.split(":")[0],)}
    return {}


def _download_hint(url: str, exc: Exception) -> str:
    """One-line failure report; bot-check errors point to the cookie options."""
    message = str(exc)
    if "Sign in to confirm" in message or "bot" in message.lower():
        return ("%s: YouTube asked for sign-in (bot check). Export your browser cookies to a "
                "cookies.txt file (see https://github.com/yt-dlp/yt-dlp/wiki/Extractors#"
                "exporting-youtube-cookies) and set VOICEOVER_YTDLP_COOKIES=/path/cookies.txt, "
                "or VOICEOVER_YTDLP_COOKIES_FROM_BROWSER=chrome. Detail: %s"
                % (url, message[:160]))
    return "%s: %s" % (url, message[:200])


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", (text or "clip").lower()).strip("-")
    return (slug or "clip")[:40]


def _mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return "%02d:%02d" % (seconds // 60, seconds % 60)


def download_video(url: str, dest_dir: Path,
                   cookies: Optional[str] = None) -> Tuple[Path, Dict[str, Any]]:
    """Download one YouTube URL (<=1080p mp4). Returns (path, info dict)."""
    yt_dlp = _import_yt_dlp()
    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {"format": ("bv*[height<=1080][ext=mp4]+ba[ext=m4a]/"
                       "b[height<=1080][ext=mp4]/bv*+ba/b"),
            "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
            "merge_output_format": "mp4", "noplaylist": True,
            "quiet": True, "no_warnings": True, **_cookie_opts(cookies)}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True) or {}
        filepath = ((info.get("requested_downloads") or [{}])[0].get("filepath")
                    or ydl.prepare_filename(info))
    path = Path(filepath)
    if not path.is_file():
        candidates = sorted(dest_dir.glob("%s.*" % info.get("id", path.stem)))
        if not candidates:
            raise ConfigError("Download produced no file for %s" % url)
        path = candidates[0]
    return path, info


def _vtt_seconds(stamp: str) -> float:
    parts = stamp.strip().replace(",", ".").split(":")
    try:
        return float(parts[-1]) + 60.0 * float(parts[-2]) + (3600.0 * float(parts[-3]) if len(parts) > 2 else 0.0)
    except (ValueError, IndexError):
        return 0.0


def parse_vtt(text: str) -> str:
    """VTT subtitles -> "[mm:ss] text" lines, one per cue, duplicates removed."""
    cues: List[Tuple[float, str]] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        stamp_lines = [line for line in lines if "-->" in line]
        if not stamp_lines:
            continue
        start = _vtt_seconds(stamp_lines[0].split("-->")[0])
        after = lines.index(stamp_lines[0]) + 1
        body = " ".join(lines[after:])
        body = re.sub(r"<[^>]+>", "", body).strip()
        if body:
            cues.append((start, body))
    merged: List[Tuple[float, str]] = []
    for start, body in cues:
        if merged and merged[-1][1] == body:
            continue
        merged.append((start, body))
    return "\n".join("[%s] %s" % (_mmss(start), body) for start, body in merged)


def fetch_transcript(url: str, dest_dir: Path, cookies: Optional[str] = None) -> str:
    """Download manual or automatic subtitles and return them as timestamped text."""
    yt_dlp = _import_yt_dlp()
    dest_dir.mkdir(parents=True, exist_ok=True)
    before = set(dest_dir.glob("*.vtt"))
    opts = {"skip_download": True, "writesubtitles": True, "writeautomaticsub": True,
            "subtitleslangs": list(SUBTITLE_LANGS), "subtitlesformat": "vtt",
            "outtmpl": str(dest_dir / "%(id)s.%(ext)s"), "quiet": True, "no_warnings": True,
            **_cookie_opts(cookies)}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception:
        pass
    def lang_rank(path: Path) -> int:
        lang = path.name.split(".")[-2]
        return SUBTITLE_LANGS.index(lang) if lang in SUBTITLE_LANGS else 99

    new = sorted(set(dest_dir.glob("*.vtt")) - before, key=lang_rank)
    if not new:
        return ""
    return parse_vtt(new[0].read_text(encoding="utf-8", errors="replace"))


def probe_duration(path: Path) -> float:
    """Media duration in seconds via ffprobe (0.0 when unavailable)."""
    from .audio import ffmpeg_path
    import subprocess
    exe = ffmpeg_path()
    if not exe or not Path(path).is_file():
        return 0.0
    ffprobe = Path(exe).with_name("ffprobe")
    if not ffprobe.is_file() and ffprobe.with_suffix(".exe").is_file():
        ffprobe = ffprobe.with_suffix(".exe")
    proc = subprocess.run([str(ffprobe), "-v", "error", "-show_entries", "format=duration",
                           "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        return float(proc.stdout.decode().strip())
    except ValueError:
        return 0.0


def _fallback_segments(duration: float, n_clips: int, clip_seconds: float) -> List[Dict[str, Any]]:
    """Evenly spaced windows when there is no transcript to reason about."""
    span = max(duration - clip_seconds, 0.0)
    step = span / n_clips if n_clips else span
    return [{"start": round(min(step * i + step / 4.0, span), 1),
             "end": round(min(step * i + step / 4.0 + clip_seconds, duration), 1),
             "title": "Highlight %d" % (i + 1),
             "reason": "evenly spaced window (no transcript available)"}
            for i in range(n_clips)]


def suggest_segments(client: Any, model: str, transcript: str, duration: float,
                     n_clips: int = 3, clip_seconds: float = 10.0,
                     title: str = "", transport: Any = None) -> List[Dict[str, Any]]:
    """LLM-picked highlight windows [{start, end, title, reason}], clamped to the video."""
    from .ai_video import _parse_json
    duration = max(duration, 1.0)
    if not (transcript or "").strip():
        return _fallback_segments(duration, n_clips, clip_seconds)
    user = ("Video title: %s\nDuration: %d seconds\n\nTranscript with timestamps:\n%s"
            % (title or "(unknown)", int(duration), transcript[:MAX_TRANSCRIPT_CHARS]))
    content = client.chat(model, [{"role": "system",
                                   "content": CLIP_SYSTEM.format(n=n_clips, clip=int(clip_seconds))},
                                  {"role": "user", "content": user}])
    raw = _parse_json(content)
    if not isinstance(raw, list):
        raise ConfigError("The model did not return a list of segments.")
    segments: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0.0, min(float(item.get("start", 0)), duration - 1.0))
            end = max(start + 1.0, min(float(item.get("end", start + clip_seconds)), duration))
        except (TypeError, ValueError):
            continue
        segments.append({"start": round(start, 1), "end": round(end, 1),
                         "title": str(item.get("title", "Highlight"))[:80],
                         "reason": str(item.get("reason", ""))[:200]})
        if len(segments) >= n_clips:
            break
    return segments or _fallback_segments(duration, n_clips, clip_seconds)


def cut_clip(src: Path, start: float, end: float, dest: Path) -> Path:
    """Stream-copy [start, end] of ``src`` into ``dest`` (fast, no re-encode)."""
    from .audio import _ffmpeg, ffmpeg_path
    if not ffmpeg_path():
        raise ConfigError("ffmpeg is required to cut clips (https://ffmpeg.org).")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(["-ss", "%.2f" % max(0.0, start), "-i", str(src),
             "-t", "%.2f" % max(0.5, end - start), "-c", "copy", str(dest)])
    return dest


def extract_frames(src: Path, dest_dir: Path, every: float = 3.0,
                   size: int = 1280) -> List[Path]:
    """Dump one still image every ``every`` seconds of ``src``; returns the paths."""
    from .audio import _ffmpeg, ffmpeg_path
    if not ffmpeg_path():
        raise ConfigError("ffmpeg is required to extract frames (https://ffmpeg.org).")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for old in dest_dir.glob("frame_*.jpg"):
        old.unlink()
    _ffmpeg(["-i", str(src), "-vf", "fps=1/%.1f,scale=%d:-2" % (max(0.5, every), size),
             "-q:v", "3", str(dest_dir / "frame_%03d.jpg")])
    return sorted(dest_dir.glob("frame_*.jpg"))


def build_library(urls: List[str], workdir: Path, client: Any, model: str, *,
                  n_clips: int = 3, clip_seconds: float = 10.0, stills: bool = True,
                  frame_every: float = 3.0, cookies: Optional[str] = None,
                  progress: Optional[Callable[[str], None]] = None) -> Dict[str, List[Any]]:
    """Process every URL: download, pick highlights, cut clips, extract stills.

    A URL that fails (removed video, bot check, network) is reported in
    ``library["errors"]`` and the remaining URLs are still processed. Returns
    ``{"clips": [paths], "frames": [paths], "segments": [segment dicts],
    "sources": [titles], "errors": [messages]}`` under ``workdir/clip_library``.
    """
    workdir = Path(workdir) / "clip_library"
    library: Dict[str, List[Any]] = {"clips": [], "frames": [], "segments": [],
                                     "sources": [], "errors": []}
    for url in urls:
        url = (url or "").strip()
        if not url or not url.startswith("http"):
            continue
        if progress:
            progress("Downloading %s" % url)
        try:
            path, info = download_video(url, workdir / "sources", cookies=cookies)
        except Exception as exc:
            library["errors"].append(_download_hint(url, exc))
            continue
        title = str(info.get("title") or path.stem)
        duration = float(info.get("duration") or probe_duration(path) or 0.0)
        if progress:
            progress("Fetching subtitles for %s" % title)
        transcript = fetch_transcript(url, workdir / "sources", cookies=cookies)
        if progress:
            progress("Picking highlights from %s" % title)
        segments = suggest_segments(client, model, transcript, duration,
                                    n_clips=n_clips, clip_seconds=clip_seconds, title=title)
        stem = _slug(title)
        for j, seg in enumerate(segments, 1):
            clip_path = cut_clip(path, seg["start"], seg["end"],
                                 workdir / "clips" / ("%s_%02d.mp4" % (stem, j)))
            library["clips"].append(str(clip_path))
            library["segments"].append({"source": title, **seg})
            if stills:
                if progress:
                    progress("Extracting frames from %s (%d/%d)" % (seg["title"], j, len(segments)))
                for frame in extract_frames(clip_path, workdir / "frames" / ("%s_%02d" % (stem, j)),
                                            every=frame_every):
                    library["frames"].append(str(frame))
        library["sources"].append(title)
    return library
