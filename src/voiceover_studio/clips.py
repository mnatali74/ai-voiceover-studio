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

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .errors import ConfigError, ProviderError

USER_AGENT = "ai-voiceover-studio (+https://github.com/yanghhu/ai-voiceover-studio)"
SUBTITLE_LANGS = ("it", "en", "es", "fr", "de", "pt")
MAX_TRANSCRIPT_CHARS = 12000

CLIP_SYSTEM = (
    "You pick the most reusable highlight segments from a video transcript. Respond with ONLY "
    "a JSON array, no markdown fences, no commentary:\n"
    '[{{"start": <seconds number>, "end": <seconds number>, '
    '"title": "short clip title", "reason": "why other videos can reuse it"}}]\n'
    "{pick_instruction} of about {clip} seconds each. They must not overlap, must stay inside "
    "the video, and must be self-contained: they will be reused as B-roll in other videos, so "
    "avoid segments that depend on earlier context."
)

PICK_ALL = ("Pick every distinct reusable highlight segment, as many as the transcript "
            "supports (at least 4)")

CLASSIFY_SYSTEM = (
    "You catalog one B-roll clip cut from a larger video. Respond with ONLY a JSON object, "
    "no markdown fences:\n"
    '{{"description": "2-3 precise sentences describing exactly what the clip shows", '
    '"tags": ["5-10 short lowercase tags"], "mood": "one or two words", '
    '"setting": "where/when", "camera": "aerial, drone, static, pan, closeup, ...", '
    '"has_people": true or false, "usable": true or false}}\n'
    "Be concrete: name objects, places, colors, light. Tags are the search keys of a library."
)

SELECT_SYSTEM = (
    "You assemble a video from a catalog of tagged B-roll clips. The user gives the idea of the "
    "video. Respond with ONLY a JSON array, ordered as the final video, no markdown fences:\n"
    '[{{"id": "<clip id from the catalog>", "narration": "one spoken sentence in {language} "'
    '"for this shot"}}]\n'
    "Choose the 3 to 8 clips whose descriptions and tags best match the idea, never reuse a "
    "clip twice, and make the first shot a strong visual hook."
)

SEARCH_SYSTEM = (
    "You are a stock-footage researcher. The user describes the video they are building. "
    "Respond with ONLY a JSON array of {n} different YouTube search queries (short, English) "
    "that would find cinematic B-roll for it. Rules: favor landscapes, aerial and drone shots, "
    "nature, cityscapes, close-ups of objects and places; avoid videos whose subject is people "
    "(no vlogs, tutorials, interviews, talking heads). Every query must target a different "
    "visual angle so the final video draws from several distinct sources."
)

_HUMAN_WORDS = re.compile(r"\b(vlog|vlogs|tutorial|interview|podcast|unboxing|review|reviews|"
                          r"talking|webinar|webcam|facecam|selfie|reaction|q&a|storytime)\b", re.I)


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


def title_avoids_people(title: str) -> bool:
    """Cheap heuristic: drop videos whose title suggests people as the subject."""
    return not _HUMAN_WORDS.search(title or "")


def suggest_searches(client: Any, model: str, prompt: str, n: int = 5) -> List[str]:
    """LLM turns the video prompt into n varied B-roll search queries."""
    from .ai_video import _parse_json
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Write the prompt of the video you want to build.")
    content = client.chat(model, [{"role": "system", "content": SEARCH_SYSTEM.format(n=n)},
                                  {"role": "user", "content": prompt}])
    raw = _parse_json(content)
    if not isinstance(raw, list):
        raise ConfigError("The model did not return a list of search queries.")
    queries = [str(q).strip() for q in raw if isinstance(q, str) and str(q).strip()][:n]
    if not queries:
        raise ConfigError("No usable search queries were returned.")
    return queries


def search_youtube(query: str, limit: int = 4,
                   cookies: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search YouTube (no download) and return candidate videos without people-focused titles."""
    yt_dlp = _import_yt_dlp()
    opts = {"extract_flat": True, "quiet": True, "no_warnings": True,
            "noplaylist": True, **_cookie_opts(cookies)}
    with yt_dlp.YoutubeDL(opts) as ydl:
        data = ydl.extract_info("ytsearch%d:%s" % (limit, query), download=False) or {}
    results = []
    for entry in data.get("entries") or []:
        vid = entry.get("id") or entry.get("url", "")
        title = str(entry.get("title") or "")
        if not vid or not title_avoids_people(title):
            continue
        results.append({"id": vid, "title": title,
                        "channel": str(entry.get("channel") or entry.get("uploader") or ""),
                        "duration": float(entry.get("duration") or 0),
                        "url": entry.get("url") or "https://www.youtube.com/watch?v=%s" % vid})
    return results


def find_sources(client: Any, model: str, prompt: str, *, sources: int = 5,
                 per_query: int = 4, cookies: Optional[str] = None,
                 progress: Optional[Callable[[str], None]] = None
                 ) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Prompt -> varied searches -> candidate videos from at least ``sources`` queries."""
    queries = suggest_searches(client, model, prompt, n=max(1, sources))
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for query in queries:
        if progress:
            progress("Searching YouTube: %s" % query)
        for item in search_youtube(query, limit=per_query, cookies=cookies):
            if item["id"] not in seen:
                seen.add(item["id"])
                candidates.append({"query": query, **item})
    return queries, candidates


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
                     n_clips: Optional[int] = 3, clip_seconds: float = 10.0,
                     title: str = "", transport: Any = None) -> List[Dict[str, Any]]:
    """LLM-picked highlight windows [{start, end, title, reason}], clamped to the video.

    ``n_clips=None`` asks the model for every distinct highlight it can find.
    """
    from .ai_video import _parse_json
    duration = max(duration, 1.0)
    if n_clips is None:
        n_clips = 24
    if not (transcript or "").strip():
        return _fallback_segments(duration, min(n_clips, 12), clip_seconds)
    user = ("Video title: %s\nDuration: %d seconds\n\nTranscript with timestamps:\n%s"
            % (title or "(unknown)", int(duration), transcript[:MAX_TRANSCRIPT_CHARS]))
    instruction = PICK_ALL if n_clips >= 24 else "Pick %d segments" % n_clips
    content = client.chat(model, [{"role": "system",
                                   "content": CLIP_SYSTEM.format(pick_instruction=instruction,
                                                                 clip=int(clip_seconds))},
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
    return segments or _fallback_segments(duration, min(n_clips, 12), clip_seconds)


def cut_clip(src: Path, start: float, end: float, dest: Path,
             audio: bool = False) -> Path:
    """Stream-copy [start, end] of ``src`` into ``dest`` (fast, no re-encode).

    Audio is dropped by default: the clips are B-roll, the final video gets its own
    voiceover and music.
    """
    from .audio import _ffmpeg, ffmpeg_path
    if not ffmpeg_path():
        raise ConfigError("ffmpeg is required to cut clips (https://ffmpeg.org).")
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = ["-ss", "%.2f" % max(0.0, start), "-i", str(src),
            "-t", "%.2f" % max(0.5, end - start), "-c", "copy"]
    if not audio:
        args.append("-an")
    args.append(str(dest))
    _ffmpeg(args)
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


def catalog_path(workdir: Path) -> Path:
    return Path(workdir) / "clip_library" / "catalog.json"


def load_catalog(workdir: Path) -> List[Dict[str, Any]]:
    """Read the clip catalog (a JSON database) from the session workdir."""
    path = catalog_path(workdir)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return data if isinstance(data, list) else []


def save_catalog(workdir: Path, entries: List[Dict[str, Any]]) -> Path:
    path = catalog_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _transcript_excerpt(transcript: str, start: float, end: float) -> str:
    """Transcript lines whose timestamp falls inside [start-5, end+5]."""
    lines = []
    for line in (transcript or "").splitlines():
        match = re.match(r"\[(\d+):(\d+)\]", line)
        if not match:
            continue
        when = int(match.group(1)) * 60 + int(match.group(2))
        if start - 5 <= when <= end + 5:
            lines.append(line.split("]", 1)[1].strip())
    return " ".join(lines)[:600]


def classify_clip(client: Any, model: str, clip_path: Path, source_title: str,
                  context: str = "") -> Dict[str, Any]:
    """Detailed LLM classification of one clip: description, tags, mood, camera, people.

    Uses a still frame when ffmpeg can extract one; falls back to text-only if the
    model cannot see images.
    """
    from .ai_video import _first_frame, _parse_json_object
    user = ("Source video title: %s\nSpoken context around the clip: %s"
            % (source_title or "(unknown)", context or "(none)"))
    content: List[Dict[str, Any]] = [{"type": "text", "text": user}]
    frame = Path(str(clip_path) + ".frame.jpg")
    if clip_path.suffix.lower() in (".mp4", ".webm", ".mov", ".mkv"):
        if _first_frame(clip_path, frame):
            encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + encoded}})
    try:
        answer = client.chat(model, [{"role": "system", "content": CLASSIFY_SYSTEM},
                                     {"role": "user", "content": content}])
    except ProviderError:
        answer = client.chat(model, [{"role": "system", "content": CLASSIFY_SYSTEM},
                                     {"role": "user", "content": [{"type": "text", "text": user}]}])
    try:
        data = _parse_json_object(answer)
    except ProviderError:
        return {"description": answer[:300], "tags": [], "mood": "", "setting": "",
                "camera": "", "has_people": None, "usable": True}
    tags = [str(t).lower()[:40] for t in (data.get("tags") or []) if str(t).strip()][:12]
    return {"description": str(data.get("description", ""))[:400],
            "tags": tags,
            "mood": str(data.get("mood", ""))[:60],
            "setting": str(data.get("setting", ""))[:120],
            "camera": str(data.get("camera", ""))[:60],
            "has_people": bool(data.get("has_people")),
            "usable": bool(data.get("usable", True))}


def select_clips(client: Any, model: str, prompt: str, catalog: List[Dict[str, Any]],
                 language: str = "the user's language") -> List[Dict[str, Any]]:
    """Prompt + catalog -> ordered [{entry, narration}] for the final video."""
    from .ai_video import _parse_json
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Write the prompt of the video you want to build.")
    if not catalog:
        raise ValueError("The clip library is empty: download some clips first.")
    listing = "\n".join(
        "%s | %s | tags: %s | camera: %s | people: %s"
        % (entry.get("id"), entry.get("description", "")[:200],
           ", ".join(entry.get("tags") or []), entry.get("camera", ""),
           entry.get("has_people"))
        for entry in catalog)
    content = client.chat(model, [{"role": "system", "content": SELECT_SYSTEM.format(language=language)},
                                  {"role": "user", "content": "Video idea: %s\n\nCatalog:\n%s"
                                   % (prompt, listing[:24000])}])
    raw = _parse_json(content)
    if not isinstance(raw, list):
        raise ConfigError("The model did not return a selection.")
    by_id = {entry.get("id"): entry for entry in catalog}
    selected: List[Dict[str, Any]] = []
    used = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry = by_id.get(str(item.get("id", "")))
        if entry is None or entry.get("id") in used:
            continue
        used.add(entry.get("id"))
        selected.append({"entry": entry,
                         "narration": str(item.get("narration", "")).strip()[:400]})
        if len(selected) >= 8:
            break
    if not selected:
        raise ConfigError("None of the chosen clips is in the catalog.")
    return selected


def build_video_script(selected: List[Dict[str, Any]], title: str = "") -> str:
    """Selected clips -> Markdown script with ![](clip) and narration per shot."""
    lines = []
    if title.strip():
        lines += ["# " + title.strip(), ""]
    for item in selected:
        entry = item["entry"]
        if entry.get("clip_path"):
            lines += ["![](%s)" % entry["clip_path"], ""]
        lines += [item["narration"], ""]
    return "\n".join(lines).strip() + "\n"


def build_library(urls: List[str], workdir: Path, client: Any, model: str, *,
                  n_clips: Optional[int] = 3, clip_seconds: float = 10.0, stills: bool = True,
                  frame_every: float = 3.0, keep_audio: bool = False,
                  cookies: Optional[str] = None, delay: float = 5.0,
                  sleep: Callable[[float], None] = time.sleep,
                  progress: Optional[Callable[[str], None]] = None) -> Dict[str, List[Any]]:
    """Process every URL: download, pick highlights, cut clips, extract stills, classify.

    Waits ``delay`` seconds between YouTube requests to stay under the radar, and
    appends every classified clip to the JSON catalog on disk (incrementally, so a
    crash never loses what was already processed). A URL that fails (removed video,
    bot check, network) is reported in ``library["errors"]`` and the remaining URLs
    are still processed. Returns ``{"clips", "frames", "segments", "sources",
    "errors", "catalog"}`` under ``workdir/clip_library``.
    """
    workdir = Path(workdir)
    lib_dir = workdir / "clip_library"
    catalog = load_catalog(workdir)
    library: Dict[str, List[Any]] = {"clips": [], "frames": [], "segments": [],
                                     "sources": [], "errors": [], "catalog": catalog}
    first = True
    for url in urls:
        url = (url or "").strip()
        if not url or not url.startswith("http"):
            continue
        if not first and delay > 0:
            if progress:
                progress("Waiting %.0f s before the next download (rate limit)" % delay)
            sleep(delay)
        first = False
        if progress:
            progress("Downloading %s" % url)
        try:
            path, info = download_video(url, lib_dir / "sources", cookies=cookies)
        except Exception as exc:
            library["errors"].append(_download_hint(url, exc))
            continue
        title = str(info.get("title") or path.stem)
        duration = float(info.get("duration") or probe_duration(path) or 0.0)
        if progress:
            progress("Fetching subtitles for %s" % title)
        if delay > 0:
            sleep(min(delay, 3.0))
        transcript = fetch_transcript(url, lib_dir / "sources", cookies=cookies)
        if progress:
            progress("Picking highlights from %s" % title)
        segments = suggest_segments(client, model, transcript, duration,
                                    n_clips=n_clips, clip_seconds=clip_seconds, title=title)
        stem = _slug(title)
        for j, seg in enumerate(segments, 1):
            clip_path = cut_clip(path, seg["start"], seg["end"],
                                 lib_dir / "clips" / ("%s_%02d.mp4" % (stem, j)),
                                 audio=keep_audio)
            library["clips"].append(str(clip_path))
            library["segments"].append({"source": title, **seg})
            frames: List[Path] = []
            if stills:
                if progress:
                    progress("Extracting frames from %s (%d/%d)" % (seg["title"], j, len(segments)))
                frames = extract_frames(clip_path, lib_dir / "frames" / ("%s_%02d" % (stem, j)),
                                        every=frame_every)
                library["frames"].extend(str(frame) for frame in frames)
            if progress:
                progress("Classifying %s (%d/%d)" % (seg["title"], j, len(segments)))
            try:
                details = classify_clip(client, model, clip_path, title,
                                        _transcript_excerpt(transcript, seg["start"], seg["end"]))
            except (ProviderError, ConfigError, ConnectionError) as exc:
                details = {"description": "classification failed: %s" % str(exc)[:150],
                           "tags": [], "mood": "", "setting": "", "camera": "",
                           "has_people": None, "usable": True}
            entry = {"id": "%s_%02d" % (stem, j), "source_url": url, "source_title": title,
                     "clip_path": str(clip_path), "start": seg["start"], "end": seg["end"],
                     "segment_title": seg["title"], "frames": [str(f) for f in frames],
                     **details}
            catalog.append(entry)
            library["catalog"] = catalog
            save_catalog(workdir, catalog)
        library["sources"].append(title)
    return library
