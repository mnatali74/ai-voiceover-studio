"""AI video builder: prompt -> scene plan -> keyless stock assets -> script.

The pipeline behind the *AI Video* tab:

    plan_scenes()     prompt -> [{narration, query}] via an OpenRouter chat model
    find_asset()      keyless sources: Wikimedia Commons (video and images), Openverse,
                      then a generated image from Pollinations when nothing matches
    verify_asset()    optional vision check via OpenRouter: does the picture match the scene?
    build_script()    scenes + assets -> Markdown script for :func:`video.render_video`

HTTP goes through the same injectable transport as the voice providers, so tests
run fully offline.
"""
from __future__ import annotations

import base64
import json
import os
import random
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .errors import ConfigError, ProviderError
from .providers.miragea import HttpResponse, Transport, urllib_transport

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
USER_AGENT = "ai-voiceover-studio (+https://github.com/yanghhu/ai-voiceover-studio)"

DEFAULT_MODELS = [
    "openrouter/auto",
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
]

PLAN_SYSTEM = (
    "You are a video director. Break the user's idea into {n} short scenes for a narrated "
    "faceless video. Respond with ONLY a JSON array, no markdown fences, no commentary:\n"
    '[{{"narration": "one or two spoken sentences in {language}", '
    '"query": "2-5 simple English keywords for a stock footage search"}}]\n'
    "The narrations must flow as one continuous script: hook first, then the story, "
    "then a closing line. Keep every narration under 40 words."
)

VERIFY_PROMPT = (
    'Does this picture plausibly illustrate the scene "{query}"? '
    'Answer with ONLY a JSON object: {{"match": true or false, "reason": "short reason"}}'
)


@dataclass
class Scene:
    index: int
    narration: str
    query: str
    asset_path: Optional[str] = None
    source: str = ""
    verified: Optional[bool] = None
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class OpenRouterClient:
    """Minimal OpenAI-compatible client for openrouter.ai (stdlib only)."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = OPENROUTER_BASE_URL,
                 transport: Optional[Transport] = None, timeout: float = 120.0):
        self.api_key = (api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
        if not self.api_key:
            raise ConfigError("OpenRouter needs an API key: set OPENROUTER_API_KEY "
                              "(https://openrouter.ai/keys)")
        self.base_url = base_url.rstrip("/")
        self.transport = transport or urllib_transport
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": "Bearer %s" % self.api_key,
                "Content-Type": "application/json", "User-Agent": USER_AGENT}

    def request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        response = self.transport(method, self.base_url + path, self._headers(), payload, self.timeout)
        if response.status >= 400:
            detail = response.body.decode("utf-8", "replace")[:300]
            raise ProviderError("OpenRouter %s %s failed (%d): %s"
                                % (method, path, response.status, detail))
        return response.json()

    def chat(self, model: str, messages: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
        data = self.request("POST", "/chat/completions",
                            {"model": model, "messages": messages, "max_tokens": max_tokens})
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Unexpected OpenRouter response shape: %s" % exc) from None

    def models(self) -> List[str]:
        data = self.request("GET", "/models")
        return sorted(item["id"] for item in data.get("data", []) if item.get("id"))


def _parse_json(text: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise ProviderError("The model did not return valid JSON: %s" % text[:200]) from None


def plan_scenes(client: OpenRouterClient, prompt: str, *, duration: float = 30.0,
                language: str = "auto", model: str = DEFAULT_MODELS[0]) -> List[Scene]:
    """Ask the model for a scene breakdown; returns scenes with narration + search query."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Write a prompt describing the video you want.")
    n = max(3, min(12, round(duration / 8.0)))
    lang = "the same language as the prompt" if language in ("", "auto") else language
    content = client.chat(model, [{"role": "system", "content": PLAN_SYSTEM.format(n=n, language=lang)},
                                  {"role": "user", "content": prompt}])
    raw = _parse_json(content)
    if not isinstance(raw, list) or not raw:
        raise ProviderError("The model returned no scenes.")
    scenes = []
    for i, item in enumerate(raw[:12]):
        if not isinstance(item, dict) or not item.get("narration"):
            continue
        scenes.append(Scene(i, str(item["narration"]).strip(),
                            str(item.get("query", "")).strip() or prompt[:60]))
    if not scenes:
        raise ProviderError("The model returned scenes without narration.")
    for i, scene in enumerate(scenes):
        scene.index = i
    return scenes


def _download(url: str, dest: Path, transport: Transport) -> Path:
    response = transport("GET", url, {"User-Agent": USER_AGENT}, None, 60.0)
    if response.status >= 400 or not response.body:
        raise ProviderError("Download failed (%d): %s" % (response.status, url[:120]))
    dest.write_bytes(response.body)
    return dest


def _commons_search(query: str, filetype: str, transport: Transport) -> List[Dict[str, Any]]:
    params = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": "%s filetype:%s" % (query, filetype), "gsrlimit": "6", "gsrnamespace": "6",
        "prop": "imageinfo", "iiprop": "url|size|mime"})
    url = "https://commons.wikimedia.org/w/api.php?" + params
    try:
        data = transport("GET", url, {"User-Agent": USER_AGENT}, None, 20.0)
        if data.status >= 400:
            return []
        pages = data.json().get("query", {}).get("pages", {})
    except (ConnectionError, ValueError):
        return []
    items = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        if info.get("url"):
            items.append(info)
    return items


def _openverse_search(query: str, orientation: str, transport: Transport) -> List[Dict[str, Any]]:
    params = {"q": query, "page_size": "6"}
    if orientation == "portrait":
        params["aspect_ratio"] = "tall"
    url = "https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(params)
    try:
        data = transport("GET", url, {"User-Agent": USER_AGENT}, None, 20.0)
        if data.status >= 400:
            return []
        return data.json().get("results", [])
    except (ConnectionError, ValueError):
        return []


def _pollinations_url(query: str, width: int, height: int) -> str:
    seed = random.randint(1, 10**6)
    prompt = urllib.parse.quote(query)
    return ("https://image.pollinations.ai/prompt/%s?width=%d&height=%d&nologo=true&seed=%d"
            % (prompt, width, height, seed))


def find_asset(query: str, orientation: str, workdir: Path, index: int,
               transport: Optional[Transport] = None) -> Tuple[Path, str]:
    """Find or generate one visual for a scene. Returns (path, source description)."""
    transport = transport or urllib_transport
    workdir.mkdir(parents=True, exist_ok=True)
    query = (query or "").strip() or "abstract background"
    portrait = orientation == "portrait"
    width, height = (720, 1280) if portrait else (1280, 720)

    for info in _commons_search(query, "video", transport):
        mime = info.get("mime", "")
        if mime in ("video/webm", "video/mp4") and info.get("size", 10**9) < 80 * 1024 * 1024:
            ext = "webm" if mime == "video/webm" else "mp4"
            try:
                path = _download(info["url"], workdir / ("scene_%02d.%s" % (index, ext)), transport)
                return path, "wikimedia video"
            except ProviderError:
                continue

    for item in _openverse_search(query, orientation, transport):
        url = item.get("url") or item.get("thumbnail")
        if not url:
            continue
        try:
            path = _download(url, workdir / ("scene_%02d.jpg" % index), transport)
            return path, "openverse image"
        except ProviderError:
            continue

    for info in _commons_search(query, "bitmap", transport):
        if info.get("width", 0) >= 640:
            try:
                path = _download(info["url"], workdir / ("scene_%02d.jpg" % index), transport)
                return path, "wikimedia image"
            except ProviderError:
                continue

    return generate_asset(query, orientation, workdir, index, transport)


def generate_asset(query: str, orientation: str, workdir: Path, index: int,
                   transport: Optional[Transport] = None) -> Tuple[Path, str]:
    """Generate one image for the scene with Pollinations (keyless)."""
    transport = transport or urllib_transport
    workdir.mkdir(parents=True, exist_ok=True)
    query = (query or "").strip() or "abstract background"
    width, height = (720, 1280) if orientation == "portrait" else (1280, 720)
    try:
        path = _download(_pollinations_url(query, width, height),
                         workdir / ("scene_%02d.jpg" % index), transport)
        return path, "generated (pollinations)"
    except ProviderError as exc:
        raise ProviderError("No asset found for scene %d (%r): %s" % (index, query, exc)) from None


def _first_frame(video: Path, dest: Path) -> bool:
    from .audio import ffmpeg_path, _ffmpeg
    if not ffmpeg_path():
        return False
    try:
        _ffmpeg(["-ss", "1", "-i", str(video), "-frames:v", "1", "-q:v", "3", str(dest)])
        return dest.is_file()
    except ConfigError:
        return False


def verify_asset(client: OpenRouterClient, scene: Scene, model: str) -> Tuple[bool, str]:
    """Vision check: does the asset plausibly match the scene? Videos are checked on one frame."""
    path = scene.asset_path
    if not path or not Path(path).is_file():
        return False, "no asset"
    media = Path(path)
    frame = media
    if media.suffix.lower() in (".mp4", ".webm", ".mov", ".mkv"):
        frame = media.with_suffix(".frame.jpg")
        if not _first_frame(media, frame):
            return True, "skipped: video frame check needs ffmpeg"
    encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
    content = [{"type": "text", "text": VERIFY_PROMPT.format(query=scene.query)},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}]
    answer = client.chat(model, [{"role": "user", "content": content}], max_tokens=200)
    try:
        verdict = _parse_json_object(answer)
        return bool(verdict.get("match")), str(verdict.get("reason", ""))[:200]
    except ProviderError:
        return True, "skipped: unreadable vision answer"


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except ValueError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ProviderError("no JSON object in vision answer") from None
        data = json.loads(cleaned[start:end + 1])
    if not isinstance(data, dict):
        raise ProviderError("vision answer is not an object")
    return data


def build_script(scenes: List[Scene], title: str = "") -> str:
    """Scenes -> Markdown script with ![](asset) per scene, ready for render_video."""
    lines = []
    if title.strip():
        lines += ["# " + title.strip(), ""]
    for scene in scenes:
        if scene.asset_path:
            lines += ["![](%s)" % scene.asset_path, ""]
        lines += [scene.narration, ""]
    return "\n".join(lines).strip() + "\n"


def plan_and_fetch(client: OpenRouterClient, prompt: str, workdir: Path, *,
                   duration: float = 30.0, language: str = "auto",
                   model: str = DEFAULT_MODELS[0], orientation: str = "landscape",
                   verify: bool = True, vision_model: str = DEFAULT_MODELS[0],
                   transport: Optional[Transport] = None,
                   progress: Optional[Any] = None) -> List[Scene]:
    """Full builder: plan scenes, fetch one asset each, optionally verify with vision."""
    scenes = plan_scenes(client, prompt, duration=duration, language=language, model=model)
    for scene in scenes:
        path, source = find_asset(scene.query, orientation, workdir, scene.index, transport)
        scene.asset_path, scene.source = str(path), source
        if progress:
            progress(scene)
        if verify and client.api_key:
            scene.verified, scene.reason = verify_asset(client, scene, vision_model)
    return scenes
