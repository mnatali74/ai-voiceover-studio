"""Gradio web studio: write a script, pick engine, language and speed, listen.

    voiceover studio --host 127.0.0.1 --port 7860

Needs the optional web extra::

    pip install "ai-voiceover-studio[webui]"

The UI is a thin layer over :class:`voiceover_studio.studio.Studio` and
:func:`voiceover_studio.video.render_video`; every handler can also be called
directly from Python for tests.
"""
from __future__ import annotations

import functools
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ai_video import DEFAULT_MODELS
from .overlays import CARD_STYLES
from .errors import ConfigError, ProviderError, VoiceoverError
from .providers import available_providers

import gradio as gr

LANGUAGES = ["auto", "en", "it", "es", "fr", "de", "pt", "nl", "ru", "uk", "pl",
             "tr", "ar", "hi", "zh", "ja", "ko"]

DEFAULT_SCRIPT = """# Welcome to AI Voiceover Studio

Write your narration here, one paragraph per idea.

[NARRATOR] Speaker tags like this give each speaker its own voice.

Add a pause with `[pause 500]`, or **bold** words to stress them in the captions.
"""


def _friendly(fn):
    """Show the real error message in the UI instead of a bare 'Error' toast."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gr.Error:
            raise
        except Exception as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            raise gr.Error("%s: %s" % (exc.__class__.__name__, detail[:400])) from exc
    return wrapped


def _examples_dir() -> Optional[Path]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "examples"
        if candidate.is_dir() and any(candidate.glob("*.md")):
            return candidate
    return None


def example_scripts() -> List[str]:
    folder = _examples_dir()
    if folder is None:
        return []
    return sorted(str(p) for p in folder.glob("*.md"))


def _engine_choices() -> List[str]:
    return ["auto"] + available_providers()


def new_session(workdir: Optional[Path] = None) -> Dict[str, Any]:
    base = Path(workdir).resolve() if workdir else Path(tempfile.gettempdir()) / "voiceover-studio"
    session = base / uuid.uuid4().hex[:12]
    session.mkdir(parents=True, exist_ok=True)
    return {"dir": session}


def _voice_overrides(language: str, speed: float):
    from .providers.base import Voice
    return Voice(**{k: v for k, v in (("language", language), ("speed", speed))
                    if v and v != "auto" and (k != "speed" or v != 1.0)})


def generate_voiceover(script: str, provider: str, language: str, speed: float,
                       gap: float, output_name: str, session: Dict[str, Any]
                       ) -> Tuple[str, Optional[str], Optional[str], List[List[Any]]]:
    """Render the script to audio + SRT. Returns (status, audio, srt, rows)."""
    text = (script or "").strip()
    if not text:
        raise ValueError("Write a script first, or load an example.")
    from .studio import Studio
    name = output_name.strip() or "voiceover.mp3"
    if not Path(name).suffix:
        name += ".mp3"
    workdir = Path(session["dir"])
    script_path = workdir / "script.md"
    script_path.write_text(text, encoding="utf-8")
    outdir = workdir / "voiceover"
    studio = Studio(provider or "auto")
    overrides = _voice_overrides(language, speed)
    result = studio.render_script(script_path, outdir, output_name=name, gap=gap,
                                  voice_overrides=overrides)
    rows = [[s.index, s.speaker, s.voice, round(s.start, 2), round(s.end, 2), s.text]
            for s in result.segments]
    status = ("Rendered **%d** segments, **%.1f s** with the **%s** engine."
              % (len(rows), result.duration, studio.provider.name))
    return status, result.audio_path, result.srt_path, rows


def say_line(text: str, provider: str, voice: str, language: str, speed: float,
             session: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Voice one piece of text. Returns (status, audio)."""
    if not (text or "").strip():
        raise ValueError("Write the text to speak.")
    from .studio import Studio
    studio = Studio(provider or "auto")
    out = Path(session["dir"]) / ("say." + studio.provider.native_format)
    overrides = _voice_overrides(language, speed)
    opts = {"language": overrides.language, "speed": overrides.speed}
    result = studio.say(text.strip(), voice or None, out,
                        **{k: v for k, v in opts.items() if v is not None})
    return "Voiced with **%s** (%s)." % (studio.provider.name, result.get("voice")), result["output"]


SAMPLE_SENTENCES = {
    "it": "Ciao, questa è una prova della voce selezionata.",
    "en": "Hello, this is a quick preview of the selected voice.",
    "es": "Hola, esta es una prueba rápida de la voz seleccionada.",
    "fr": "Bonjour, ceci est un aperçu rapide de la voix sélectionnée.",
    "de": "Hallo, dies ist eine kurze Probe der ausgewählten Stimme.",
    "pt": "Olá, esta é uma prévia rápida da voz selecionada.",
    "ru": "Привет, это быстрое прослушивание выбранного голоса.",
    "zh": "你好，这是所选语音的快速试听。",
    "ja": "こんにちは、選択した音声の簡単なプレビューです。",
    "ko": "안녕하세요, 선택한 음성의 미리 듣기입니다.",
    "ar": "مرحبًا، هذه معاينة سريعة للصوت المحدد.",
    "hi": "नमस्ते, यह चुने गए आवाज़ का तेज़ पूर्वावलोकन है।",
}


def _sample_sentence(voice_id: str) -> str:
    prefix = voice_id.strip().split("-")[0].lower()
    return SAMPLE_SENTENCES.get(prefix, SAMPLE_SENTENCES["en"])


def pick_voice(evt: Any, rows: List[List[Any]]) -> str:
    """Fill the preview box with the voice id of the clicked table row."""
    try:
        return str(rows[evt.index[0]][0])
    except (AttributeError, IndexError, TypeError):
        return ""


def test_voice(voice_id: str, provider: str, session: Dict[str, Any]
               ) -> Tuple[str, Optional[str]]:
    """Synthesize a short sample line with the selected voice. Returns (status, audio)."""
    voice_id = (voice_id or "").strip()
    if not voice_id:
        raise ValueError("Click a voice in the table first.")
    from .studio import Studio
    studio = Studio(provider or "auto")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", voice_id)
    out = Path(session["dir"]) / ("preview_" + safe + "." + studio.provider.native_format)
    result = studio.say(_sample_sentence(voice_id), voice_id, out)
    return "Preview of **%s** with the **%s** engine." % (voice_id, studio.provider.name), result["output"]


def list_voices(provider: str, filter_text: str) -> Tuple[str, List[List[Any]]]:
    """List voices of the engine. Returns (status, rows)."""
    from .studio import Studio
    studio = Studio(provider or "auto")
    found, offset = [], 0
    while True:
        items, more = studio.provider.list_voices(100, offset)
        found.extend(items)
        offset += len(items)
        if not (more and items) or offset >= 1000:
            break
    needle = (filter_text or "").strip().lower()
    if needle:
        found = [v for v in found
                 if needle in ("%s %s %s" % (v.voice_id, v.name, v.language)).lower()]
    rows = [[v.voice_id, v.language or "-", v.name] for v in found]
    return "**%d** voices for the **%s** engine." % (len(rows), studio.provider.name), rows


def save_broll(files: Optional[List[Any]], session: Dict[str, Any],
               extra: Optional[List[Any]] = None) -> Optional[Path]:
    """Copy uploaded images/clips (plus picked library assets) into the session."""
    items = list(files or []) + [p for p in (extra or []) if p]
    if not items:
        return None
    folder = Path(session["dir"]) / "broll"
    folder.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(items, 1):
        src = Path(item if isinstance(item, (str, Path)) else item.name)
        shutil.copy(src, folder / ("%03d_%s" % (i, src.name)))
    return folder


def render_video_clip(script: str, provider: str, language: str, speed: float,
                      vertical: bool, style: str, music: str, broll: Optional[List[Any]],
                      session: Dict[str, Any], library: Optional[List[str]] = None,
                      transition_style: str = "mix", transition: float = 0.5,
                      voice: Optional[str] = None, small: bool = False,
                      output_name: str = "video.mp4") -> Tuple[str, Optional[str]]:
    """Render the script to an MP4 (needs ffmpeg). Returns (status, video).

    With ``small=True`` the render is a fast low-resolution draft (preview)."""
    text = (script or "").strip()
    if not text:
        raise ValueError("Write a script first, or load an example.")
    from .studio import Studio
    from .video import render_video
    workdir = Path(session["dir"])
    script_path = workdir / "script.md"
    script_path.write_text(text, encoding="utf-8")
    studio = Studio(provider or "auto")
    out = workdir / output_name
    overrides = _voice_overrides(language, speed)
    if small:
        size = (540, 960) if vertical else (960, 540)
    else:
        size = (1080, 1920) if vertical else (1920, 1080)
    result = render_video(studio, save_broll(broll, session, extra=library), script_path, out,
                          size=size,
                          style="clean" if style == "none" else ("auto" if style == "auto" else style),
                          subtitles="none" if style == "none" else "burn",
                          music=music or None, voice_overrides=overrides,
                          default_voice=voice or None,
                          transition=float(transition), transition_style=transition_style)
    status = ("Video **%.1f s**, %d shots, %s captions, %s transitions, engine **%s**."
              % (result["duration"], result["shots"], result["style"], transition_style,
                 studio.provider.name))
    return status, str(out)


def ai_plan(prompt: str, duration: float, language: str, orientation: str, model: str,
            verify: bool, vision_model: str, session: Dict[str, Any]
            ) -> Tuple[str, List[Tuple[str, str]], List[List[Any]]]:
    """Prompt -> scene plan + keyless assets (OpenRouter for planning and vision)."""
    from .ai_video import OpenRouterClient, plan_and_fetch
    if not (prompt or "").strip():
        raise ValueError("Write a prompt describing the video you want.")
    client = OpenRouterClient()
    workdir = Path(session["dir"]) / "ai_assets"
    scenes = plan_and_fetch(client, prompt, workdir, duration=duration, language=language,
                            model=model or "openrouter/auto", orientation=orientation,
                            verify=bool(verify), vision_model=vision_model or model)
    session["scenes"] = [s.__dict__ for s in scenes]
    gallery = [(s.asset_path, "Scene %d: %s (%s)" % (s.index + 1, s.query, s.source))
               for s in scenes if s.asset_path]
    rows = [[s.index + 1, s.narration, s.query, s.source,
             "ok" if s.verified else ("failed" if s.verified is False else "-")]
            for s in scenes]
    checked = sum(1 for s in scenes if s.verified)
    sources = ", ".join(sorted({s.source for s in scenes if s.asset_path})) or "none"
    status = "Planned **%d** scenes. Assets: %s." % (len(scenes), sources)
    missing = [s.index + 1 for s in scenes if not s.asset_path]
    if missing:
        status += (" No asset for scene(s) %s — they will render as animated text. "
                   "You can retry them with Regenerate."
                   % ", ".join(str(i) for i in missing))
    if verify:
        status += " Vision check passed for %d/%d." % (checked, len(scenes))
    return status, gallery, rows


def ai_regenerate(scene_number: int, orientation: str, session: Dict[str, Any]
                  ) -> Tuple[str, List[Tuple[str, str]]]:
    """Replace one scene's asset with a freshly generated image (keyless)."""
    from .ai_video import Scene, generate_asset
    scenes = [Scene(**d) for d in session.get("scenes", [])]
    if not scenes:
        raise ValueError("Plan the scenes first.")
    scene = next((s for s in scenes if s.index == int(scene_number) - 1), None)
    if scene is None:
        raise ValueError("Pick a scene number from the list.")
    try:
        path, source = generate_asset(scene.query, orientation,
                                      Path(session["dir"]) / "ai_assets", scene.index)
    except ProviderError as exc:
        return "Scene %d: generation failed — %s" % (scene.index + 1, str(exc)[:200]), \
            [(s.asset_path, "Scene %d: %s (%s)" % (s.index + 1, s.query, s.source))
             for s in scenes if s.asset_path]
    scene.asset_path, scene.source, scene.verified, scene.reason = str(path), source, None, ""
    session["scenes"] = [s.__dict__ for s in scenes]
    gallery = [(s.asset_path, "Scene %d: %s (%s)" % (s.index + 1, s.query, s.source))
               for s in scenes if s.asset_path]
    return "Scene %d regenerated (%s)." % (scene.index + 1, source), gallery


def ai_build_script(session: Dict[str, Any]) -> str:
    """Scenes -> editable Markdown script for the renderer."""
    from .ai_video import Scene, build_script
    scenes = [Scene(**d) for d in session.get("scenes", [])]
    if not scenes:
        raise ValueError("Plan the scenes first.")
    return build_script(scenes, scenes[0].narration[:60])


def ai_render(script_text: str, provider: str, language: str, speed: float, vertical: bool,
              style: str, music: str, session: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    from .ai_video import Scene, build_script
    text = (script_text or "").strip()
    if not text:
        scenes = [Scene(**d) for d in session.get("scenes", [])]
        text = build_script(scenes) if scenes else ""
    if not text:
        raise ValueError("Plan the scenes first, or write a script.")
    return render_video_clip(text, provider, language, speed, vertical, style, music, None, session)


def _embed_html(candidates: List[Dict[str, Any]], limit: int = 6) -> str:
    """Inline players for the candidates (YouTube iframe, direct <video> otherwise)."""
    import html as html_lib
    frames = "".join(
        '<iframe width="320" height="180" src="https://www.youtube.com/embed/%s" '
        'title="%s" loading="lazy" frameborder="0" allowfullscreen></iframe>'
        % (c["id"], html_lib.escape(c["title"][:60]))
        if c.get("source", "youtube") == "youtube" else
        '<video width="320" src="%s" controls preload="none" '
        'title="%s"></video>'
        % (html_lib.escape(c["url"], quote=True), html_lib.escape(c["title"][:60]))
        for c in candidates[:limit])
    return '<div style="display:flex;flex-wrap:wrap;gap:8px">%s</div>' % frames


def clip_suggest(prompt: str, model: str, sources: int, session: Dict[str, Any]
                 ) -> Tuple[str, str]:
    """Prompt -> editable search queries, one per line (the model only suggests)."""
    from .ai_video import DEFAULT_MODELS, OpenRouterClient, best_value_models
    from .clips import suggest_searches
    from .errors import ProviderError
    if not (prompt or "").strip():
        raise ValueError("Write the prompt of the video you want to build.")
    client = OpenRouterClient()
    if model in ("", "best value (auto)"):
        model = best_value_models(client)[0]
    try:
        queries = suggest_searches(client, model, prompt, n=int(sources))
    except ProviderError:
        if model == DEFAULT_MODELS[0]:
            raise
        queries = suggest_searches(client, DEFAULT_MODELS[0], prompt, n=int(sources))
        model = "%s (fallback from %s)" % (DEFAULT_MODELS[0], model)
    return ("**%d** queries suggested by model **%s**. Edit them freely (one per line), "
            "then run the search." % (len(queries), model)), "\n".join(queries)


@_friendly
def clip_search(prompt: str, model: str, sources: int, queries: str, cookies: str,
                stock: List[str], pixabay_key: str, session: Dict[str, Any]
                ) -> Tuple[str, List[List[Any]], str, Any]:
    """Search queries (edited by the user, or suggested from the prompt) -> candidates
    from the selected sources (YouTube, Wikimedia Commons, Pixabay)."""
    from .ai_video import OpenRouterClient, best_value_models
    from .clips import search_commons, search_pixabay, search_youtube, suggest_searches
    from .errors import ConfigError, ProviderError
    if not (prompt or "").strip():
        raise ValueError("Write the prompt of the video you want to build.")
    session["cookies"] = cookies or ""
    status_lines = []
    manual = [q.strip() for q in (queries or "").splitlines() if q.strip()]
    if manual:
        queries_used = manual
    else:
        client = OpenRouterClient()
        if model in ("", "best value (auto)"):
            model = best_value_models(client)[0]
        status_lines.append("Searching with model **%s**..." % model)
        queries_used = suggest_searches(client, model, prompt, n=int(sources))
    candidates, seen = [], set()
    for query in queries_used:
        status_lines.append("- Searching: %s" % query)
        for source in (stock or ["YouTube"]):
            try:
                if source == "Wikimedia Commons":
                    items = search_commons(query, limit=4)
                elif source == "Pixabay":
                    items = search_pixabay(query, pixabay_key, limit=4)
                else:
                    items = search_youtube(query, limit=4, cookies=cookies)
            except (ConfigError, ProviderError) as exc:
                status_lines.append("  - %s unavailable: %s" % (source, str(exc)[:150]))
                continue
            for item in items:
                if item["id"] not in seen:
                    seen.add(item["id"])
                    candidates.append({"query": query, **item})
    session["clip_candidates"] = candidates
    session["clip_labels"] = {}
    session["clip_titles"] = {c["url"]: c["title"] for c in candidates}
    labels = []
    for c in candidates:
        label = "%s — %s (%s)" % (c["title"][:70], c["channel"] or "?",
                                  "%d:%02d" % (int(c["duration"] // 60), int(c["duration"] % 60)))
        labels.append(label)
        session["clip_labels"][label] = c["url"]
    rows = [[i + 1, c["title"], c["channel"] or "?", c["query"]]
            for i, c in enumerate(candidates)]
    status = ("**%d** searches, **%d** candidate videos (people-focused titles and videos over "
              "30 min filtered out). Watch the previews, then select the videos "
              "to download below.\n\n%s"
              % (len(queries_used), len(candidates), "\n".join(status_lines)))
    return status, rows, _embed_html(candidates), gr.update(choices=labels, value=[])


def clip_build(picks: List[str], manual_urls: str, model: str, all_clips: bool, n_clips: int,
               clip_seconds: float, stills: bool, delay: float, video_duration: float,
               session: Dict[str, Any]
               ) -> Any:
    """Selected (or manually pasted) source URLs -> audio-free, classified clip library.

    A generator: streams the work log with elapsed times into the status box while
    the batch runs. The final video duration drives the estimate: how many shots
    (clips) and sources are needed, and the batch stops as soon as enough clips
    are cut.
    """
    import queue as queue_lib
    import threading
    import time as time_lib
    from .ai_video import OpenRouterClient, best_value_models
    from .clips import build_library, estimate_needs
    labels = session.get("clip_labels") or {}
    links = [labels[p] for p in (picks or []) if p in labels]
    links += [u for u in (manual_urls or "").split() if u.startswith("http")]
    seen, urls = set(), []
    for link in links:
        if link not in seen:
            seen.add(link)
            urls.append(link)
    if not urls:
        # Generators cannot be wrapped with _friendly (the body runs at iteration
        # time), so every raise below must be a gr.Error to be readable in the UI.
        raise gr.Error("Select at least one video source (or paste a link manually).")
    shots, sources_needed = estimate_needs(float(video_duration), float(clip_seconds))
    target_clips = int(shots * 1.5)
    client = OpenRouterClient()
    if model in ("", "best value (auto)"):
        model = best_value_models(client)[0]
    workdir = Path(session["dir"])
    status_lines = ["Target: a **%d s** video → **%d** shots of ~%d s → up to **%d** clips from "
                    "**%d** source(s); the batch stops as soon as enough clips are cut."
                    % (int(video_duration), shots, int(clip_seconds), target_clips,
                       sources_needed),
                    "Working with model **%s**..." % model]
    empty = ([], [], [], gr.update())
    yield ("\n".join(status_lines), *empty)
    messages: queue_lib.Queue = queue_lib.Queue()
    result: Dict[str, Any] = {}

    def worker() -> None:
        try:
            result["library"] = build_library(urls, workdir, client, model,
                                              n_clips=None if all_clips else int(n_clips),
                                              clip_seconds=float(clip_seconds), stills=bool(stills),
                                              delay=float(delay), cookies=session.get("cookies", ""),
                                              titles=session.get("clip_titles") or {},
                                              target_clips=target_clips,
                                              progress=messages.put)
        except Exception as exc:
            result["error"] = exc

    started = time_lib.time()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        try:
            msg = messages.get(timeout=1.0)
            while True:
                try:
                    msg = messages.get_nowait()
                except queue_lib.Empty:
                    break
            status_lines.append("- [%ds] %s" % (int(time_lib.time() - started), msg))
            yield ("\n".join(status_lines), *empty)
        except queue_lib.Empty:
            pass
    thread.join()
    if "error" in result:
        exc = result["error"]
        detail = str(exc).strip() or exc.__class__.__name__
        raise gr.Error("%s: %s" % (exc.__class__.__name__, detail[:400])) from exc
    library = result["library"]
    if not library["clips"]:
        detail = " / ".join(library.get("errors", [])[:3]) or (
            "nothing was downloaded: check the sources and the disk space")
        raise gr.Error("Step 5 produced no clips: %s" % detail[:400])
    status_lines.append("- [%ds] done" % int(time_lib.time() - started))
    session["library"] = library
    picks_all = library["clips"] + library["frames"]
    status = ("**%d** audio-free clips and **%d** frames from %d video(s), model **%s**. "
              "Catalog: **%d** entries. Source videos are deleted after cutting to save disk "
              "space.\n\n%s"
              % (len(library["clips"]), len(library["frames"]), len(library["sources"]),
                 model, len(library["catalog"]), "\n".join(status_lines)))
    for error in library.get("errors", []):
        status += "\n\n⚠️ %s" % error
    for seg in library["segments"]:
        status += "\n- %s — %s (%.0fs→%.0fs)" % (seg["title"], seg["source"],
                                                  seg["start"], seg["end"])
    frames_gallery = [(f, Path(f).parent.name) for f in library["frames"][:60]]
    from .clips import load_catalog
    catalog_rows = [[entry.get("id"), entry.get("source_title", "")[:40],
                     entry.get("description", "")[:160], ", ".join(entry.get("tags") or [])[:120]]
                    for entry in library["catalog"]]
    yield status, library["clips"], frames_gallery, catalog_rows, \
        gr.update(choices=picks_all, value=[])


def clip_catalog_rows(session: Dict[str, Any]) -> List[List[Any]]:
    """Catalog database -> table rows (clip, source, description, tags)."""
    from .clips import load_catalog
    return [[entry.get("id"), entry.get("source_title", "")[:40],
             entry.get("description", "")[:160], ", ".join(entry.get("tags") or [])[:120]]
            for entry in load_catalog(Path(session["dir"]))]


@_friendly
def clip_select(prompt: str, model: str, language: str, video_duration: float,
                clip_seconds: float, session: Dict[str, Any], narration: str = "",
                match_narration: bool = True) -> Tuple[str, str, List[List[Any]]]:
    """Prompt + clip catalog -> the model picks the best clips and writes the script.

    With a narration (the step-1 script, editable) the model only pairs each paragraph with
    the best clip and the text is used exactly as written. Without a narration the model
    also writes one spoken sentence per shot; with ``match_narration`` those are written to
    read in about the target duration."""
    from .ai_video import OpenRouterClient, best_value_models
    from .clips import (build_video_script, estimate_needs, estimate_speech_seconds,
                        load_catalog, select_clips)
    catalog = load_catalog(Path(session["dir"]))
    if not catalog:
        raise ValueError("The clip library is empty: download some clips first.")
    client = OpenRouterClient()
    if model in ("", "best value (auto)"):
        model = best_value_models(client)[0]
    lang = "the same language as the prompt" if language in ("", "auto") else language
    shots, _ = estimate_needs(float(video_duration), float(clip_seconds))
    narration = (narration or session.get("wizard_narration") or "").strip()
    if narration:
        selected = select_clips(client, model, prompt, catalog, language=lang,
                                max_shots=shots, narration=narration)
        script = build_video_script(selected, title=prompt[:60])
        session["selected_clips"] = [item["entry"].get("id") for item in selected
                                     if item["entry"].get("id")]
        rows = [[item["entry"].get("id") or "-",
                 item["entry"].get("description", "")[:120], item["narration"]]
                for item in selected]
        reading = estimate_speech_seconds(script)
        status = ("Selected **%d** clips for the **%d** narration scenes with model **%s**. "
                  "The narration is used **exactly as written** (reading time about "
                  "**%.0f s**); edit the script below only to change wording, not the "
                  "footage."
                  % (sum(1 for item in selected if item["entry"].get("id")),
                     len(selected), model, reading))
        return status, script, rows
    target = float(video_duration) if match_narration else 0.0
    selected = select_clips(client, model, prompt, catalog, language=lang, max_shots=shots,
                            target_seconds=target)
    script = build_video_script(selected, title=prompt[:60])
    session["selected_clips"] = [item["entry"]["id"] for item in selected]
    rows = [[item["entry"]["id"], item["entry"].get("description", "")[:120], item["narration"]]
            for item in selected]
    status = ("Selected **%d** clips from the catalog (**%d** available) with model **%s**."
              % (len(selected), len(catalog), model))
    reading = estimate_speech_seconds(script)
    status += " The narration reads in about **%.0f s**." % reading
    if match_narration:
        status += (" Target was %.0f s: to stretch or shorten the video, move the duration "
                   "slider and re-select, or edit the script text." % float(video_duration))
    return status, script, rows


@_friendly
def clip_render(script_text: str, provider: str, language: str, speed: float, voice: str,
                vertical: bool, style: str, music: str, transition_style: str = "mix",
                transition: float = 0.5, session: Optional[Dict[str, Any]] = None
                ) -> Tuple[str, Optional[str]]:
    """Render the clip-based script with the chosen narrator voice and transitions."""
    text = (script_text or "").strip()
    if not text:
        raise ValueError("Select the clips and build the script first.")
    return render_video_clip(text, provider, language, speed, vertical, style, music,
                             None, session, voice=voice or None,
                             transition_style=transition_style,
                             transition=float(transition))


@_friendly
def clip_preview(script_text: str, vertical: bool, session: Dict[str, Any]
                 ) -> Tuple[str, Optional[str]]:
    """HeyGen-style fast draft: silent, low-resolution, small file — check the flow first."""
    text = (script_text or "").strip()
    if not text:
        raise ValueError("Select the clips and build the script first.")
    status, out = render_video_clip(text, "dummy", "auto", 1.0, vertical, "clean", "", None,
                                    session, small=True, output_name="preview.mp4")
    session["preview_path"] = out
    session["vertical"] = bool(vertical)
    return status, out


@_friendly
def clip_polish(script_text: str, session: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    """AI review & polish: watch the preview, score it, fix what is fixable, re-render."""
    from .ai_video import OpenRouterClient, best_value_models
    from .clips import load_catalog
    from .polish import polish_video
    video = session.get("preview_path")
    if not video or not Path(video).is_file():
        raise ValueError("Run the quick preview first (step 8): the polish reviews that "
                         "render.")
    client = OpenRouterClient()
    model = best_value_models(client)[0]
    catalog = load_catalog(Path(session["dir"]))
    log: List[str] = []

    def rerender(script: str) -> str:
        _, out = clip_preview(script, session.get("vertical", False), session)
        return out

    result = polish_video(client, model, video, script_text,
                          Path(session["dir"]) / "polish",
                          overlay_plan=session.get("overlay_plan") or [],
                          catalog=catalog, rerender=rerender,
                          progress=lambda msg: log.append("- %s" % msg))
    session["preview_path"] = result["video"]
    status = ("Polish completato with model **%s**: score **%.0f/10** after **%d** round(s)."
              "\n\n%s" % (model, result["score"], result["rounds"], "\n".join(log)))
    if result["fixes"]:
        status += "\n\n**Applied fixes:** " + "; ".join(result["fixes"]) + "."
    else:
        status += "\n\nNo automatic fix was needed or available."
    return status, result["script"], result["video"]


@_friendly
def clip_illustrate(script_text: str, every: int, style: str, session: Dict[str, Any],
                    plan: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str]:
    """Presentation style: transparent cards (text, icons, lines) over the video shots.

    Every N narration blocks get a `![](card.png){overlay}` line: the renderer composites
    the card on top of that shot's clip (slide-in + gentle drift) while the video keeps
    playing underneath. Cards are drawn locally from the overlay plan designed in the
    wizard (title, bullet points, icon concepts); without a plan the narration line
    becomes the card title. The script stays editable; failures never abort the batch."""
    from .overlays import render_card
    text = (script_text or "").strip()
    if not text:
        raise ValueError("Build the script first (Select the best clips…).")
    every = max(1, int(every or 1))
    workdir = Path(session["dir"]) / "illustrations"
    workdir.mkdir(parents=True, exist_ok=True)
    if plan is None:
        plan = session.get("overlay_plan") or []
    plan_by_scene: Dict[int, Dict[str, Any]] = {}
    for item in plan:
        try:
            plan_by_scene[int(item.get("scene", 0))] = item
        except (TypeError, ValueError):
            continue
    out, notes, added, narrated = [], [], 0, 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("![") or stripped.startswith("#"):
            out.append(line)
            continue
        narrated += 1
        if narrated % every == 0:
            added += 1
            item = dict(plan_by_scene.get(narrated) or {"title": stripped[:80],
                                                        "bullets": [], "icons": []})
            try:
                path = render_card(item, workdir / ("overlay_%02d" % added), style=style)
            except ConfigError as exc:
                raise
            except Exception as exc:
                notes.append("Overlay %d failed: %s" % (added, str(exc)[:140]))
                path = None
            if path is not None:
                out.append("![](%s){overlay}" % path)
        out.append(line)
    if not added:
        raise ValueError("No narration shot found to overlay.")
    status = ("Added **%d** overlay cards (style **%s**): text, icons and lines drawn over "
              "the clips, one every %d narration shot(s). The video keeps playing underneath "
              "— run the quick preview to check it."
              % (added - len(notes), style or "Scuro", every))
    for note in notes:
        status += "\n\n⚠️ %s" % note
    return status, "\n".join(out).strip() + "\n"


@_friendly
def wizard_narrate(prompt: str, model: str, language: str, duration: float,
                   session: Dict[str, Any]) -> Tuple[str, str, str, Any]:
    """Step 1: prompt -> narration script + subtitles advice + reading-time duration."""
    from .ai_video import OpenRouterClient, best_value_models
    from .clips import estimate_speech_seconds, plan_narration
    if not (prompt or "").strip():
        raise ValueError("Write the prompt of the video you want to build.")
    client = OpenRouterClient()
    if model in ("", "best value (auto)"):
        model = best_value_models(client)[0]
    plan = plan_narration(client, model, prompt, language=language, duration=float(duration))
    narration = plan["narration"]
    session["wizard_narration"] = narration
    reading = estimate_speech_seconds(narration)
    scenes = len([p for p in narration.split("\n\n") if p.strip()])
    status = ("Narration written with model **%s**: **%d** scene paragraphs, reading time "
              "about **%.0f s**. Edit it freely, then move on." % (model, scenes, reading))
    advice = ("**Subtitles: %s** — %s\n\nChoose the caption style in the last step "
              "('none' = no captions)." % ("yes" if plan["subtitles"] else "no",
                                             plan["reason"] or "-"))
    return status, narration, advice, gr.update(value=float(max(15, min(300, round(reading)))))


@_friendly
def reading_time(narration: str) -> Tuple[str, Any]:
    """Step 2: narration -> reading-time display + target-duration slider update."""
    from .clips import estimate_speech_seconds
    reading = estimate_speech_seconds(narration or "")
    return ("Reading time of the narration: **%.0f s** — by default the video lasts this "
            "long; move the duration slider in step 1 to stretch or shorten it." % reading), \
        gr.update(value=float(max(15, min(300, round(reading)))))


@_friendly
def wizard_scenes(narration: str, prompt: str, model: str, session: Dict[str, Any]
                  ) -> Tuple[str, List[List[Any]], str]:
    """Step 3: narration -> what each scene should show + stock-search queries."""
    from .ai_video import OpenRouterClient, best_value_models
    from .clips import plan_scene_queries
    if not (narration or "").strip():
        raise ValueError("Write the narration first (step 1).")
    client = OpenRouterClient()
    if model in ("", "best value (auto)"):
        model = best_value_models(client)[0]
    scenes = plan_scene_queries(client, model, narration, prompt)
    session["wizard_scenes"] = scenes
    rows = [[i + 1, s["scene"], s["query"]] for i, s in enumerate(scenes)]
    queries = "\n".join(s["query"] for s in scenes)
    return ("**%d** scenes defined with model **%s**. The search queries below are prefilled "
            "from the scenes — edit them freely before searching." % (len(scenes), model)), \
        rows, queries


@_friendly
def wizard_overlays(narration: str, prompt: str, model: str, session: Dict[str, Any]
                    ) -> Tuple[str, List[List[Any]]]:
    """Step 4: narration -> the text/icon elements of the overlay layer (presentation)."""
    from .ai_video import OpenRouterClient, best_value_models
    from .clips import plan_overlays
    if not (narration or "").strip():
        raise ValueError("Write the narration first (step 1).")
    client = OpenRouterClient()
    if model in ("", "best value (auto)"):
        model = best_value_models(client)[0]
    overlays = plan_overlays(client, model, narration, prompt)
    session["overlay_plan"] = overlays
    rows = [[o["scene"], o["title"], "; ".join(o["bullets"]), ", ".join(o["icons"])]
            for o in overlays]
    return ("**%d** overlay elements designed with model **%s**. They become transparent "
            "infographics over the clips when you run *Add overlays to the script* "
            "(step 7)." % (len(overlays), model)), rows


def set_presentation(choice: str, session: Dict[str, Any]) -> str:
    """Step 4: keep or drop the overlay plan when the user switches style."""
    if choice == "yes":
        return "Presentation style **on**: design the overlay elements below, then apply " \
            "them in step 7."
    session.pop("overlay_plan", None)
    return "Presentation style **off**: no overlay layer, plain video with (optional) " \
        "captions."


@_friendly
def wizard_more_queries(prompt: str, model: str, sources: int, queries: str,
                        session: Dict[str, Any]) -> Tuple[str, str]:
    """Step 5: append fresh alternative queries for candidates the user does not like."""
    from .ai_video import OpenRouterClient, best_value_models
    from .clips import suggest_searches
    if not (prompt or "").strip():
        raise ValueError("Write the prompt of the video you want to build.")
    client = OpenRouterClient()
    if model in ("", "best value (auto)"):
        model = best_value_models(client)[0]
    existing = [q.strip() for q in (queries or "").splitlines() if q.strip()]
    fresh = suggest_searches(client, model, prompt, n=int(sources))
    added = [q for q in fresh if q.lower() not in {e.lower() for e in existing}]
    if not added:
        return "The model returned no new queries — edit the list manually.", queries
    return ("Added **%d** alternative queries with model **%s**. Search again to see new "
            "candidates." % (len(added), model)), "\n".join(existing + added)


def voice_choices(provider: str) -> Any:
    """Dropdown update with every voice id of the selected engine."""
    try:
        from .studio import Studio
        voices, _ = Studio(provider or "auto").provider.list_voices(limit=1000, offset=0)
        return gr.update(choices=[v.voice_id for v in voices])
    except Exception:
        return gr.update(choices=[])



def build_ui(workdir: Optional[Path] = None) -> Any:
    """Assemble the Gradio app (call ``.launch()`` on the result to serve it)."""
    examples = example_scripts()
    with gr.Blocks(title="AI Voiceover Studio") as app:
        session = gr.State(new_session(workdir))
        gr.Markdown("# AI Voiceover Studio\n"
                    "Turn a script into narration, subtitles and video. "
                    "Default engine: free Edge neural voices, no API key.",
                    )

        with gr.Tab("Voiceover"):
            script = gr.Textbox(label="Script (Markdown)", value=DEFAULT_SCRIPT,
                                lines=14, placeholder="# Title\n\n[NARRATOR] Your narration...")
            with gr.Accordion("Examples", open=False):
                example = gr.Dropdown(choices=examples, label="Load an example script",
                                      interactive=True)
            with gr.Row():
                provider = gr.Dropdown(choices=_engine_choices(), value="auto", label="Engine")
                language = gr.Dropdown(choices=LANGUAGES, value="auto", label="Language")
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
            with gr.Row():
                gap = gr.Slider(0.0, 2.0, value=0.35, step=0.05,
                                label="Pause between segments (s)")
                output_name = gr.Textbox(value="voiceover.mp3", label="Output file name")
            generate = gr.Button("Generate voiceover", variant="primary")
            status = gr.Markdown()
            audio = gr.Audio(label="Voiceover")
            with gr.Row():
                srt = gr.File(label="Subtitles (SRT)")
            segments = gr.Dataframe(headers=["#", "Speaker", "Voice", "Start", "End", "Text"],
                                    label="Segments")
            if examples:
                example.change(lambda path: Path(path).read_text(encoding="utf-8"),
                               inputs=example, outputs=script)
            generate.click(generate_voiceover,
                           inputs=[script, provider, language, speed, gap, output_name, session],
                           outputs=[status, audio, srt, segments])

        with gr.Tab("Quick line"):
            line = gr.Textbox(label="Text to speak", lines=3)
            with gr.Row():
                say_provider = gr.Dropdown(choices=_engine_choices(), value="auto", label="Engine")
                say_voice = gr.Textbox(label="Voice id (optional, e.g. en-US-AvaNeural)")
                say_language = gr.Dropdown(choices=LANGUAGES, value="auto", label="Language")
                say_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
            say_button = gr.Button("Speak", variant="primary")
            say_status = gr.Markdown()
            say_audio = gr.Audio(label="Result")
            say_button.click(say_line,
                             inputs=[line, say_provider, say_voice, say_language, say_speed, session],
                             outputs=[say_status, say_audio])

        with gr.Tab("Voices"):
            with gr.Row():
                voices_provider = gr.Dropdown(choices=_engine_choices(), value="auto", label="Engine")
                voices_filter = gr.Textbox(label="Filter (id, name or language)")
                voices_button = gr.Button("List voices")
            voices_status = gr.Markdown()
            voices_table = gr.Dataframe(headers=["Voice id", "Language", "Name"], label="Voices")
            with gr.Row():
                preview_voice = gr.Textbox(label="Selected voice (click a row above)", interactive=False)
                preview_button = gr.Button("Test voice")
            preview_audio = gr.Audio(label="Preview")
            voices_button.click(list_voices, inputs=[voices_provider, voices_filter],
                                outputs=[voices_status, voices_table])
            def on_voice_select(evt: gr.SelectData, rows: List[List[Any]]) -> str:
                return pick_voice(evt, rows)

            voices_table.select(on_voice_select, inputs=voices_table, outputs=preview_voice)
            preview_button.click(test_voice, inputs=[preview_voice, voices_provider, session],
                                 outputs=[voices_status, preview_audio])

        with gr.Tab("Video"):
            gr.Markdown("Renders an MP4 with captions and optional music. Needs "
                        "[ffmpeg](https://ffmpeg.org) on the machine running the studio.")
            with gr.Row():
                video_vertical = gr.Checkbox(value=False, label="Vertical 9:16 (Shorts / TikTok)")
                video_style = gr.Dropdown(choices=["auto", "tiktok", "bold", "clean", "minimal",
                                                  "box", "none"],
                                          value="auto", label="Caption style ('none' = no captions)")
                video_music = gr.Dropdown(choices=["", "ambient", "pulse"], value="",
                                          label="Background music")
            broll = gr.File(label="B-roll images or video clips (optional, used in upload order "
                                  "for paragraphs without ![](...)",
                            file_count="multiple", file_types=["image", "video"])
            with gr.Row():
                video_transition = gr.Dropdown(choices=["mix", "fade", "slide", "zoom", "cut"],
                                               value="mix", label="Transition style")
                video_transition_time = gr.Slider(0, 1.5, value=0.5, step=0.1,
                                                  label="Transition duration (s)")
            library_pick = gr.CheckboxGroup(choices=[], label="Reuse clips or frames from the "
                                                             "Clip studio library", value=[])
            video_button = gr.Button("Render video", variant="primary")
            video_status = gr.Markdown()
            video_out = gr.Video(label="Video")
            video_button.click(render_video_clip,
                               inputs=[script, provider, language, speed, video_vertical,
                                       video_style, video_music, broll, session, library_pick,
                                       video_transition, video_transition_time],
                               outputs=[video_status, video_out])

        with gr.Tab("AI Video"):
            gr.Markdown("Describe the video you want: an AI plans the scenes, finds keyless "
                        "stock visuals (Wikimedia, Openverse) or generates them, optionally "
                        "verifies each asset with a vision model, then renders the MP4. "
                        "Needs `OPENROUTER_API_KEY`.")
            ai_prompt = gr.Textbox(label="Prompt", lines=3,
                                   placeholder="Es.: una breve storia sulle piramidi egizie")
            with gr.Row():
                ai_duration = gr.Slider(10, 120, value=30, step=5, label="Target duration (s)")
                ai_language = gr.Dropdown(choices=LANGUAGES, value="auto", label="Language")
                ai_orientation = gr.Dropdown(choices=["landscape", "portrait"], value="landscape",
                                             label="Orientation")
            with gr.Row():
                ai_model = gr.Dropdown(choices=DEFAULT_MODELS, value=DEFAULT_MODELS[0],
                                       allow_custom_value=True,
                                       label="Planner model (OpenRouter)")
                ai_vision_model = gr.Dropdown(choices=DEFAULT_MODELS,
                                              value=DEFAULT_MODELS[1], allow_custom_value=True,
                                              label="Vision model (asset verification)")
                ai_verify = gr.Checkbox(value=True, label="Verify assets with vision")
            ai_plan_button = gr.Button("Plan scenes and find assets", variant="primary")
            ai_status = gr.Markdown()
            with gr.Row():
                ai_gallery = gr.Gallery(label="Scene assets", columns=4, height=260)
                ai_scenes = gr.Dataframe(headers=["#", "Narration", "Query", "Source", "Vision"],
                                         label="Scene plan")
            with gr.Row():
                ai_scene_pick = gr.Dropdown(label="Scene to regenerate", choices=[],
                                            allow_custom_value=True)
                ai_regen_button = gr.Button("Regenerate asset (new generated image)")
            ai_build_button = gr.Button("Build the script from these scenes")
            ai_script = gr.Textbox(label="Generated script (editable)", lines=10)
            with gr.Row():
                ai_vertical = gr.Checkbox(value=False, label="Vertical 9:16")
                ai_style = gr.Dropdown(choices=["auto", "tiktok", "bold", "clean", "minimal",
                                                "box", "none"],
                                       value="auto", label="Caption style ('none' = no captions)")
                ai_music = gr.Dropdown(choices=["", "ambient", "pulse"], value="ambient",
                                       label="Background music")
            ai_render_button = gr.Button("Render video from this script", variant="primary")
            ai_video_status = gr.Markdown()
            ai_video_out = gr.Video(label="Video")

            def _scene_choices(session_data):
                return gr.update(choices=[str(d["index"] + 1) for d in session_data.get("scenes", [])])

            ai_plan_button.click(ai_plan,
                                 inputs=[ai_prompt, ai_duration, ai_language, ai_orientation,
                                         ai_model, ai_verify, ai_vision_model, session],
                                 outputs=[ai_status, ai_gallery, ai_scenes])
            ai_plan_button.click(_scene_choices, inputs=session, outputs=ai_scene_pick)
            ai_regen_button.click(ai_regenerate,
                                  inputs=[ai_scene_pick, ai_orientation, session],
                                  outputs=[ai_status, ai_gallery])
            ai_build_button.click(ai_build_script, inputs=session, outputs=ai_script)
            ai_render_button.click(ai_render,
                                   inputs=[ai_script, provider, ai_language, speed, ai_vertical,
                                           ai_style, ai_music, session],
                                   outputs=[ai_video_status, ai_video_out])

        with gr.Tab("Video wizard"):
            gr.Markdown("Guided flow, one step at a time: the AI writes the narration from "
                        "your prompt, you pick the voice (the video duration follows the "
                        "reading time), the AI describes what each scene should show, you "
                        "choose the presentation style (with or without a transparent overlay "
                        "layer), the matching clips are found, previewed and selected, you "
                        "pick the transitions, and finally you get a fast preview and the "
                        "final render. Needs `OPENROUTER_API_KEY`, "
                        "[yt-dlp](https://github.com/yt-dlp/yt-dlp) "
                        "(`pip install \"ai-voiceover-studio[clip]\"`), a JavaScript runtime "
                        "(deno) and ffmpeg.")
            with gr.Accordion("Step 1 — Prompt and narration", open=True):
                clip_prompt = gr.Textbox(label="Prompt of the video to build", lines=2,
                                         placeholder="Es.: video promozionale sull'enoturismo "
                                                     "nelle Marche, stile documentaristico")
                with gr.Row():
                    clip_language = gr.Dropdown(choices=LANGUAGES, value="auto",
                                                label="Language")
                    clip_duration = gr.Slider(15, 300, value=60, step=5,
                                              label="Target duration (s) — set to the "
                                                    "narration reading time after step 1; "
                                                    "move it to stretch or shorten the video")
                clip_model = gr.Dropdown(choices=["best value (auto)"] + DEFAULT_MODELS,
                                         value="best value (auto)", allow_custom_value=True,
                                         label="Model (OpenRouter) — 'best value' picks the "
                                               "cheapest capable model")
                clip_narrate_button = gr.Button("1. Write the narration", variant="primary")
                clip_narrate_status = gr.Markdown()
                clip_narration = gr.Textbox(label="Narration (editable, one paragraph per "
                                                  "scene)", lines=8)
                clip_subtitles = gr.Markdown()
            with gr.Accordion("Step 2 — Voice and duration", open=False):
                with gr.Row():
                    clip_voice = gr.Dropdown(choices=[], label="Narrator voice",
                                             allow_custom_value=True)
                    clip_voices_button = gr.Button("Load voices")
                clip_reading = gr.Markdown()
            with gr.Accordion("Step 3 — Scenes", open=False):
                clip_scenes_button = gr.Button("2. Define what each scene shows",
                                               variant="primary")
                clip_scenes_status = gr.Markdown()
                clip_scenes_table = gr.Dataframe(headers=["#", "Scene", "Search query"],
                                                 label="Scene plan")
            with gr.Accordion("Step 4 — Presentation style (overlay layer)", open=False):
                clip_presentation = gr.Radio(choices=["no", "yes"], value="no",
                                             label="Overlay layer with elements that explain "
                                                   "better (text and icons floating over the "
                                                   "video)?")
                clip_presentation_status = gr.Markdown()
                with gr.Row():
                    clip_every = gr.Dropdown(choices=["1", "2", "3"], value="2",
                                             label="Overlay every N scenes")
                    clip_illust_style = gr.Dropdown(choices=list(CARD_STYLES),
                                                    value="Scuro",
                                                    label="Card style (text, icons, lines "
                                                          "drawn locally — no AI image)")
                clip_overlays_button = gr.Button("3. Design the overlay elements")
                clip_overlays_status = gr.Markdown()
                clip_overlays_table = gr.Dataframe(headers=["Scene", "Title", "Points",
                                                            "Icons"],
                                                   label="Overlay elements (presentation "
                                                         "layer)")
            with gr.Accordion("Step 5 — Find the clips", open=False):
                with gr.Row():
                    clip_stock = gr.CheckboxGroup(choices=["YouTube", "Wikimedia Commons",
                                                           "Pixabay"],
                                                  value=["YouTube", "Wikimedia Commons"],
                                                  label="Sources to search (Commons is "
                                                        "keyless and has precise descriptions, "
                                                        "no bot checks)")
                    clip_pixabay = gr.Textbox(label="Pixabay API key (optional) — free, shown "
                                                    "at the top of pixabay.com/api/docs/ when "
                                                    "logged in", value="", lines=1)
                clip_cookies = gr.Textbox(label="Cookies file (optional) — path to a "
                                                "cookies.txt export, needed when YouTube asks "
                                                "to sign in", value="", lines=1,
                                          placeholder="/percorso/cookies.txt")
                clip_sources_number = gr.Slider(3, 10, value=5, step=1,
                                                label="Video sources to search (min 3)")
                clip_queries = gr.Textbox(label="Search queries (one per line, prefilled from "
                                                "the scenes, editable)", lines=5)
                with gr.Row():
                    clip_search_button = gr.Button("4. Search the sources", variant="primary")
                    clip_more_button = gr.Button("Suggest alternative queries")
                clip_search_status = gr.Markdown()
                clip_results = gr.Dataframe(headers=["#", "Title", "Channel", "Search"],
                                            label="Candidate videos")
                clip_previews = gr.HTML()
                clip_pick = gr.CheckboxGroup(choices=[], label="Select the videos to download "
                                                              "and cut (swap any that do not "
                                                              "look right before continuing)",
                                             value=[])
            with gr.Accordion("Step 6 — Download and cut", open=False):
                with gr.Row():
                    clip_all = gr.Checkbox(value=True,
                                           label="Extract every highlight (recommended)")
                    clip_n = gr.Slider(1, 6, value=2, step=1,
                                       label="…or this many clips per video")
                    clip_len = gr.Slider(5, 30, value=10, step=1, label="Clip length (s)")
                    clip_delay = gr.Slider(0, 30, value=10, step=1,
                                           label="Pause between YouTube requests (s, "
                                                 "rate-limit protection)")
                clip_stills = gr.Checkbox(value=True,
                                          label="Also extract still frames from the clips")
                clip_manual = gr.Textbox(label="Or paste video links manually (optional)",
                                         lines=2,
                                         placeholder="https://www.youtube.com/watch?v=...")
                clip_button = gr.Button("5. Download selected videos, extract and classify "
                                        "all clips", variant="primary")
                clip_status = gr.Markdown()
                with gr.Row():
                    clip_files = gr.Files(label="Cut clips — audio-free (reuse them as B-roll)")
                    clip_frames = gr.Gallery(label="Extracted frames", columns=6, height=260)
                clip_catalog = gr.Dataframe(headers=["Clip", "Source", "Description", "Tags"],
                                            label="Clip catalog (database, saved on disk)")
            with gr.Accordion("Step 7 — Selection, script and overlays", open=False):
                clip_select_button = gr.Button("6. Select the best clips and write the "
                                               "script", variant="primary")
                clip_select_status = gr.Markdown()
                clip_script = gr.Textbox(label="Final script (editable)", lines=8)
                clip_selected = gr.Dataframe(headers=["Clip", "Description", "Narration"],
                                             label="Chosen shots")
                clip_illustrate_button = gr.Button("7. Add overlays to the script "
                                                   "(presentation style)")
                clip_illust_status = gr.Markdown()
            with gr.Accordion("Step 8 — Transitions", open=False):
                clip_transition = gr.Radio(choices=["mix", "fade", "slide", "zoom", "cut"],
                                           value="mix",
                                           label="Animation between one clip and the next")
                clip_transition_time = gr.Slider(0, 1.5, value=0.5, step=0.1,
                                                 label="Transition duration (s)")
            with gr.Accordion("Step 9 — Preview and final video", open=False):
                with gr.Row():
                    clip_vertical = gr.Checkbox(value=False, label="Vertical 9:16")
                    clip_style = gr.Dropdown(choices=["auto", "tiktok", "bold", "clean",
                                                      "minimal", "box", "none"],
                                             value="auto",
                                             label="Caption style ('none' = no captions)")
                    clip_music = gr.Dropdown(choices=["", "ambient", "pulse"], value="ambient",
                                             label="Background music")
                clip_preview_button = gr.Button("8. Quick preview (silent, fast, low "
                                                "resolution)")
                clip_preview_status = gr.Markdown()
                clip_preview_out = gr.Video(label="Preview draft")
                clip_polish_button = gr.Button("9. AI polish — the model watches the preview "
                                               "and improves it until professional")
                clip_polish_status = gr.Markdown()
                clip_render_button = gr.Button("10. Render final video", variant="primary")
                clip_video_status = gr.Markdown()
                clip_video_out = gr.Video(label="Final video")

            clip_narrate_button.click(wizard_narrate,
                                      inputs=[clip_prompt, clip_model, clip_language,
                                              clip_duration, session],
                                      outputs=[clip_narrate_status, clip_narration,
                                               clip_subtitles, clip_duration])
            clip_narration.change(reading_time, inputs=clip_narration,
                                  outputs=[clip_reading, clip_duration])
            clip_voices_button.click(voice_choices, inputs=[provider], outputs=[clip_voice])
            clip_scenes_button.click(wizard_scenes,
                                     inputs=[clip_narration, clip_prompt, clip_model, session],
                                     outputs=[clip_scenes_status, clip_scenes_table,
                                              clip_queries])
            clip_presentation.change(set_presentation,
                                     inputs=[clip_presentation, session],
                                     outputs=[clip_presentation_status])
            clip_overlays_button.click(wizard_overlays,
                                       inputs=[clip_narration, clip_prompt, clip_model,
                                               session],
                                       outputs=[clip_overlays_status, clip_overlays_table])
            clip_more_button.click(wizard_more_queries,
                                   inputs=[clip_prompt, clip_model, clip_sources_number,
                                           clip_queries, session],
                                   outputs=[clip_search_status, clip_queries])
            clip_search_button.click(clip_search,
                                     inputs=[clip_prompt, clip_model, clip_sources_number,
                                             clip_queries, clip_cookies, clip_stock,
                                             clip_pixabay, session],
                                     outputs=[clip_search_status, clip_results, clip_previews,
                                              clip_pick])
            clip_button.click(clip_build,
                              inputs=[clip_pick, clip_manual, clip_model, clip_all, clip_n,
                                      clip_len, clip_stills, clip_delay, clip_duration,
                                      session],
                              outputs=[clip_status, clip_files, clip_frames, clip_catalog,
                                       library_pick])
            clip_select_button.click(clip_select,
                                     inputs=[clip_prompt, clip_model, clip_language,
                                             clip_duration, clip_len, clip_narration,
                                             session],
                                     outputs=[clip_select_status, clip_script, clip_selected])
            clip_illustrate_button.click(clip_illustrate,
                                         inputs=[clip_script, clip_every,
                                                 clip_illust_style, session],
                                         outputs=[clip_illust_status, clip_script])
            clip_preview_button.click(clip_preview,
                                      inputs=[clip_script, clip_vertical, session],
                                      outputs=[clip_preview_status, clip_preview_out])
            clip_polish_button.click(clip_polish,
                                     inputs=[clip_script, session],
                                     outputs=[clip_polish_status, clip_script,
                                              clip_preview_out])
            clip_render_button.click(clip_render,
                                     inputs=[clip_script, provider, clip_language, speed,
                                             clip_voice, clip_vertical, clip_style, clip_music,
                                             clip_transition, clip_transition_time, session],
                                     outputs=[clip_video_status, clip_video_out])

    return app


def launch(host: str = "127.0.0.1", port: int = 7860, workdir: Optional[Path] = None,
           **kwargs: Any) -> None:
    """Build and serve the studio (blocking)."""
    app = build_ui(workdir=workdir)
    app.queue()
    kwargs.setdefault("show_error", True)
    app.launch(server_name=host, server_port=port, **kwargs)
