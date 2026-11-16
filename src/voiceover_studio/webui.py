"""Gradio web studio: write a script, pick engine, language and speed, listen.

    voiceover studio --host 127.0.0.1 --port 7860

Needs the optional web extra::

    pip install "ai-voiceover-studio[webui]"

The UI is a thin layer over :class:`voiceover_studio.studio.Studio` and
:func:`voiceover_studio.video.render_video`; every handler can also be called
directly from Python for tests.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ai_video import DEFAULT_MODELS
from .errors import ProviderError, VoiceoverError
from .providers import available_providers

import gradio as gr

LANGUAGES = ["auto", "en", "it", "es", "fr", "de", "pt", "nl", "ru", "uk", "pl",
             "tr", "ar", "hi", "zh", "ja", "ko"]

DEFAULT_SCRIPT = """# Welcome to AI Voiceover Studio

Write your narration here, one paragraph per idea.

[NARRATOR] Speaker tags like this give each speaker its own voice.

Add a pause with `[pause 500]`, or **bold** words to stress them in the captions.
"""


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


def save_broll(files: Optional[List[Any]], session: Dict[str, Any]) -> Optional[Path]:
    """Copy uploaded images/clips into the session in upload order; returns the folder."""
    if not files:
        return None
    folder = Path(session["dir"]) / "broll"
    folder.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(files, 1):
        src = Path(item if isinstance(item, (str, Path)) else item.name)
        shutil.copy(src, folder / ("%03d_%s" % (i, src.name)))
    return folder


def render_video_clip(script: str, provider: str, language: str, speed: float,
                      vertical: bool, style: str, music: str, broll: Optional[List[Any]],
                      session: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Render the script to an MP4 (needs ffmpeg). Returns (status, video)."""
    text = (script or "").strip()
    if not text:
        raise ValueError("Write a script first, or load an example.")
    from .studio import Studio
    from .video import render_video
    workdir = Path(session["dir"])
    script_path = workdir / "script.md"
    script_path.write_text(text, encoding="utf-8")
    studio = Studio(provider or "auto")
    out = workdir / "video.mp4"
    overrides = _voice_overrides(language, speed)
    result = render_video(studio, save_broll(broll, session), script_path, out,
                          size=(1080, 1920) if vertical else (1920, 1080),
                          style="auto" if style == "auto" else style,
                          music=music or None, voice_overrides=overrides)
    status = ("Video **%.1f s**, %d shots, %s captions, engine **%s**."
              % (result["duration"], result["shots"], result["style"], studio.provider.name))
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
                video_style = gr.Dropdown(choices=["auto", "tiktok", "bold", "clean", "minimal", "box"],
                                          value="auto", label="Caption style")
                video_music = gr.Dropdown(choices=["", "ambient", "pulse"], value="",
                                          label="Background music")
            broll = gr.File(label="B-roll images or video clips (optional, used in upload order "
                                  "for paragraphs without ![](...)",
                            file_count="multiple", file_types=["image", "video"])
            video_button = gr.Button("Render video", variant="primary")
            video_status = gr.Markdown()
            video_out = gr.Video(label="Video")
            video_button.click(render_video_clip,
                               inputs=[script, provider, language, speed, video_vertical,
                                       video_style, video_music, broll, session],
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
                ai_style = gr.Dropdown(choices=["auto", "tiktok", "bold", "clean", "minimal", "box"],
                                       value="auto", label="Caption style")
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

    return app


def launch(host: str = "127.0.0.1", port: int = 7860, workdir: Optional[Path] = None,
           **kwargs: Any) -> None:
    """Build and serve the studio (blocking)."""
    app = build_ui(workdir=workdir)
    app.queue()
    app.launch(server_name=host, server_port=port, **kwargs)
