"""AI-illustration slides: no stock or YouTube clips, only AI images brought to life.

Pipeline: narration (one paragraph per scene) -> one concrete image prompt per paragraph
(planned by the model, fully editable) -> illustration generated with an OpenRouter image
model -> animated into a looping GIF with a Ken Burns move (Pillow, offline and free) ->
text, frames and symbols drawn as a transparent card and baked over the animation.

The resulting ``.gif`` files are treated as video shots by the render engine (ffmpeg loops
them with ``-stream_loop``), so the final clips never look static and the narration audio,
captions and transitions all keep working unchanged.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .errors import ConfigError

IMAGE_SYSTEM = """You design the illustrations of a narrated video. For every numbered \
paragraph of the narration write ONE image prompt for an AI image model: a rich, concrete \
visual description of the single picture that best illustrates that paragraph (subject, \
setting, composition, colors), written in the same language as the narration so it can be \
used directly.

The image must contain NO words, letters, numbers or captions. Avoid people's faces and \
brands. Answer ONLY with a JSON list of strings, one per paragraph, in the same order."""

# Model output sizes per aspect ratio: small enough to keep the GIFs light, big enough for
# a 1080p render (the engine scales shots up; a slight upscale is invisible after grading).
ASPECT_SIZES: Dict[str, Tuple[int, int]] = {"16:9": (960, 540), "1:1": (900, 900),
                                            "9:16": (540, 960)}

# Motion presets, in the same order as video._moves (Ken Burns loop baked into the GIF).
MOTIONS = ["push-in", "pan", "pull-out", "pan + rise"]

GIF_OVERLAY_OPACITY = 0.92  # matches the renderer's OVERLAY_OPACITY


def _paragraphs(narration: str) -> List[str]:
    return [p.strip() for p in (narration or "").split("\n\n")
            if p.strip() and not p.strip().startswith("#")]


def plan_image_prompts(client: Any, model: str, narration: str, prompt: str = "") -> List[str]:
    """Narration (one paragraph per scene) -> one editable image prompt per paragraph."""
    from .ai_video import _parse_json
    paragraphs = _paragraphs(narration)
    if not paragraphs:
        raise ValueError("Write the narration first (step 1).")
    user = ("Context of the video: %s\n\n" % prompt.strip() if (prompt or "").strip() else "") \
        + "\n\n".join("%d. %s" % (i, p) for i, p in enumerate(paragraphs, 1))
    content = client.chat(model, [{"role": "system", "content": IMAGE_SYSTEM},
                                  {"role": "user", "content": user}])
    raw = _parse_json(content)
    if not isinstance(raw, list) or not raw:
        raise ConfigError("The model did not return the illustration prompts.")
    prompts = []
    for item in raw:
        if isinstance(item, dict):
            item = item.get("prompt", "")
        if not isinstance(item, str) or not item.strip():
            continue
        prompts.append(item.strip()[:600])
    # pad so paragraph i always has a prompt (first sentence as a fallback description)
    while len(prompts) < len(paragraphs):
        fallback = re.split(r"(?<=[.!?])\s+", paragraphs[len(prompts)])[0]
        prompts.append(fallback[:200])
    return prompts[:len(paragraphs)]


def animate_still(src: Any, dest_base: Any, *, overlay: Optional[Any] = None,
                  size: Tuple[int, int] = (960, 540), motion: str = "push-in",
                  zoom: float = 0.12, frames: int = 16, loop_seconds: float = 1.8) -> Path:
    """Turn a still image into a looping Ken Burns GIF (optionally under a transparent card).

    The move presets mirror the renderer's Ken Burns table; ``overlay`` is a card PNG drawn
    by :mod:`voiceover_studio.overlays` (text, icons, accent lines) baked at the 42%-width
    spot with the same opacity the render engine uses, so the GIF preview matches the video.
    """
    from .video import _moves
    try:
        from PIL import Image
    except ImportError:
        raise ConfigError("Pillow is required to animate the illustrations: "
                          "pip install Pillow (part of 'ai-voiceover-studio[webui]').") from None
    src, dest = Path(src), Path(str(dest_base) + ".gif")
    with Image.open(src) as still:
        base = still.convert("RGB")
    w, h = size
    max_zoom = 1.0 + zoom * 1.4  # worst-case crop margin across all move presets
    target_w = max(w, int(w * max_zoom + 0.5))
    target_h = max(h, int(h * max_zoom + 0.5))
    scale = max(target_w / base.width, target_h / base.height)
    base = base.resize((max(target_w, round(base.width * scale)),
                        max(target_h, round(base.height * scale))), Image.LANCZOS)
    card = None
    if overlay is not None:
        ow = max(1, round(w * 0.42))
        oh = max(1, round(ow * 900 / 1600))
        card = Image.open(Path(overlay)).convert("RGBA").resize((ow, oh), Image.LANCZOS)
        alpha = card.getchannel("A").point(lambda v: int(v * GIF_OVERLAY_OPACITY))
        card.putalpha(alpha)
    z0, z1, x0, x1, y0, y1 = _moves(zoom)[MOTIONS.index(motion) % len(_moves(zoom))]
    active = []
    for i in range(frames):
        u = i / max(1, frames - 1)
        z = z0 + (z1 - z0) * u
        cw, ch = base.width / z, base.height / z
        left = (base.width - cw) * (x0 + (x1 - x0) * u)
        top = (base.height - ch) * (y0 + (y1 - y0) * u)
        frame = base.crop((int(left), int(top), int(left + cw), int(top + ch))) \
            .resize((w, h), Image.LANCZOS)
        if card is not None:
            frame_rgba = frame.convert("RGBA")
            frame_rgba.alpha_composite(card, (w - card.width - max(16, round(w * 0.035)),
                                              h - card.height - max(16, round(h * 0.035))))
            frame = frame_rgba.convert("RGB")
        active.append(frame.convert("P", palette=Image.ADAPTIVE, colors=256))
    per_frame = max(20, int(round(loop_seconds * 1000 / frames)))
    active[0].save(dest, save_all=True, append_images=active[1:], duration=per_frame,
                   loop=0, optimize=False)
    return dest


def generate_stills(client: Any, model: str, prompts: List[str], out_dir: Path, *,
                    style: str = "", extra: str = "", aspect: str = "16:9",
                    progress: Optional[Callable[[str], None]] = None
                    ) -> Tuple[List[Optional[str]], List[str]]:
    """One illustration per prompt via the OpenRouter image model (the paid step).

    Returns ``(stills, errors)``: ``stills[i]`` is the image path or ``None`` when that
    illustration failed; a failed scene never aborts the batch."""
    from .ai_video import generate_illustration
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stills: List[Optional[str]] = []
    errors: List[str] = []
    for i, desc in enumerate(prompts, 1):
        if progress:
            progress("Generating illustration %d/%d with %s: %s"
                     % (i, len(prompts), model, desc[:80]))
        try:
            still = generate_illustration(client, desc, out_dir / ("still_%02d" % i),
                                          model=model, style=style, aspect_ratio=aspect,
                                          extra=extra)
            stills.append(str(still))
        except Exception as exc:
            stills.append(None)
            errors.append("Illustration %d failed: %s" % (i, str(exc)[:200]))
        if progress:
            progress("Illustration %d/%d done (%s)"
                     % (i, len(prompts), "ok" if stills[-1] else "failed"))
    return stills, errors


def animate_slides(stills: List[Optional[str]], out_dir: Path, *, paragraphs: List[str],
                   aspect: str = "16:9", motion: str = "push-in",
                   overlay_plan: Optional[List[Dict[str, Any]]] = None,
                   with_cards: bool = True, card_style: str = "Scuro",
                   progress: Optional[Callable[[str], None]] = None
                   ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Local, free step: Ken Burns GIFs (with the text/frame/symbol card baked when asked).

    Returns ``(slides, errors)``; ``slides[i]`` is ``{"scene": n, "gif": path}`` when the
    animation succeeded, otherwise nothing is appended and the scene falls back to an
    animated-text shot in the script."""
    from .overlays import render_card
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_scene = {o.get("scene"): o for o in (overlay_plan or []) if isinstance(o, dict)}
    slides: List[Dict[str, Any]] = []
    errors: List[str] = []
    for i, still in enumerate(stills, 1):
        if not still or not Path(still).is_file():
            continue
        if progress:
            progress("Animating illustration %d/%d (%s) into a GIF"
                     % (i, len(stills), motion))
        card: Optional[Path] = None
        if with_cards:
            item = by_scene.get(i) or {"scene": i, "title": paragraphs[i - 1][:80],
                                       "bullets": [], "icons": []}
            card = render_card(item, out_dir / ("card_%02d" % i), style=card_style)
        try:
            gif = animate_still(still, out_dir / ("slide_%02d" % i), overlay=card,
                                size=ASPECT_SIZES.get(aspect, ASPECT_SIZES["16:9"]),
                                motion=motion)
            slides.append({"scene": i, "gif": str(gif), "still": still})
        except Exception as exc:
            errors.append("Animation of scene %d failed: %s" % (i, str(exc)[:200]))
        if progress:
            progress("Scene %d/%d animated" % (i, len(stills)))
    return slides, errors


def build_slide_script(slides: List[Dict[str, Any]], narration: str, title: str = "") -> str:
    """Narration + animated GIFs -> Markdown script (the text stays exactly as written)."""
    paragraphs = _paragraphs(narration)
    if not paragraphs:
        raise ValueError("Write the narration first (step 1).")
    lines: List[str] = []
    if title.strip():
        lines += ["# " + title.strip(), ""]
    gifs = {s["scene"]: s["gif"] for s in slides}
    for i, paragraph in enumerate(paragraphs, 1):
        if gifs.get(i):
            lines += ["![](%s)" % gifs[i], ""]
        lines += [paragraph, ""]
    return "\n".join(lines).strip() + "\n"
