"""Offline tests for the AI video builder (OpenRouter + keyless assets, all mocked)."""
import base64

import pytest

from voiceover_studio.ai_video import OpenRouterClient, Scene, build_script, find_asset, \
    generate_asset, plan_scenes, verify_asset
from voiceover_studio.errors import ConfigError, ProviderError

PLAN_JSON = '[{"narration": "Pyramids rise over the desert.", "query": "egypt pyramids"},' \
            '{"narration": "Inside, workers move giant blocks.", "query": "ancient workers"}]'


def client_with(fake, extra_routes=()):
    fake.add("POST", "https://openrouter.ai/api/v1/chat/completions",
             (200, {"choices": [{"message": {"content": PLAN_JSON}}]}))
    for route in extra_routes:
        fake.add(*route)
    return OpenRouterClient(api_key="test-key", transport=fake)


def chat_payload(content):
    return (200, {"choices": [{"message": {"content": content}}]})


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        OpenRouterClient(transport=lambda *a: None)


def test_plan_scenes_parses_model_json(fake):
    scenes = plan_scenes(client_with(fake), "A short story about pyramids", duration=30.0)
    assert len(scenes) == 2
    assert scenes[0].narration.startswith("Pyramids")
    assert scenes[1].query == "ancient workers"
    assert [s.index for s in scenes] == [0, 1]


def test_plan_scenes_requires_prompt(fake):
    with pytest.raises(ValueError):
        plan_scenes(client_with(fake), "   ")


def test_find_asset_uses_openverse_then_downloads(fake, tmp_path):
    listing = (200, {"results": [{"url": "https://images.example.org/photo.jpg"}]})
    fake.add("GET", "https://commons.wikimedia.org/w/api.php", (200, {}))
    fake.add("GET", "https://api.openverse.org/v1/images/", listing)
    fake.add("GET", "https://images.example.org/photo.jpg", (200, b"jpeg-bytes"))
    path, source = find_asset("pyramids", "landscape", tmp_path, 0, transport=fake)
    assert source == "openverse image"
    assert path.read_bytes() == b"jpeg-bytes"


def test_find_asset_falls_back_to_generation(fake, tmp_path):
    fake.add("GET", "https://commons.wikimedia.org/w/api.php", (200, {}))
    fake.add("GET", "https://api.openverse.org/v1/images/", (200, {"results": []}))
    fake.add("GET", "https://image.pollinations.ai/prompt/", (200, b"generated-jpeg"))
    path, source = find_asset("desert dunes", "portrait", tmp_path, 3, transport=fake)
    assert source == "generated (pollinations)"
    assert path.read_bytes() == b"generated-jpeg"


def test_generate_asset_direct(fake, tmp_path):
    fake.add("GET", "https://image.pollinations.ai/prompt/", (200, b"gen"))
    path, source = generate_asset("space nebula", "landscape", tmp_path, 1, transport=fake)
    assert source == "generated (pollinations)" and path.is_file()


def test_find_asset_raises_when_everything_fails(fake, tmp_path):
    fake.add("GET", "https://commons.wikimedia.org/w/api.php", (200, {}))
    fake.add("GET", "https://api.openverse.org/v1/images/", (200, {"results": []}))
    fake.add("GET", "https://image.pollinations.ai/prompt/", (500, {"error": "down"}))
    with pytest.raises(ProviderError):
        find_asset("ocean", "landscape", tmp_path, 2, transport=fake)


def test_verify_asset_parses_verdict(fake, tmp_path):
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"\xff\xd8fakejpeg")
    scene = Scene(0, "text", "egypt pyramids", asset_path=str(image))
    client = OpenRouterClient(api_key="k", transport=fake)
    fake.add("POST", "https://openrouter.ai/api/v1/chat/completions",
             chat_payload('{"match": false, "reason": "shows a city"}'))
    match, reason = verify_asset(client, scene, "google/gemini-2.0-flash-001")
    assert match is False and reason == "shows a city"
    sent = fake.calls[-1]["body"].decode()
    assert base64.b64encode(b"\xff\xd8fakejpeg").decode() in sent


def test_verify_asset_skips_video_without_ffmpeg(fake, tmp_path, monkeypatch):
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    video = tmp_path / "scene.mp4"
    video.write_bytes(b"not-a-real-video")
    scene = Scene(0, "text", "query", asset_path=str(video))
    match, reason = verify_asset(OpenRouterClient(api_key="k", transport=fake), scene, "m")
    assert match is True and "ffmpeg" in reason


def test_build_script_includes_assets_and_narration():
    scenes = [Scene(0, "First line.", "q1", asset_path="assets/scene_00.jpg"),
              Scene(1, "Second line.", "q2")]
    text = build_script(scenes, title="Pyramids")
    assert text.startswith("# Pyramids")
    assert "![](assets/scene_00.jpg)" in text
    assert "First line." in text and "Second line." in text
