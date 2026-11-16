"""Offline tests for the Gradio web studio (skipped when gradio is not installed)."""
import time
from pathlib import Path

import pytest

pytest.importorskip("gradio")

from voiceover_studio.webui import _sample_sentence, build_ui, example_scripts, generate_voiceover, \
    list_voices, new_session, pick_voice, save_broll, say_line  # noqa: E402
from voiceover_studio.webui import test_voice as preview_voice_line  # noqa: E402


class _Evt:
    def __init__(self, index):
        self.index = index


def test_build_ui(tmp_path):
    app = build_ui(workdir=tmp_path)
    assert app is not None


def test_new_session(tmp_path):
    session = new_session(tmp_path)
    assert session["dir"].is_dir()


def test_generate_voiceover_with_dummy(tmp_path):
    session = new_session(tmp_path)
    status, audio, srt, rows = generate_voiceover(
        "# Demo\n\nHello world, this is the studio.\n\n[NARRATOR] Second segment.",
        "dummy", "auto", 1.0, 0.35, "voiceover.wav", session)
    assert "2" in status
    assert audio and audio.endswith(".wav")
    assert srt and srt.endswith(".srt")
    assert len(rows) == 2
    assert rows[0][1] == "DEFAULT" or rows[1][1] == "NARRATOR"


def test_generate_voiceover_empty_script(tmp_path):
    session = new_session(tmp_path)
    with pytest.raises(ValueError):
        generate_voiceover("   ", "dummy", "auto", 1.0, 0.35, "voiceover.wav", session)


def test_say_line_with_dummy(tmp_path):
    session = new_session(tmp_path)
    status, audio = say_line("One line only.", "dummy", "", "auto", 1.0, session)
    assert audio and Path(audio).is_file()


def test_list_voices_with_dummy(tmp_path):
    status, rows = list_voices("dummy", "")
    assert "dummy" in status
    assert isinstance(rows, list)


def test_example_scripts_found():
    scripts = example_scripts()
    assert any(s.endswith("script.md") for s in scripts)


def test_test_voice_with_dummy(tmp_path):
    session = new_session(tmp_path)
    status, audio = preview_voice_line("dummy-voice", "dummy", session)
    assert audio and Path(audio).is_file()


def test_test_voice_without_selection(tmp_path):
    session = new_session(tmp_path)
    with pytest.raises(ValueError):
        preview_voice_line("", "dummy", session)


def test_sample_sentence_matches_voice_language():
    assert "prova" in _sample_sentence("it-IT-GiuseppeNeural")
    assert _sample_sentence("xx-XX-Unknown") == _sample_sentence("en-US-AvaNeural")


def test_pick_voice_uses_clicked_row():
    rows = [["it-IT-GiuseppeNeural", "it", "Giuseppe"], ["en-US-AvaNeural", "en-US", "Ava"]]
    assert pick_voice(_Evt([1, 0]), rows) == "en-US-AvaNeural"
    assert pick_voice(_Evt(None), rows) == ""


def test_save_broll_copies_in_upload_order(tmp_path):
    session = new_session(tmp_path)
    a = tmp_path / "moon.jpg"
    a.write_bytes(b"x")
    b = tmp_path / "stars.jpg"
    b.write_bytes(b"y")
    folder = save_broll([str(a), str(b)], session)
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["001_moon.jpg", "002_stars.jpg"]


def test_new_session_is_absolute(tmp_path):
    session = new_session(tmp_path / "rel")
    assert Path(session["dir"]).is_absolute()


def test_ai_regenerate_replaces_asset(tmp_path, monkeypatch):
    import voiceover_studio.ai_video as ai_video
    from voiceover_studio.webui import ai_plan, ai_regenerate

    def fake_plan(client, prompt, workdir, **kwargs):
        scene = ai_video.Scene(0, "Line one.", "pyramids")
        scene.asset_path, scene.source = str(workdir / "scene_00.jpg"), "openverse image"
        return [scene]

    def fake_generate(query, orientation, workdir, index, transport=None):
        workdir.mkdir(parents=True, exist_ok=True)
        target = workdir / "scene_00.jpg"
        target.write_bytes(b"new")
        return target, "generated (pollinations)"

    monkeypatch.setattr(ai_video, "plan_and_fetch", fake_plan)
    monkeypatch.setattr(ai_video, "generate_asset", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    session = new_session(tmp_path)
    ai_plan("pyramids", 30.0, "auto", "landscape", "m", False, "m", session)
    status, gallery = ai_regenerate("1", "landscape", session)
    assert "regenerated" in status
    assert gallery[0][0].endswith("scene_00.jpg")


def test_save_broll_empty(tmp_path):
    session = new_session(tmp_path)
    assert save_broll(None, session) is None
    assert save_broll([], session) is None


def test_save_broll_includes_library_picks(tmp_path):
    session = new_session(tmp_path)
    asset = tmp_path / "clip_01.mp4"
    asset.write_bytes(b"video")
    folder = save_broll(None, session, extra=[str(asset)])
    assert folder is not None and list(folder.glob("*clip_01.mp4"))


def test_clip_search_populates_candidates(tmp_path, monkeypatch):
    from voiceover_studio.webui import clip_search
    import voiceover_studio.clips as clips

    def fake_search(query, limit=4, cookies=None):
        return [{"id": "abc", "title": "Aerial vineyard hills", "channel": "Scenery",
                 "duration": 305.0, "url": "https://youtu.be/abc"},
                {"id": "def", "title": "Wine cellar barrels closeup", "channel": "Nature",
                 "duration": 95.0, "url": "https://youtu.be/def"}]

    monkeypatch.setattr(clips, "search_youtube", fake_search)
    monkeypatch.setattr(clips, "suggest_searches",
                        lambda client, model, prompt, n=5:
                        ["aerial vineyards", "wine cellar barrels"])
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    session = new_session(tmp_path)
    status, rows, previews, update = clip_search("enoturismo Marche", "some/model", 5, "", "",
                                                 ["YouTube"], "", session)
    assert "**2** searches, **2** candidate" in status
    assert rows[0][1] == "Aerial vineyard hills"
    assert "youtube.com/embed/abc" in previews and "youtube.com/embed/def" in previews
    assert update["choices"][0].startswith("Aerial vineyard hills")
    assert session["clip_labels"][update["choices"][0]] == "https://youtu.be/abc"


def test_clip_search_uses_manual_queries_without_llm(tmp_path, monkeypatch):
    from voiceover_studio.webui import clip_search
    import voiceover_studio.clips as clips

    def fake_search(query, limit=4, cookies=None):
        return [{"id": query[:3], "title": "Result of %s" % query, "channel": "C",
                 "duration": 120.0, "url": "https://youtu.be/" + query[:3]}]

    monkeypatch.setattr(clips, "search_youtube", fake_search)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    session = new_session(tmp_path)
    status, rows, previews, update = clip_search("enoturismo Marche", "best value (auto)", 5,
                                                 "aerial vineyard hills\n\nwine cellar barrels",
                                                 "", ["YouTube"], "", session)
    assert "**2** searches, **2** candidate" in status
    assert rows[0][3] == "aerial vineyard hills" and rows[1][3] == "wine cellar barrels"
    assert session["clip_labels"][update["choices"][0]].endswith("aer")


def test_clip_search_merges_commons_candidates(tmp_path, monkeypatch):
    from voiceover_studio.webui import clip_search
    import voiceover_studio.clips as clips

    def fake_commons(query, limit=4, transport=None):
        return [{"id": "commons-hills", "title": "Vineyard hills",
                 "channel": "Wikimedia Commons", "duration": 42.0,
                 "url": "https://upload.wikimedia.org/hills.webm", "source": "commons"}]

    monkeypatch.setattr(clips, "search_commons", fake_commons)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    session = new_session(tmp_path)
    status, rows, previews, update = clip_search("enoturismo", "", 5, "vineyard hills", "",
                                                 ["Wikimedia Commons"], "", session)
    assert rows[0][2] == "Wikimedia Commons"
    assert "<video" in previews and "hills.webm" in previews
    assert session["clip_titles"]["https://upload.wikimedia.org/hills.webm"] == "Vineyard hills"
    assert session["clip_labels"][update["choices"][0]] == "https://upload.wikimedia.org/hills.webm"


def test_clip_suggest_fills_editable_queries(tmp_path, monkeypatch):
    from voiceover_studio.webui import clip_suggest
    import voiceover_studio.clips as clips

    monkeypatch.setattr(clips, "suggest_searches",
                        lambda client, model, prompt, n=5:
                        ["q%d" % i for i in range(1, n + 1)])
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    session = new_session(tmp_path)
    status, text = clip_suggest("enoturismo Marche", "some/model", 3, session)
    assert "**3** queries" in status
    assert text == "q1\nq2\nq3"


def test_clip_build_uses_selected_sources(tmp_path, monkeypatch):
    import voiceover_studio.clips as clips
    from voiceover_studio.webui import clip_build

    def fake_build(urls, workdir, client, model, **kwargs):
        assert urls == ["https://youtu.be/abc", "https://youtu.be/xyz"]
        assert kwargs.get("keep_audio", False) is False
        assert kwargs["n_clips"] is None and kwargs["delay"] == 5.0
        assert kwargs["target_clips"] == 9
        kwargs["progress"]("Downloading https://youtu.be/abc")
        time.sleep(1.5)
        catalog = [{"id": "c_01", "source_title": "Vineyards",
                    "clip_path": str(workdir / "clip_library" / "clips" / "c_01.mp4"),
                    "description": "Green hills", "tags": ["hills"]}]
        return {"clips": [str(workdir / "clip_library" / "clips" / "c_01.mp4")],
                "frames": [str(workdir / "clip_library" / "frames" / "f_001.jpg")],
                "segments": [{"source": "Vineyards", "title": "Drone shot", "start": 5.0,
                              "end": 15.0, "reason": "b-roll"}],
                "sources": ["Vineyards"], "errors": [], "catalog": catalog}

    monkeypatch.setattr(clips, "build_library", fake_build)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    session = new_session(tmp_path)
    session["clip_labels"] = {"Aerial hills — C (5:00)": "https://youtu.be/abc"}
    yields = list(clip_build(
        ["Aerial hills — C (5:00)"], "https://youtu.be/xyz", "some/model", True, 2,
        10, True, 5.0, 60, session))
    assert len(yields) >= 2
    assert "Downloading https://youtu.be/abc" in yields[-2][0]
    status, clip_files, frames, catalog_rows, update = yields[-1]
    assert "Target: a **60 s** video → **6** shots" in status
    assert "audio-free clips and **1** frames" in status and "Catalog: **1** entries" in status
    assert "Drone shot" in status
    assert clip_files[0].endswith("c_01.mp4")
    assert catalog_rows[0][0] == "c_01"
    assert update["choices"][0].endswith("c_01.mp4")
    assert session["library"]["clips"]


def test_clip_select_builds_script(tmp_path, monkeypatch):
    import voiceover_studio.clips as clips
    from voiceover_studio.webui import clip_select

    def fake_select(client, model, prompt, catalog, language="the user's language",
                    max_shots=8):
        assert language == "italian" and max_shots == 6
        return [{"entry": catalog[0], "narration": "Colline infinite."}]

    monkeypatch.setattr(clips, "select_clips", fake_select)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    session = new_session(tmp_path)
    clips.save_catalog(session["dir"], [{"id": "c_01", "description": "hills",
                                         "tags": ["hills"], "clip_path": "/tmp/c_01.mp4"}])
    status, script, rows = clip_select("enoturismo", "some/model", "italian", 60, 10, session)
    assert "Selected **1** clips" in status
    assert "![](/tmp/c_01.mp4)" in script and "Colline infinite." in script
    assert rows[0][2] == "Colline infinite."
    assert session["selected_clips"] == ["c_01"]


def test_voice_choices_never_crashes(tmp_path):
    from voiceover_studio.webui import voice_choices
    update = voice_choices("dummy")
    assert "choices" in update
