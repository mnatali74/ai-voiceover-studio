"""Offline tests for the Gradio web studio (skipped when gradio is not installed)."""
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


def test_save_broll_empty(tmp_path):
    session = new_session(tmp_path)
    assert save_broll(None, session) is None
    assert save_broll([], session) is None
