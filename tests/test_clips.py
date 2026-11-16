"""Offline tests for the clip studio (yt-dlp and ffmpeg mocked, OpenRouter faked)."""
import json

import pytest

from voiceover_studio.clips import (build_library, cut_clip, extract_frames, parse_vtt,
                                    probe_duration, suggest_segments)
from voiceover_studio.errors import ConfigError
from voiceover_studio.providers.miragea import HttpResponse

VTT = """WEBVTT

00:00:01.000 --> 00:00:04.000
<v Roger>Hello world

00:00:04.000 --> 00:00:08.000
Hello world

00:00:08.000 --> 00:00:12.000
<v Roger>Second <b>sentence</b> here
"""

SEGMENTS_JSON = ('[{"start": 5, "end": 15, "title": "Opening drone shot", '
                 '"reason": "establishing b-roll"}, '
                 '{"start": 9999, "end": 10005, "title": "Outro", "reason": "closing"}]')


def chat_client(fake, content):
    fake.add("POST", "https://openrouter.ai/api/v1/chat/completions",
             (200, {"choices": [{"message": {"content": content}}]}))
    from voiceover_studio.ai_video import OpenRouterClient
    return OpenRouterClient(api_key="test-key", transport=fake)


def test_parse_vtt_merges_and_cleans():
    text = parse_vtt(VTT)
    assert "[00:01] Hello world" in text
    assert "[00:08] Second sentence here" in text
    assert "<v Roger>" not in text and "<b>" not in text
    assert text.count("Hello world") == 1


def test_suggest_segments_clamps_to_duration(fake):
    client = chat_client(fake, SEGMENTS_JSON)
    segments = suggest_segments(client, "m", "text", duration=60.0, n_clips=2,
                                clip_seconds=10.0, title="Test video")
    assert segments[0]["start"] == 5.0 and segments[0]["end"] == 15.0
    assert segments[1]["end"] == 60.0 and segments[1]["start"] < segments[1]["end"]
    assert segments[1]["title"] == "Outro"


def test_suggest_segments_falls_back_without_transcript():
    segments = suggest_segments(None, "m", "", duration=90.0, n_clips=3, clip_seconds=10.0)
    assert len(segments) == 3
    assert all(seg["end"] > seg["start"] and seg["end"] <= 90.0 for seg in segments)


def test_cut_clip_uses_stream_copy(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("voiceover_studio.audio.ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr("voiceover_studio.audio._ffmpeg", lambda args: calls.append(args))
    src = tmp_path / "src.mp4"
    src.write_bytes(b"video")
    dest = cut_clip(src, 5.0, 12.5, tmp_path / "out" / "clip.mp4")
    assert dest.parent.name == "out"
    assert calls[0][:2] == ["-ss", "5.00"]
    assert "-t" in calls[0] and "7.50" in calls[0]
    assert "-c" in calls[0] and "copy" in calls[0]


def test_cut_clip_requires_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr("voiceover_studio.audio.ffmpeg_path", lambda: None)
    with pytest.raises(ConfigError):
        cut_clip(tmp_path / "src.mp4", 0, 5, tmp_path / "c.mp4")


def test_extract_frames_writes_pattern(monkeypatch, tmp_path):
    calls = []

    def fake_ffmpeg(args):
        calls.append(args)
        for i in (1, 2):
            (tmp_path / "clip_frames" / ("frame_%03d.jpg" % i)).write_bytes(b"jpg")

    monkeypatch.setattr("voiceover_studio.audio.ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr("voiceover_studio.audio._ffmpeg", fake_ffmpeg)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"video")
    frames = extract_frames(src, tmp_path / "clip_frames", every=3.0)
    assert "fps=1/3.0" in calls[0][3]
    assert len(frames) == 2 and all(f.is_file() for f in frames)


def test_probe_duration_reads_ffprobe(monkeypatch, tmp_path):
    import subprocess
    media = tmp_path / "src.mp4"
    media.write_bytes(b"v")

    class Proc:
        stdout = b"42.5\n"
        stderr = b""

    monkeypatch.setattr("voiceover_studio.audio.ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Proc())
    assert probe_duration(media) == 42.5


def test_build_library_cuts_suggested_segments(fake, monkeypatch, tmp_path):
    from voiceover_studio import clips
    source = tmp_path / "sources" / "abc.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    monkeypatch.setattr(clips, "download_video",
                        lambda url, dest: (source, {"id": "abc", "title": "Vineyards",
                                                    "duration": 120.0}))
    monkeypatch.setattr(clips, "fetch_transcript", lambda url, dest: "[00:10] great shots")
    monkeypatch.setattr(clips, "probe_duration", lambda path: 120.0)
    cut, frames = [], []
    monkeypatch.setattr(clips, "cut_clip",
                        lambda src, start, end, dest: (cut.append((start, end)), dest)[1])
    monkeypatch.setattr(clips, "extract_frames",
                        lambda src, dest, every=3.0: frames.append(dest)
                        or [dest / "frame_001.jpg", dest / "frame_002.jpg"])
    client = chat_client(fake, SEGMENTS_JSON)
    library = build_library(["https://youtu.be/abc"], tmp_path, client, "m",
                            n_clips=2, clip_seconds=10.0, stills=True)
    assert len(library["clips"]) == 2 and library["clips"][0].endswith("vineyards_01.mp4")
    assert cut[0] == (5.0, 15.0) and cut[1][1] == 120.0
    assert library["segments"][0]["source"] == "Vineyards"
    assert len(library["frames"]) == 4 and "frame_001.jpg" in library["frames"][0]


def test_build_library_skips_bad_urls(fake, monkeypatch, tmp_path):
    from voiceover_studio import clips
    monkeypatch.setattr(clips, "download_video", lambda *a: (_ for _ in ()).throw(AssertionError()))
    library = build_library(["", "not-a-url"], tmp_path, None, "m")
    assert library == {"clips": [], "frames": [], "segments": [], "sources": []}
