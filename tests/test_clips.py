"""Offline tests for the clip studio (yt-dlp and ffmpeg mocked, OpenRouter faked)."""
import base64
import json

import pytest

from voiceover_studio.clips import (build_library, build_video_script, catalog_path,
                                    classify_clip, cut_clip, extract_frames, find_sources,
                                    load_catalog, parse_vtt, probe_duration, save_catalog,
                                    search_youtube, select_clips, suggest_searches,
                                    suggest_segments, title_avoids_people,
                                    _transcript_excerpt)
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
    assert "-an" in calls[0]
    cut_clip(src, 5.0, 12.5, tmp_path / "out" / "audio.mp4", audio=True)
    assert "-an" not in calls[1]


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
                        lambda url, dest, cookies=None: (source, {"id": "abc",
                                                                  "title": "Vineyards",
                                                                  "duration": 120.0}))
    monkeypatch.setattr(clips, "fetch_transcript", lambda url, dest, cookies=None: "[00:10] great shots")
    monkeypatch.setattr(clips, "probe_duration", lambda path: 120.0)
    cut, frames = [], []
    monkeypatch.setattr(clips, "cut_clip",
                        lambda src, start, end, dest, audio=False:
                        (cut.append((start, end)), dest)[1])
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
    assert library["clips"] == [] and library["errors"] == []


def test_build_library_continues_after_download_failure(fake, monkeypatch, tmp_path):
    from voiceover_studio import clips
    source = tmp_path / "sources" / "ok.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")

    def fake_download(url, dest, cookies=None):
        if "blocked" in url:
            raise ConfigError("Sign in to confirm you're not a bot")
        return source, {"id": "ok", "title": "Good video", "duration": 60.0}

    monkeypatch.setattr(clips, "download_video", fake_download)
    monkeypatch.setattr(clips, "fetch_transcript", lambda url, dest, cookies=None: "")
    monkeypatch.setattr(clips, "cut_clip", lambda src, start, end, dest, audio=False: dest)
    monkeypatch.setattr(clips, "extract_frames", lambda src, dest, every=3.0: [])
    monkeypatch.setattr(clips, "classify_clip",
                        lambda client, model, path, title, context: {"description": "x", "tags": []})
    library = build_library(["https://youtu.be/blocked", "https://youtu.be/ok"],
                            tmp_path, None, "m", n_clips=1, clip_seconds=10.0, stills=False,
                            delay=0.0)
    assert len(library["clips"]) == 1 and library["sources"] == ["Good video"]
    assert "bot check" in library["errors"][0] and "cookies.txt" in library["errors"][0]


def test_title_avoids_people_filters_subjects():
    assert title_avoids_people("Aerial drone view of Tuscan vineyards at sunset")
    assert not title_avoids_people("My VLOG: wine tasting in Tuscany")
    assert not title_avoids_people("Wine TUTORIAL for beginners")


def test_suggest_searches_parses_query_list(fake):
    queries_json = '["aerial drone vineyard hills", "italian countryside sunset", "wine cellar barrels"]'
    client = chat_client(fake, queries_json)
    queries = suggest_searches(client, "m", "enoturismo nelle Marche", n=3)
    assert queries == ["aerial drone vineyard hills", "italian countryside sunset",
                       "wine cellar barrels"]


class FakeYdl:
    def __init__(self, opts):
        FakeYdl.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url, download=False):
        assert url.startswith("ytsearch") and download is False
        return {"entries": [
            {"id": "abc", "title": "Aerial drone vineyard hills", "duration": 300,
             "channel": "Scenery HD", "url": "https://www.youtube.com/watch?v=abc"},
            {"id": "bad", "title": "Wine tasting VLOG with friends", "duration": 600,
             "channel": "People TV", "url": "https://www.youtube.com/watch?v=bad"},
            {"id": "def", "title": "Italian countryside sunset", "duration": 240,
             "channel": "Nature", "url": None},
        ]}


def test_search_youtube_flat_and_filtered(monkeypatch):
    from voiceover_studio import clips
    monkeypatch.setattr(clips, "_import_yt_dlp",
                        lambda: type("M", (), {"YoutubeDL": FakeYdl}))
    results = search_youtube("vineyard aerial", limit=3)
    assert [r["id"] for r in results] == ["abc", "def"]
    assert results[1]["url"].endswith("watch?v=def")
    assert FakeYdl.opts["extract_flat"] is True


def test_find_sources_deduplicates_across_queries(fake, monkeypatch):
    from voiceover_studio import clips
    monkeypatch.setattr(clips, "search_youtube",
                        lambda query, limit=4, cookies=None: [
                            {"id": "abc", "title": "Aerial hills", "channel": "C",
                             "duration": 120.0, "url": "https://youtu.be/abc"}])
    client = chat_client(fake, '["q1", "q2", "q3"]')
    queries, candidates = find_sources(client, "m", "vineyards", sources=3)
    assert queries == ["q1", "q2", "q3"]
    assert len(candidates) == 1 and candidates[0]["query"] == "q1"


def test_transcript_excerpt_keeps_nearby_lines():
    transcript = "[00:05] opening line\n[00:30] middle line\n[02:00] far away"
    assert "opening line" in _transcript_excerpt(transcript, 8.0, 15.0)
    assert "middle line" not in _transcript_excerpt(transcript, 8.0, 15.0)
    assert "far away" in _transcript_excerpt(transcript, 115.0, 125.0)


def test_classify_clip_parses_details(fake, tmp_path, monkeypatch):
    image = tmp_path / "clip_01.mp4.frame.jpg"
    image.write_bytes(b"\xff\xd8fakejpeg")
    monkeypatch.setattr("voiceover_studio.ai_video._first_frame", lambda video, dest: True)
    answer = ('{"description": "Aerial view of green vineyard rows at sunset", '
              '"tags": ["aerial", "vineyard", "sunset"], "mood": "calm", '
              '"setting": "Italian countryside", "camera": "drone", '
              '"has_people": false, "usable": true}')
    client = chat_client(fake, answer)
    details = classify_clip(client, "m", tmp_path / "clip_01.mp4", "Vineyards", "wine talk")
    assert details["description"].startswith("Aerial view")
    assert "vineyard" in details["tags"] and details["has_people"] is False
    sent = fake.calls[-1]["body"].decode()
    assert base64.b64encode(b"\xff\xd8fakejpeg").decode() in sent


def test_classify_clip_falls_back_without_vision(fake, tmp_path, monkeypatch):
    from voiceover_studio.providers.miragea import HttpResponse
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    monkeypatch.setattr("voiceover_studio.ai_video._first_frame", lambda video, dest: False)

    def flaky(method, url, headers, body, timeout):
        payload = json.dumps({"choices": [{"message": {"content":
            '{"description": "barrels", "tags": ["cellar"], "usable": true}'}}]}).encode()
        if b"image_url" in (body or b""):
            return HttpResponse(404, {}, b'{"error": "no vision"}')
        return HttpResponse(200, {}, payload)

    from voiceover_studio.ai_video import OpenRouterClient
    client = OpenRouterClient(api_key="k", transport=flaky)
    details = classify_clip(client, "m", media, "Cellar video")
    assert details["description"] == "barrels" and details["tags"] == ["cellar"]


def test_catalog_roundtrip(tmp_path):
    entries = [{"id": "a_01", "description": "hills", "tags": ["hills"]}]
    save_catalog(tmp_path, entries)
    assert catalog_path(tmp_path).is_file()
    assert load_catalog(tmp_path) == entries
    assert load_catalog(tmp_path / "missing") == []


def test_select_clips_maps_ids_and_ignores_unknown(fake):
    catalog = [{"id": "a_01", "description": "green hills aerial", "tags": ["hills"],
                "camera": "drone", "has_people": False, "clip_path": "/tmp/a_01.mp4"},
               {"id": "b_01", "description": "wine barrels", "tags": ["cellar"],
                "camera": "static", "has_people": False, "clip_path": "/tmp/b_01.mp4"}]
    selection_json = '[{"id": "b_01", "narration": "Nelle cantine..."}, ' \
                     '{"id": "ghost_99", "narration": "ignored"}, ' \
                     '{"id": "a_01", "narration": "Colline a perdita d occhio."}]'
    client = chat_client(fake, selection_json)
    selected = select_clips(client, "m", "enoturismo", catalog, language="italian")
    assert [item["entry"]["id"] for item in selected] == ["b_01", "a_01"]
    assert selected[0]["narration"] == "Nelle cantine..."


def test_build_video_script_pairs_clips_and_narration():
    selected = [{"entry": {"id": "a_01", "clip_path": "/tmp/a_01.mp4"},
                 "narration": "Colline a perdita d occhio."},
                {"entry": {"id": "b_01", "clip_path": ""},
                 "narration": "Nelle cantine il tempo si ferma."}]
    text = build_video_script(selected, title="Enoturismo")
    assert text.startswith("# Enoturismo")
    assert "![](/tmp/a_01.mp4)" in text and "Colline" in text
    assert "b_01" not in text


def test_build_library_classifies_and_saves_catalog(fake, monkeypatch, tmp_path):
    from voiceover_studio import clips
    source = tmp_path / "sources" / "abc.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    monkeypatch.setattr(clips, "download_video",
                        lambda url, dest, cookies=None: (source, {"id": "abc",
                                                                  "title": "Vineyards",
                                                                  "duration": 120.0}))
    monkeypatch.setattr(clips, "fetch_transcript", lambda url, dest, cookies=None: "[00:05] rows of vines")
    monkeypatch.setattr(clips, "cut_clip", lambda src, start, end, dest, audio=False: dest)
    monkeypatch.setattr(clips, "extract_frames", lambda src, dest, every=3.0: [])
    monkeypatch.setattr(clips, "classify_clip",
                        lambda client, model, path, title, context:
                        {"description": "Green hills", "tags": ["hills"], "mood": "calm",
                         "setting": "Tuscany", "camera": "drone", "has_people": False,
                         "usable": True})
    client = chat_client(fake, '[{"start": 5, "end": 15, "title": "Hills", "reason": "r"}, '
                               '{"start": 30, "end": 40, "title": "Rows", "reason": "r"}]')
    sleeps = []
    library = build_library(["https://youtu.be/abc"], tmp_path, client, "m",
                            n_clips=None, clip_seconds=10.0, stills=True, delay=7.0,
                            sleep=sleeps.append)
    assert len(library["catalog"]) == 2
    assert library["catalog"][0]["id"] == "vineyards_01"
    assert library["catalog"][0]["description"] == "Green hills"
    assert load_catalog(tmp_path)[1]["segment_title"] == "Rows"
    assert sleeps == [3.0]  # pause between download and subtitle fetch, none between sources


def test_cookie_opts_from_argument_and_env(monkeypatch):
    from voiceover_studio.clips import _cookie_opts
    monkeypatch.delenv("VOICEOVER_YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("VOICEOVER_YTDLP_COOKIES_FROM_BROWSER", raising=False)
    assert _cookie_opts() == {}
    assert _cookie_opts("/tmp/cookies.txt") == {"cookiefile": "/tmp/cookies.txt"}
    monkeypatch.setenv("VOICEOVER_YTDLP_COOKIES", "/env/cookies.txt")
    assert _cookie_opts() == {"cookiefile": "/env/cookies.txt"}
    monkeypatch.delenv("VOICEOVER_YTDLP_COOKIES")
    monkeypatch.setenv("VOICEOVER_YTDLP_COOKIES_FROM_BROWSER", "chrome:Profile 1")
    assert _cookie_opts() == {"cookiesfrombrowser": ("chrome",)}
