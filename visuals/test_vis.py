"""Tests for the parts that don't need a GPU or MIDI hardware: python3 -m unittest test_vis"""
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vis_core import (CLOCK, CONTINUE, PPQN, SPP, START, STOP, MidiState,
                      beat_pulse, decay, load_show, Hit,
                      dive_zoom, julia_c, MANDELBROT_SPOTS,
                      layer_matrix, one_in, pick_twist, TWISTS, roll_pulse)
from vis_media import ArraySource, ClipPool, ClipSource, ImageSource, cv2, is_colorless
import vis_color as vc
import clip_grab as cg
import vis_text as vt
import vis_shuffle as vs
from vis_core import generated_item, GENERATED_KINDS


class TestMidiState(unittest.TestCase):
    def setUp(self):
        self.s = MidiState(visual_ch=16, drum_ch=10)

    def test_bpm_from_clock(self):
        tick = 60 / (128 * PPQN)
        for i in range(48):
            self.s.feed([CLOCK], t=i * tick)
        self.assertAlmostEqual(self.s.bpm, 128, places=3)

    def test_ticks_only_count_while_running(self):
        for i in range(10):
            self.s.feed([CLOCK], t=i * 0.02)
        self.assertEqual(self.s.ticks, 0)
        self.s.feed([START], t=1)
        for i in range(24):
            self.s.feed([CLOCK], t=1 + i * 0.02)
        self.assertEqual(self.s.ticks, 24)
        self.s.feed([STOP], t=2)
        self.s.feed([CLOCK], t=2.02)
        self.assertEqual(self.s.ticks, 24)
        self.s.feed([CONTINUE], t=3)
        self.assertTrue(self.s.running)

    def test_song_position_pointer(self):
        self.s.feed([SPP, 0x10, 0x01])  # 16 + 128 = 144 sixteenths
        self.assertEqual(self.s.ticks, 144 * 6)
        self.assertIn(("locate", 864), self.s.drain_events())

    def test_visual_channel_events(self):
        self.s.feed([0x9F, 36, 127])
        self.s.feed([0x9F, 36, 0])        # note on vel 0 = off
        self.s.feed([0x8F, 37, 0])
        self.s.feed([0xBF, 2, 127])
        self.s.feed([0xCF, 3])
        self.s.feed([0x90, 36, 100])      # channel 1: ignored
        ev = self.s.drain_events()
        self.assertEqual(ev, [("note_on", 36, 1.0), ("note_off", 36), ("note_off", 37), ("program", 3)])
        self.assertEqual(self.s.cc_value(2, 0.0), 1.0)
        self.assertEqual(self.s.cc_value(9, 0.25), 0.25)
        self.assertEqual(self.s.drain_events(), [])

    def test_drum_hits(self):
        self.s.feed([0x99, 36, 127], t=5.0)
        self.s.feed([0x99, 38, 64], t=5.5)
        self.s.feed([0x99, 42, 32], t=6.0)
        self.s.feed([0x99, 50, 127], t=7.0)  # unmapped note
        kick, snare, hat = self.s.hits()
        self.assertEqual((kick.t, kick.vel), (5.0, 1.0))
        self.assertEqual(snare.t, 5.5)
        self.assertEqual(hat.t, 6.0)
        self.assertEqual(self.s.drain_events(), [])  # drums don't trigger scenes

    def test_beats_interpolate_and_offset(self):
        tick = 60 / (120 * PPQN)
        self.s.feed([START], t=0)
        for i in range(48):
            self.s.feed([CLOCK], t=i * tick)
        last = 47 * tick
        self.assertAlmostEqual(self.s.beats(last), 2.0, places=6)
        self.assertAlmostEqual(self.s.beats(last + tick / 2), 2 + 0.5 / PPQN, places=6)
        self.assertAlmostEqual(self.s.beats(last + 10), 2 + 1 / PPQN, places=6)  # capped at one tick
        self.assertAlmostEqual(self.s.beats(last, offset_s=0.25), 2.5, places=6)  # 0.25 s at 120 bpm

    def test_idle_and_poke(self):
        self.assertIsNone(self.s.idle_for(10))
        self.s.poke("kick", 1.0, t=1)
        self.assertIsNone(self.s.idle_for(10))  # keyboard hits aren't MIDI activity
        self.s.feed([CLOCK], t=4)
        self.assertEqual(self.s.idle_for(10), 6)

    def test_garbage_is_ignored(self):
        for msg in ([], [0x9F], [0xF0, 1, 2, 0xF7], [SPP, 1]):
            self.s.feed(msg)


class TestEnvelopes(unittest.TestCase):
    def test_decay(self):
        h = Hit(1.0, 0.5)
        self.assertEqual(decay(0.5, h, 0.1), 0.0)
        self.assertAlmostEqual(decay(1.0, h, 0.1), 0.5)
        self.assertLess(decay(1.5, h, 0.1), 0.01)

    def test_beat_pulse(self):
        self.assertAlmostEqual(beat_pulse(3.0), 1.0)
        self.assertGreater(beat_pulse(3.1), beat_pulse(3.5))

    def test_flash_is_gone(self):
        import vis_core
        self.assertNotIn("flash", vis_core.DEFAULT_CC)
        self.assertNotIn("flash", vis_core.DEFAULT_CC_VALUES)
        with tempfile.TemporaryDirectory() as d:                # an older show file still asking for it
            (Path(d) / "s.toml").write_text("[output]\nmax_flash_hz = 3\n[cc]\nflash = 6\n[cc_defaults]\nflash = 0.25\n")
            show = load_show(Path(d) / "s.toml")
        self.assertNotIn("flash", show.cc_map)
        self.assertNotIn("flash", show.cc_defaults)
        self.assertNotIn("max_flash_hz", show.output)
        self.assertIn("flash has been removed", " ".join(show.warnings))


class TestFractals(unittest.TestCase):
    def test_dive_goes_in_and_back_out(self):
        self.assertAlmostEqual(dive_zoom(0, 16, 1000), 1.0)
        self.assertAlmostEqual(dive_zoom(64, 16, 1000), 1000.0)   # full depth after 16 bars
        self.assertAlmostEqual(dive_zoom(128, 16, 1000), 1.0)     # back at the top after 32
        self.assertAlmostEqual(dive_zoom(32, 16, 1000), dive_zoom(96, 16, 1000))  # mirror image
        zs = [dive_zoom(b, 16, 1000) for b in range(0, 65, 4)]
        self.assertEqual(zs, sorted(zs))                          # only ever deeper on the way in

    def test_julia_walks_a_circle(self):
        x, y = julia_c(0, 32, 0.7885, start=0.5)
        self.assertAlmostEqual(x, -0.7885)
        self.assertAlmostEqual(y, 0.0)
        x2, y2 = julia_c(128, 32, 0.7885, start=0.5)              # one full trip later: same shape
        self.assertAlmostEqual(x2, x)
        self.assertAlmostEqual(y2, y, places=9)
        self.assertAlmostEqual(math.hypot(*julia_c(37, 32, 0.7885)), 0.7885)

    def test_fractal_items_load(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "s.toml").write_text("""
[[song]]
  [[song.item]]
  note = 40
  type = "mandelbrot"
  spot = "spiral"
  max_zoom = 1e9
  [[song.item]]
  note = 41
  type = "julia"
  palette = "neon"
  iterations = 5
  [[song.item]]
  note = 42
  type = "mandelbrot"
  center = [0.25, 0.0]
""")
            show = load_show(Path(d) / "s.toml")
        items = show.songs[0].items
        m, j = items[40], items[41]
        self.assertIsNone(m.path)
        self.assertEqual(m.params["center"], MANDELBROT_SPOTS["spiral"])
        self.assertEqual(m.params["max_zoom"], 50000.0)            # capped
        self.assertEqual(m.fit, "cover")
        self.assertEqual((m.params["palette"], m.params["lightness"]), ("okhsl", 0.65))  # okhsl by default
        self.assertEqual((m.params["smoothing"], m.params["log_color"]), (3, True))       # calm, no shimmer
        self.assertEqual((j.params["smoothing"], j.params["morph_bars"], j.params["color_bars"]), (2, 64, 8))
        self.assertEqual(j.params["palette"], "okhsl")             # unknown palette falls back to okhsl
        self.assertEqual(j.params["iterations"], 16)                # clamped
        self.assertEqual(items[42].params["center"], (0.25, 0.0))
        joined = " ".join(show.warnings)
        self.assertIn("50000", joined)
        self.assertIn("palette", joined)


class TestColorMirror(unittest.TestCase):
    def test_okhsl_known_values(self):
        # published okhsl values for the sRGB primaries
        for rgb, hue_deg, light in (((1, 0, 0), 29.23, 0.5681), ((0, 1, 0), 142.50, 0.8445),
                                    ((0, 0, 1), 264.05, 0.3666)):
            h, s, l = vc.srgb_to_okhsl(*rgb)
            self.assertAlmostEqual(h * 360, hue_deg, delta=0.01)
            self.assertAlmostEqual(s, 1.0, places=3)
            self.assertAlmostEqual(l, light, places=3)
        self.assertEqual(vc.srgb_to_okhsl(0.5, 0.5, 0.5)[1], 0.0)  # gray has no saturation

    def test_round_trip(self):
        import random
        rnd = random.Random(3)
        for _ in range(500):
            c = [rnd.random() for _ in range(3)]
            back = vc.okhsl_to_srgb(*vc.srgb_to_okhsl(*c))
            self.assertLess(max(abs(x - y) for x, y in zip(c, back)), 1e-4)

    def test_white_stays_white(self):
        cream, warm, red = (245 / 255, 238 / 255, 220 / 255), (240 / 255, 232 / 255, 225 / 255), (200 / 255, 40 / 255, 40 / 255)
        for near_white in ((1.0, 1.0, 1.0), cream, warm):
            self.assertEqual(vc.white_keep(near_white, 0.92), 1.0)   # keeps its own look
            self.assertEqual(vc.white_keep(near_white, 0.0), 0.0)    # 0 turns it off
        self.assertEqual(vc.white_keep(red, 0.92), 0.0)              # real colors still turn
        self.assertEqual(vc.white_keep((0.78, 0.78, 0.78), 0.92), 0.0)
        pink = (250 / 255, 215 / 255, 225 / 255)                     # just under: eases in, no hard edge
        self.assertTrue(0.0 < vc.white_keep(pink, 0.92) < 1.0)
        import vis_core
        self.assertEqual(vis_core.DEFAULT_OUTPUT["keep_white"], 0.92)

    def test_mirror_like_bram(self):
        import random
        rnd = random.Random(4)
        for _ in range(300):
            c, ang = [rnd.random() for _ in range(3)], rnd.random()
            h, s, l = vc.srgb_to_okhsl(*c)
            h2, s2, l2 = vc.srgb_to_okhsl(*vc.mirror(c, ang))
            self.assertAlmostEqual(l2, l, places=3)              # lightness kept
            if s > 0.05:
                self.assertAlmostEqual(s2, s, delta=2e-3)        # saturation kept (okhsl can read a hair over 1 at the sRGB edge)
                self.assertAlmostEqual(min((h2 - (ang - h)) % 1, (ang - h - h2) % 1), 0, places=3)
            twice = vc.mirror(vc.mirror(c, ang), ang)            # mirroring twice gives it back
            self.assertLess(max(abs(x - y) for x, y in zip(c, twice)), 1 / 255)  # under one level
        for ang in (0.1, 0.6):
            for t in vc.true_hues(ang):                          # true hues stay put
                c = vc.okhsl_to_srgb(t, 0.7, 0.6)
                self.assertLess(max(abs(x - y) for x, y in zip(c, vc.mirror(c, ang))), 1e-4)


FAKE_YTDLP = r"""#!/usr/bin/env python3
# Stand-in for yt-dlp in tests: answers playlist queries and writes a test video.
import json, subprocess, sys
a = sys.argv[1:]
if "--flat-playlist" in a:
    print(json.dumps({"entries": [{"id": "AAAAAAAAAAA", "title": "First", "duration": 125},
                                  {"id": "BBBBBBBBBBB", "title": "Second", "duration": 61.0}]}))
    sys.exit(0)
if "fail" in a[-1]:
    print("ERROR: Video unavailable", file=sys.stderr); sys.exit(1)
out = a[a.index("-o") + 1].replace("%(ext)s", "mkv")
dur = 4.0
if "--download-sections" in a:
    s, e = a[a.index("--download-sections") + 1].lstrip("*").split("-")
    dur = float(e) - float(s)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=2560x1440:rate=24:duration={dur}",
                "-c:v", "mjpeg", "-q:v", "8", out], check=True)
"""


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not installed")
class TestClipGrab(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        fake = self.d / "bin" / "yt-dlp"
        fake.parent.mkdir()
        fake.write_text(FAKE_YTDLP)
        fake.chmod(0o755)
        self.old_path = os.environ["PATH"]
        os.environ["PATH"] = f"{fake.parent}{os.pathsep}{self.old_path}"

    def tearDown(self):
        os.environ["PATH"] = self.old_path
        self.tmp.cleanup()

    def test_import_adds_only_new_videos(self):
        lst = self.d / "clips.txt"
        lst.write_text("https://youtu.be/AAAAAAAAAAA  0:10 0:20  note=40\n")
        subprocess.run([sys.executable, str(Path(cg.__file__)), "import", "https://pl", str(lst)],
                       check=True, capture_output=True)
        text = lst.read_text()
        self.assertEqual(text.count("AAAAAAAAAAA"), 1)               # already there, kept as is
        self.assertIn("watch?v=BBBBBBBBBBB  # Second  (1:01)", text)
        clips, errors = cg.read_list(lst)
        self.assertEqual(errors, [])
        self.assertEqual([c.whole for c in clips], [False, True])     # new lines default to the whole video

    def test_fetch_cuts_and_converts(self):
        lst = self.d / "clips.txt"
        lst.write_text("# test\n"
                       "https://youtu.be/CCCCCCCCCCC  1:00 1:02.5  song=0 note=42 name=spin\n"
                       "https://youtu.be/DDDDDDDDDDD\n"
                       "https://youtu.be/failfailfail  0:01 0:02\n"
                       "https://youtu.be/EEEEEEEEEEE  0:05\n")
        out = self.d / "clips"
        (out).mkdir()
        (out / "old_whole.mp4").write_bytes(b"x")
        r = subprocess.run([sys.executable, str(Path(cg.__file__)), "fetch", str(lst), "--out", str(out)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)                             # one failure, one bad line
        self.assertIn("1 failed", r.stdout)
        self.assertIn("Video unavailable", r.stdout)
        self.assertIn("line 5: give both a start and an end", r.stdout)
        self.assertIn("no longer in the list (left alone): old_whole.mp4", r.stdout)
        self.assertIn('path = "clips/spin_1m00s-1m02.5s.mp4"', r.stdout)
        spin = out / "spin_1m00s-1m02.5s.mp4"
        clip = ClipSource(spin)
        self.assertEqual(clip.frame.shape[:2], (1080, 1920))          # scaled down to 1080p
        self.assertAlmostEqual(clip.fps, 30, delta=0.01)
        self.assertAlmostEqual(clip.end, 2.5, delta=0.1)              # just the segment
        clip.close()
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                "frame=pict_type", "-of", "csv=p=0", str(spin)], capture_output=True, text=True)
        types = [t.strip(",") for t in probe.stdout.split()]
        self.assertTrue(types and set(types) == {"I"})               # every frame a keyframe
        self.assertTrue((out / "DDDDDDDDDDD_whole.mp4").exists())
        self.assertEqual(list(out.glob("*.part*")), [])               # no half-made files left
        # second run skips what's made
        r2 = subprocess.run([sys.executable, str(Path(cg.__file__)), "fetch", str(lst), "--out", str(out)],
                            capture_output=True, text=True)
        self.assertIn("0 made, 2 already there", r2.stdout)


class TestSurprises(unittest.TestCase):
    def test_one_in_rate(self):
        import random
        rng = random.Random(5)
        hits = sum(one_in(rng, 32) for _ in range(32000))
        self.assertAlmostEqual(hits / 32000, 1 / 32, delta=0.004)
        self.assertFalse(any(one_in(rng, 0) for _ in range(100)))      # 0 = never

    def test_roll_pulse(self):
        import random
        from collections import Counter
        rng = random.Random(9)
        c = Counter(roll_pulse(rng, 32, 20) for _ in range(40000))
        self.assertAlmostEqual(c["big"] / 40000, 1 / 32, delta=0.004)
        self.assertAlmostEqual(c["skip"] / 40000, (31 / 32) / 20, delta=0.004)   # big ones are never skipped
        self.assertEqual(set(c), {"big", "skip", "normal"})
        self.assertEqual({roll_pulse(rng, 0, 0) for _ in range(50)}, {"normal"})  # 0 turns both off
        c3 = Counter(roll_pulse(rng, 32, 20, 8) for _ in range(40000))
        self.assertAlmostEqual(c3["medium"] / 40000, (31 / 32) * (19 / 20) / 8, delta=0.006)
        self.assertAlmostEqual(c3["big"] / 40000, 1 / 32, delta=0.004)
        import vis_core
        o = vis_core.DEFAULT_OUTPUT                                         # same peaks as before, gentler normal
        self.assertEqual((o["medium_pulse_size"], o["big_pulse_size"]), (2.0, 6.0))

    def test_pick_twist(self):
        import random
        rng = random.Random(6)
        picks = [pick_twist(rng, 4) for _ in range(8000)]
        plain = picks.count((0, False)) / len(picks)
        self.assertAlmostEqual(plain, 0.75, delta=0.02)
        self.assertEqual(set(picks) - {(0, False)}, set(TWISTS))      # all five twists turn up
        self.assertEqual({pick_twist(rng, 0) for _ in range(50)}, {(0, False)})
        faces = {pick_twist(rng, 2, "mirror") for _ in range(400)}       # faces: mirror only, never turned
        self.assertEqual(faces, {(0, False), (0, True)})
        self.assertEqual({pick_twist(rng, 1, False) for _ in range(50)}, {(0, False)})

    def test_matrix_without_twist_matches_plain_fit(self):
        oa = 16 / 9
        for a in (0.5, 1.0, 3.0):
            wider = a > oa
            cover = (oa / a, 1.0) if wider else (1.0, a / oa)
            contain = (1.0, a / oa) if wider else (oa / a, 1.0)
            for mode, (fx, fy) in (("cover", cover), ("contain", contain)):
                m = layer_matrix(a, oa, mode)
                for got, want in zip(m, (fx, 0.0, 0.0, fy)):
                    self.assertAlmostEqual(got, want)

    def test_twisted_pieces_fit(self):
        import numpy as np
        oa = 16 / 9
        for a in (0.6, 1.0, 2.5):
            for turns, mirrored in ((0, False),) + TWISTS:
                m = layer_matrix(a, oa, "contain", turns, mirrored)
                M = np.array([[m[0], m[2]], [m[1], m[3]]])
                inv = np.linalg.inv(M)
                corners = np.array([[x, y] for x in (-0.5, 0.5) for y in (-0.5, 0.5)]).T
                on_screen = inv @ corners                      # every corner of the piece is on screen
                self.assertLessEqual(np.abs(on_screen).max(), 0.5 + 1e-9)
                self.assertAlmostEqual(np.abs(on_screen).max(), 0.5)  # and it touches an edge
                mc = layer_matrix(a, oa, "cover", turns, mirrored)
                Mc = np.array([[mc[0], mc[2]], [mc[1], mc[3]]])
                self.assertLessEqual(np.abs(Mc @ corners).max(), 0.5 + 1e-9)  # cover: screen is all picture

    def test_twist_setting_loads(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.png").write_bytes(b"x")
            (Path(d) / "s.toml").write_text('[[song]]\n  [[song.item]]\n  note = 1\n  path = "a.png"\n  twist = false\n'
                                            '  [[song.item]]\n  note = 2\n  path = "a.png"\n'
                                            '  [[song.item]]\n  note = 3\n  path = "a.png"\n  twist = "mirror"\n'
                                            '  [[song.item]]\n  note = 4\n  path = "a.png"\n  twist = "sideways"\n')
            show = load_show(Path(d) / "s.toml")
        self.assertFalse(show.songs[0].items[1].twist)
        self.assertTrue(show.songs[0].items[2].twist)
        self.assertEqual(show.songs[0].items[3].twist, "mirror")
        self.assertFalse(show.songs[0].items[4].twist)                 # unknown value: kept upright
        self.assertIn('twist must be true, false or "mirror"', " ".join(show.warnings))
        self.assertEqual((show.output["big_pulse_chance"], show.output["twist_chance"]), (32, 4))
        self.assertEqual(show.output["skip_pulse_chance"], 20)
        self.assertIsNone(show.songs[0].items[1].skip_pulse_chance)         # art follows the show

    def test_fractals_drop_beats_more_often(self):
        self.assertEqual(generated_item("mandelbrot").skip_pulse_chance, 10)
        self.assertEqual(generated_item("julia").skip_pulse_chance, 10)
        self.assertIsNone(generated_item("hex").skip_pulse_chance)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.png").write_bytes(b"x")
            (Path(d) / "s.toml").write_text('[[song]]\n  [[song.item]]\n  note = 40\n  type = "mandelbrot"\n'
                                            '  [[song.item]]\n  note = 44\n  path = "a.png"\n  skip_pulse_chance = 0\n')
            items = load_show(Path(d) / "s.toml").songs[0].items
        self.assertEqual(items[40].skip_pulse_chance, 10)
        self.assertEqual(items[44].skip_pulse_chance, 0)                    # this piece never drops a beat


def spans_text(line):
    return "".join(s[0] for s in line)


class TestTextScenes(unittest.TestCase):
    def test_log_since(self):
        s = MidiState()
        s.feed([0x99, 36, 100], t=1.0)
        seq, new = s.log_since(0)
        self.assertEqual(new, [(1.0, (0x99, 36, 100))])
        s.feed([CLOCK], t=1.1)
        seq2, new2 = s.log_since(seq)
        self.assertEqual((seq2, new2), (seq + 1, [(1.1, (CLOCK,))]))
        self.assertEqual(s.log_since(seq2)[1], [])
        s.poke("kick", t=2.0)                                   # keyboard hits aren't MIDI bytes
        self.assertEqual(s.log_since(seq2)[1], [])

    def test_monitor_lines_match_midi_monitor(self):
        f = vt.make_feed("monitor", "FakeMPC")
        f.feed(0.5, [START])
        f.feed(1.25, [0x99, 36, 127])
        f.feed(1.30, [0x89, 36, 0])
        f.feed(1.40, [0xBF, 2, 64])
        f.feed(1.50, [0xCF, 3])
        f.feed(1.60, [0xFE])                                    # active sensing: hidden
        lines = [spans_text(l) for l in f.visible()]
        self.assertEqual(len(lines), 5)
        self.assertIn(" START ", lines[0])
        self.assertTrue(lines[1].startswith("    1.250  FakeMPC"))
        self.assertIn("ch10 on  C1   " + vt.BLOCK * 32, lines[1])  # velocity 127 = 32 blocks
        self.assertIn("ch10 off C1", lines[2])
        self.assertIn("CC   2 = 64", lines[3])
        self.assertIn(" program 3 ", lines[4])
        self.assertEqual(f.visible()[1][4][1], "lightred")      # channel 10 color, as midi-monitor
        self.assertEqual(vt.note_name(60), "C3")                # Logic: middle C = C3

    def test_monitor_bar_lines_follow_clock(self):
        f = vt.make_feed("monitor")
        f.feed(0.0, [START])
        for i in range(97):
            f.feed(i * 0.02, [CLOCK])
        bars = [spans_text(l) for l in f.visible() if "bar" in spans_text(l)]
        self.assertEqual(len(bars), 2)
        self.assertIn("bar 1", bars[0])
        self.assertIn("bar 2     125.0 BPM", bars[1])         # midi-monitor's exact spacing

    def test_hex_and_binary_rows(self):
        h = vt.make_feed("hex")
        for _ in range(6):                                      # 18 bytes: one full row and two more
            h.feed(0, [0x99, 0x24, 0x78])
        rows = [spans_text(l) for l in h.visible()]
        self.assertEqual(rows[0], "000000  99 24 78 99 24 78 99 24 78 99 24 78 99 24 78 99")
        self.assertEqual(rows[1], "000010  24 78")              # row still filling is shown too
        b = vt.make_feed("binary")
        b.feed(0, [0xF8])
        self.assertEqual(spans_text(b.visible()[0]), "000000  11111000")
        self.assertEqual(b.visible()[0][2][1], vt.DIM)          # clock bytes dim, like --hex
        self.assertEqual(vt.byte_color(0, 0x99)[0], "lightred")
        self.assertEqual(vt.byte_color(0, 0xFA), ("black", "green"))

    def test_screen_is_crisp_cga(self):
        import numpy as np
        scr = vt.TextScreen()
        f = vt.make_feed("monitor")
        for i in range(40):
            f.feed(i, [0x90 | (i % 16), 40 + i, 20 + 2 * i])
        img = scr.render(f.visible())
        self.assertEqual(img.shape, (scr.size[1], scr.size[0], 3))
        colors = {tuple(c) for c in np.unique(img.reshape(-1, 3), axis=0)}
        self.assertLessEqual(colors, {tuple(v) for v in vt.CGA.values()})  # no anti-aliasing
        self.assertGreater(len(colors), 5)
        self.assertEqual(scr.render([]).max(), 0)               # empty screen is black

    def test_every_scene_draws_with_a_row_half_full(self):
        scr = vt.TextScreen()
        for kind in vt.TEXT_KINDS:
            f = vt.make_feed(kind, "FakeMPC")
            f.feed(0.0, [START])
            for i in range(37):                                 # leaves the hex and binary rows part full
                f.feed(i * 0.02, [CLOCK])
                f.feed(i * 0.02, [0x99, 36, 100])
            for line in f.visible():
                self.assertTrue(all(len(s) == 3 for s in line), kind)
            self.assertGreater(scr.render(f.visible()).max(), 0, kind)

    def test_generated_items(self):
        self.assertEqual(GENERATED_KINDS, ("mandelbrot", "julia", "monitor", "hex", "binary"))
        j, m = generated_item("julia"), generated_item("monitor")
        self.assertEqual((j.params["palette"], j.fit, j.mirror, j.twist), ("okhsl", "cover", True, True))
        self.assertEqual((m.params, m.fit, m.mirror, m.twist), ({}, "contain", False, False))
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "s.toml").write_text('[[song]]\n  [[song.item]]\n  note = 50\n  type = "hex"\n'
                                            '  [[song.item]]\n  note = 51\n  type = "binary"\n  mirror = true\n')
            show = load_show(Path(d) / "s.toml")
        hx, bn = show.songs[0].items[50], show.songs[0].items[51]
        self.assertEqual((hx.kind, hx.path, hx.mirror, hx.twist), ("hex", None, False, False))
        self.assertTrue(bn.mirror)                              # any text scene can opt in
        self.assertEqual(show.warnings, [])
        self.assertTrue(show.output["generate"])


class TestFakeMPC(unittest.TestCase):
    def test_stops_by_itself(self):
        import fake_mpc
        self.assertEqual(fake_mpc.stop_tick(500, 124), 500 * 124 * 24)   # clock ticks in 500 minutes
        self.assertIsNone(fake_mpc.stop_tick(0, 124))                    # 0: until Ctrl+C

    def test_scene_plan(self):
        import fake_mpc
        self.assertEqual(fake_mpc.scene_plan(["39", "40:32", "41:24", "44"], 16),
                         [(39, 16), (40, 32), (41, 24), (44, 16)])
        for bad in (["40:0"], ["200"], ["x"]):
            with self.assertRaises(ValueError):
                fake_mpc.scene_plan(bad, 16)


class TestShuffle(unittest.TestCase):
    """At 120 BPM a bar is 2 seconds, so a clip's seconds / 2 = its bars."""

    def make(self, clips, **kw):
        import random
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "clips").mkdir()
        for name in clips:
            (d / "clips" / name).write_bytes(b"x")
        (d / "clips" / "notes.txt").write_text("not a clip")
        secs = dict(clips)
        kw.setdefault("max_clip_bars", 8)
        kw.setdefault("image_bars", 4)
        sh = vs.Shuffle(d / "clips", d / "shuffle.db", rng=random.Random(1),
                        seconds_of=lambda p: secs[p.name], **kw)
        self.c = lambda n: f"clips/{n}"
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(sh.close)
        return sh, d

    def set_counts(self, sh, **counts):
        sh.pool()
        for name, c in counts.items():
            sh.db.execute("UPDATE plays SET count = ? WHERE name = ?", (c, "clips/" + name.replace("_", ".")))
        sh.db.commit()

    def test_bars_for(self):
        self.assertEqual(vs.bars_for(7.3, 120, 8), 4)          # 3.65 bars rounds to 4
        self.assertEqual(vs.bars_for(0.4, 120, 8), 1)          # never under a bar
        self.assertEqual(vs.bars_for(300, 120, 8), 8)          # long YouTube segments capped

    def test_new_clips_start_below_the_usual_count(self):
        sh, d = self.make({"a.mp4": 4, "b.mp4": 4})
        self.assertEqual(set(sh.pool()), {"clips/a.mp4", "clips/b.mp4"})  # notes.txt isn't a clip
        self.set_counts(sh, a_mp4=10, b_mp4=10)
        (d / "clips" / "c.mp4").write_bytes(b"x")
        sh.seconds_of = lambda p: 4
        sh.pool()
        self.assertEqual(sh.count("clips/c.mp4"), 10 - vs.MODE_OFFSET)  # SoundTrack's MODE_OFFSET

    def test_least_shown_first_and_fills_the_gap(self):
        sh, _ = self.make({"long.mp4": 12, "short.mp4": 6, "worn.mp4": 4})
        self.set_counts(sh, long_mp4=1, short_mp4=1, worn_mp4=9)
        first = sh.start(0, 4, 120)                             # a 4 bar gap
        self.assertEqual((first.name, first.bars), ("clips/short.mp4", 3))  # least shown that fits: 3 of 4 bars
        self.assertEqual(first.end, 12)
        self.assertIsNone(sh.update(11.9, 120))
        second = sh.update(12, 120)
        self.assertEqual((second.name, second.bars, second.start, second.end), ("clips/long.mp4", 1, 12, 16))  # cut to fit
        self.assertEqual(sh.count("clips/short.mp4"), 2)              # the finished clip counts as shown
        self.assertIsNone(sh.update(16, 120))                   # gap used up: last clip carries on
        sh.stop()                                               # scene note
        self.assertEqual(sh.count("clips/long.mp4"), 2)
        self.assertFalse(sh.active)

    def test_no_length_chains_until_the_scene_note(self):
        sh, _ = self.make({"a.mp4": 4, "b.mp4": 4})
        p = sh.start(0, 0, 120)
        seen = [p.name]
        for beat in range(1, 81):                                # 20 bars
            nxt = sh.update(beat, 120)
            if nxt:
                seen.append(nxt.name)
                self.assertEqual(nxt.start % 4, 0)              # every change on a bar line
        self.assertGreaterEqual(len(seen), 10)                  # 2 bars each, 20 bars
        self.assertTrue(all(a != b for a, b in zip(seen, seen[1:])))  # never the same clip twice in a row

    def test_skip_doesnt_count_and_lands_on_a_bar_line(self):
        sh, _ = self.make({"a.mp4": 8, "b.mp4": 8, "c.mp4": 8})
        first = sh.start(0, 16, 120)
        before = sh.count(first.name)
        nxt = sh.skip(5, 120)                                   # pressed Next in bar 2
        self.assertNotEqual(nxt.name, first.name)
        self.assertEqual(sh.count(first.name), before)          # skipped: not counted
        self.assertEqual(nxt.start, 5)
        self.assertEqual((nxt.end - 8) % 4, 0)                  # hands over on a bar line
        late = sh.skip(7.5, 120)                                # half a beat before a bar line
        self.assertGreaterEqual(late.end - 7.5, 2)              # never a sliver of a clip

    def test_never_again_sticks(self):
        sh, d = self.make({"a.mp4": 4, "b.mp4": 4})
        first = sh.start(0, 8, 120)
        nxt = sh.never_again(1, 120)
        self.assertNotEqual(nxt.name, first.name)
        self.assertNotIn(first.name, sh.pool())
        import random
        again = vs.Shuffle(d / "clips", d / "shuffle.db", rng=random.Random(2), seconds_of=lambda p: 4)
        self.addCleanup(again.close)
        self.assertNotIn(first.name, again.pool())              # remembered for the next show

    def test_counts_carry_over(self):
        sh, d = self.make({"a.mp4": 4, "b.mp4": 4})
        p = sh.start(0, 2, 120)
        sh.stop()
        import random
        again = vs.Shuffle(d / "clips", d / "shuffle.db", rng=random.Random(3), seconds_of=lambda p: 4)
        self.addCleanup(again.close)
        self.assertNotEqual(again.start(0, 2, 120).name, p.name)  # the other one is now least shown

    def test_a_first_batch_starts_level(self):
        sh, _ = self.make({"a.mp4": 4, "b.mp4": 4, "c.mp4": 4})
        sh.pool()
        self.assertEqual({sh.count(self.c(n)) for n in ("a.mp4", "b.mp4", "c.mp4")}, {-vs.MODE_OFFSET})

    def test_art_joins_the_shuffle_and_new_pieces_come_up_first(self):
        import random
        sh, d = self.make({"a.mp4": 4})
        art = d / "art"
        (art / "misc").mkdir(parents=True)
        for n in ("old1.png", "old2.jpg", "fallback.png"):
            (art / n).write_bytes(b"x")
        (art / "misc" / "parked.png").write_bytes(b"x")         # moved aside: not shown
        sh2 = vs.Shuffle(d / "clips", d / "shuffle.db", art_folder=art, skip_files=[art / "fallback.png"],
                         image_bars=4, rng=random.Random(4), seconds_of=lambda p: 4)
        self.addCleanup(sh2.close)
        self.assertEqual(set(sh2.pool()), {"clips/a.mp4", "art/old1.png", "art/old2.jpg"})
        for n in sh2.pool():
            sh2.db.execute("UPDATE plays SET count = 5 WHERE name = ?", (n,))
        sh2.db.commit()
        (art / "brand_new.gif").write_bytes(b"x")              # just downloaded
        p = sh2.start(0, 16, 120)
        self.assertEqual((p.name, p.kind, p.bars), ("art/brand_new.gif", "image", 4))  # new: shown first
        self.assertEqual(sh2.count("art/brand_new.gif"), 5 - vs.MODE_OFFSET)
        shown = [p.name]
        for beat in range(1, 200):
            q = sh2.update(beat, 120)
            if q:
                shown.append(q.name)
        sh2.stop()
        self.assertGreater(shown.count("art/brand_new.gif"), 1)   # keeps coming up while it catches up

    def test_shuffle_settings(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "s.toml").write_text('[shuffle]\nnote = 200\nimage_bars = 2\n'
                                            '[[song]]\n  [[song.item]]\n  note = 61\n  type = "julia"\n')
            show = load_show(Path(d) / "s.toml")
        sc = show.shuffle
        self.assertEqual((sc["note"], sc["skip_note"], sc["image_bars"]), (60, 61, 2))  # bad note falls back
        self.assertEqual(sc["max_clip_bars"], 16)                          # long enough for colors to develop
        self.assertEqual(sc["clips"], Path(d) / "clips")
        self.assertEqual(sc["art"], Path(d) / "art")
        joined = " ".join(show.warnings)
        self.assertIn("shuffle note must be 0 to 127", joined)
        self.assertIn("note 61 is the shuffle skip note", joined)

    def test_empty_folder(self):
        sh, _ = self.make({})
        self.assertIsNone(sh.start(0, 8, 120))
        self.assertIsNone(sh.update(100, 120))
        self.assertIsNone(sh.skip(1, 120))


class TestShowFile(unittest.TestCase):
    def test_load(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.png").write_bytes(b"x")
            (d / "show.toml").write_text("""
[midi]
port = "FakeMPC"
[output]
pulse_source = "bogus"
[cc]
hue = 20
[[song]]
program = 2
name = "Lebanese Blonde"
  [[song.item]]
  note = 36
  path = "a.png"
  fit = "cover"
  [[song.item]]
  note = 37
  type = "clip"
  path = "missing.mp4"
  start = 4
  end = 12
  bpm = 96
  hold = true
  [[song.item]]
  note = 38
  type = "gif"
  path = "a.png"
""")
            show = load_show(d / "show.toml")
        self.assertEqual(show.midi["port"], "FakeMPC")
        self.assertEqual(show.midi["visual_channel"], 16)
        self.assertEqual(show.cc_map["hue"], 20)
        self.assertEqual(show.cc_map["zoom"], 1)
        self.assertEqual(show.output["pulse_source"], "kick")
        song = show.songs[2]
        self.assertEqual(song.name, "Lebanese Blonde")
        self.assertEqual(sorted(song.items), [36, 37])
        clip = song.items[37]
        self.assertEqual((clip.kind, clip.start, clip.end, clip.bpm, clip.hold), ("clip", 4.0, 12, 96, True))
        joined = " ".join(show.warnings)
        self.assertIn("pulse_source", joined)
        self.assertIn("missing.mp4", joined)
        self.assertIn("gif", joined)
        self.assertEqual(song.items[36].fit, "cover")
        self.assertEqual(clip.fit, "contain")  # default

    def test_example_show_parses(self):
        show = load_show(Path(__file__).parent / "show.toml")
        self.assertEqual(set(show.songs), {0, 1})
        self.assertEqual(show.songs[0].items[40].kind, "mandelbrot")
        self.assertEqual(show.songs[0].items[41].kind, "julia")
        self.assertEqual(show.warnings and [w for w in show.warnings if "note 4" in w], [])
        self.assertEqual(show.songs[0].items[38].kind, "clip")


class TestMedia(unittest.TestCase):
    def test_image_is_flipped_and_shrunk(self):
        with tempfile.TemporaryDirectory() as d:
            arr = np.zeros((100, 50, 3), np.uint8)
            arr[:10] = 255  # top band white
            Image.fromarray(arr).save(Path(d) / "t.png")
            src = ImageSource(Path(d) / "t.png", max_side=40)
        self.assertEqual(src.frame.shape, (40, 20, 3))
        self.assertTrue(src.frame[-1].mean() > 100)  # top row now last (GL order)
        self.assertTrue(src.update(0.016))
        self.assertFalse(src.update(0.016))

    def test_rotation_tag_is_followed(self):
        # stored upside down with a tag saying "turn it 180": must load the right way up
        upright = np.zeros((60, 40, 3), np.uint8)
        upright[:20] = 255                                       # white band at the top
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tagged.jpg"
            stored = Image.fromarray(upright).rotate(180)
            exif = Image.Exif()
            exif[0x0112] = 3                                      # Orientation: rotate 180
            stored.save(p, exif=exif, quality=95)
            src = ImageSource(p)
        top = np.flipud(src.frame)[:20].mean()                   # frame is stored bottom row first
        bottom = np.flipud(src.frame)[-20:].mean()
        self.assertGreater(top, 200)
        self.assertLess(bottom, 50)

    def test_colorless_detection(self):
        gray = np.tile(np.linspace(0, 255, 200).astype(np.uint8)[None, :, None], (100, 1, 3))
        self.assertTrue(is_colorless(gray))                              # black and white
        sepia = gray.astype(int) + np.array([12, 4, -8])
        self.assertFalse(is_colorless(np.clip(sepia, 0, 255).astype(np.uint8)))  # faint sepia keeps its colors
        scan = gray.astype(int) + np.array([3, 1, -2])                    # a slightly warm black and white scan
        self.assertTrue(is_colorless(np.clip(scan, 0, 255).astype(np.uint8)))
        colorful = np.zeros((100, 200, 3), np.uint8)
        colorful[:, :100] = (220, 40, 40)
        colorful[:, 100:] = (40, 80, 220)
        self.assertFalse(is_colorless(colorful))
        speck = gray.copy()
        speck[:5, :5] = (255, 0, 0)                                     # a red signature corner
        self.assertTrue(is_colorless(speck))                             # still a black and white piece

    def test_colorize_setting(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.png").write_bytes(b"x")
            (Path(d) / "s.toml").write_text('[[song]]\n  [[song.item]]\n  note = 1\n  path = "a.png"\n'
                                            '  [[song.item]]\n  note = 2\n  path = "a.png"\n  colorize = false\n'
                                            '  [[song.item]]\n  note = 3\n  path = "a.png"\n  colorize = true\n')
            show = load_show(Path(d) / "s.toml")
        items = show.songs[0].items
        self.assertEqual((items[1].colorize, items[2].colorize, items[3].colorize), (None, False, True))
        self.assertTrue(show.output["colorize"])                         # automatic by default

    def test_solid(self):
        self.assertEqual(ArraySource.solid((1, 2, 3)).frame[0, 0].tolist(), [1, 2, 3])

    @unittest.skipIf(cv2 is None, "opencv not installed")
    def test_clip_loops_and_stretches(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "c.avi")
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 36))
            for i in range(50):  # 5 s; brightness encodes the frame number
                w.write(np.full((36, 64, 3), i * 5, np.uint8))
            w.release()
            clip = ClipSource(path, start=1.0, end=2.0)
            level = lambda: clip.frame.mean() / 5
            self.assertAlmostEqual(level(), 10, delta=1)
            self.assertTrue(clip.update(0.0))       # first upload
            self.assertFalse(clip.update(0.01))     # still the same frame
            clip.update(0.5)
            self.assertAlmostEqual(level(), 15, delta=1)
            clip.update(0.6)                        # past end: loops to start + 0.1
            self.assertAlmostEqual(level(), 11, delta=1)
            clip.update(0.2, rate=2.0)              # double speed
            self.assertAlmostEqual(level(), 15, delta=1)
            with self.assertRaises(ValueError):
                ClipSource(path, start=3, end=2)
            clip.close()

            # pool: clips open ahead of time and each take gets a fresh one from the start
            pool = ClipPool()
            factory = lambda: ClipSource(path, start=2.0, end=3.0)
            pool.prepare(42, factory)
            a = pool.take(42)
            self.assertAlmostEqual(a.frame.mean() / 5, 20, delta=1)
            a.update(0.5)
            b = pool.take(42)                                   # second trigger: a new copy at the start
            self.assertIsNot(a, b)
            self.assertAlmostEqual(b.frame.mean() / 5, 20, delta=1)
            c = pool.take(7, factory)                           # never prepared: opens right away
            self.assertAlmostEqual(c.frame.mean() / 5, 20, delta=1)
            with self.assertRaises(RuntimeError):
                pool.take(9, lambda: ClipSource(Path(d) / "nope.mp4"))
            for s in (a, b, c):
                s.close()
            pool.shutdown()


if __name__ == "__main__":
    unittest.main()
