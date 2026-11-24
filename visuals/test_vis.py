"""Tests for the parts that don't need a GPU or MIDI hardware: python3 -m unittest test_vis"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vis_core import (CLOCK, CONTINUE, PPQN, SPP, START, STOP, MidiState,
                      StrobeLimiter, beat_pulse, decay, load_show, Hit)
from vis_media import ArraySource, ClipSource, ImageSource, cv2


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

    def test_strobe_limiter(self):
        lim = StrobeLimiter(3)
        allowed = [t / 100 for t in range(0, 300, 5) if lim.allow(t / 100)]  # 20 Hz snare roll for 3 s
        self.assertLessEqual(len(allowed), 9)
        self.assertTrue(all(b - a >= 1 / 3 - 1e-9 for a, b in zip(allowed, allowed[1:])))
        self.assertFalse(StrobeLimiter(0).allow(5))


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

    def test_example_show_parses(self):
        show = load_show(Path(__file__).parent / "show.toml")
        self.assertEqual(set(show.songs), {0, 1})
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


if __name__ == "__main__":
    unittest.main()
