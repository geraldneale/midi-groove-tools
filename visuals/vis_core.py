"""Groovelab26 visuals: core logic.

MIDI parsing, show config loading, envelopes and the strobe limiter.
Nothing here touches the GPU or a MIDI library, so it runs and tests anywhere.
"""
from __future__ import annotations

import math
import threading
import time

try:
    import tomllib
except ImportError:  # Python 3.9 and 3.10
    import tomli as tomllib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

CLOCK, START, CONTINUE, STOP, SPP = 0xF8, 0xFA, 0xFB, 0xFC, 0xF2
PPQN = 24  # MIDI clock pulses per quarter note


@dataclass
class Hit:
    t: float = -1e9
    vel: float = 0.0


class MidiState:
    """Thread safe view of the incoming MIDI stream.

    feed() is called from the MIDI callback thread with raw bytes.
    The render loop reads through the getters and drain_events().
    """

    def __init__(self, visual_ch=16, drum_ch=10, kick_notes=(36,),
                 snare_notes=(38, 40), hat_notes=(42, 44, 46),
                 clock=time.monotonic):
        self._lock = threading.Lock()
        self._clock = clock
        self.visual_ch = visual_ch - 1
        self.drum_ch = drum_ch - 1
        self.kick_notes = set(kick_notes)
        self.snare_notes = set(snare_notes)
        self.hat_notes = set(hat_notes)

        self.running = False
        self.ticks = 0
        self.bpm = 120.0
        self._last_tick_t = None
        self._intervals = deque(maxlen=PPQN * 2)
        self.cc: dict[int, float] = {}
        self.program = 0
        self.kick, self.snare, self.hat = Hit(), Hit(), Hit()
        self.last_msg_t = None
        self._events = deque()

    # ---- input side (MIDI thread) ----
    def feed(self, data, t=None):
        if not data:
            return
        t = self._clock() if t is None else t
        status = data[0]
        with self._lock:
            self.last_msg_t = t
            if status == CLOCK:
                self._on_clock(t)
            elif status == START:
                self.running = True
                self.ticks = 0
                self._events.append(("start",))
            elif status == CONTINUE:
                self.running = True
                self._events.append(("continue",))
            elif status == STOP:
                self.running = False
                self._events.append(("stop",))
            elif status == SPP and len(data) >= 3:
                sixteenths = data[1] | (data[2] << 7)
                self.ticks = sixteenths * (PPQN // 4)
                self._events.append(("locate", self.ticks))
            elif status < 0xF0 and len(data) >= 2:
                self._on_channel(status & 0xF0, status & 0x0F, data, t)

    def poke(self, kind, vel=1.0, t=None):
        """Manual drum hit from the keyboard (rehearsal). Does not count as MIDI activity."""
        t = self._clock() if t is None else t
        with self._lock:
            setattr(self, kind, Hit(t, vel))

    def _on_clock(self, t):
        if self._last_tick_t is not None:
            dt = t - self._last_tick_t
            if 0 < dt < 0.1:  # longer gaps mean the clock paused
                self._intervals.append(dt)
            if len(self._intervals) >= PPQN // 2:
                avg = sum(self._intervals) / len(self._intervals)
                self.bpm = 60.0 / (avg * PPQN)
        self._last_tick_t = t
        if self.running:
            self.ticks += 1

    def _on_channel(self, kind, ch, data, t):
        vel = data[2] if len(data) >= 3 else 0
        if kind == 0x90 and vel > 0:
            if ch == self.visual_ch:
                self._events.append(("note_on", data[1], vel / 127))
            if ch == self.drum_ch:
                n = data[1]
                if n in self.kick_notes:
                    self.kick = Hit(t, vel / 127)
                elif n in self.snare_notes:
                    self.snare = Hit(t, vel / 127)
                elif n in self.hat_notes:
                    self.hat = Hit(t, vel / 127)
        elif kind == 0x80 or kind == 0x90:  # note off, or note on with velocity 0
            if ch == self.visual_ch:
                self._events.append(("note_off", data[1]))
        elif kind == 0xB0 and len(data) >= 3 and ch == self.visual_ch:
            self.cc[data[1]] = data[2] / 127
        elif kind == 0xC0 and ch == self.visual_ch:
            self.program = data[1]
            self._events.append(("program", data[1]))

    # ---- output side (render thread) ----
    def drain_events(self):
        with self._lock:
            ev = list(self._events)
            self._events.clear()
        return ev

    def hits(self):
        with self._lock:
            return self.kick, self.snare, self.hat

    def cc_value(self, number, default):
        with self._lock:
            return self.cc.get(number, default)

    def beats(self, now, offset_s=0.0):
        """Song position in beats, interpolated between clock pulses.

        offset_s > 0 looks ahead, so clock driven effects land earlier.
        """
        with self._lock:
            ticks, bpm, last = self.ticks, self.bpm, self._last_tick_t
            running = self.running
        frac = 0.0
        if running and last is not None:
            frac = min(1.0, (now - last) * bpm * PPQN / 60.0)
        return (ticks + frac) / PPQN + offset_s * bpm / 60.0

    def idle_for(self, now):
        """Seconds since the last real MIDI message, or None if none ever arrived."""
        with self._lock:
            return None if self.last_msg_t is None else now - self.last_msg_t


# ---- envelopes and safety ----
def decay(now, hit, tau):
    dt = now - hit.t
    return 0.0 if dt < 0 else hit.vel * math.exp(-dt / tau)


def beat_pulse(beats, sharpness=6.0):
    return math.exp(-(beats % 1.0) * sharpness)


class StrobeLimiter:
    """Caps full screen flashes at max_hz (photosensitivity safety). 0 disables flashes."""

    def __init__(self, max_hz=3.0):
        self.min_gap = 1.0 / max_hz if max_hz > 0 else None
        self.last = -1e9

    def allow(self, t):
        if self.min_gap is None or t - self.last < self.min_gap:
            return False
        self.last = t
        return True


# ---- show config ----
DEFAULT_MIDI = {"port": "MPC", "visual_channel": 16, "drum_channel": 10,
                "kick_notes": [36], "snare_notes": [38, 40], "hat_notes": [42, 44, 46]}
DEFAULT_OUTPUT = {"width": 1920, "height": 1080, "preview_width": 1280,
                  "fullscreen": False, "display": 0, "fps": 60, "offset_ms": 0,
                  "pulse_source": "kick", "max_flash_hz": 3.0, "fallback": "",
                  "idle_seconds": 5.0, "crossfade": 0.25, "fit": "contain"}
DEFAULT_CC = {"zoom": 1, "hue": 2, "kaleido": 3, "feedback": 4, "rgb_split": 5,
              "flash": 6, "brightness": 7, "drift": 8, "shimmer": 9}
DEFAULT_CC_VALUES = {"zoom": 0.5, "hue": 0.0, "kaleido": 0.0, "feedback": 0.0,
                     "rgb_split": 0.0, "flash": 0.0, "brightness": 1.0,
                     "drift": 0.5, "shimmer": 0.3}


@dataclass
class Item:
    note: int
    kind: str  # "image" or "clip"
    path: Path
    start: float = 0.0
    end: float | None = None
    bpm: float | None = None  # clip's own tempo, for tempo stretch
    hold: bool = False  # clear to black on note off
    gain: float = 1.0
    fit: str = "contain"  # contain = whole piece, black around it; cover = fill frame, crop


@dataclass
class Song:
    program: int
    name: str
    items: dict[int, Item]


@dataclass
class Show:
    midi: dict
    output: dict
    cc_map: dict
    cc_defaults: dict
    songs: dict[int, Song]
    base: Path
    warnings: list = field(default_factory=list)


def _resolve(base, p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else base / p


def load_show(path):
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    base = path.parent
    warnings = []
    midi = {**DEFAULT_MIDI, **raw.get("midi", {})}
    output = {**DEFAULT_OUTPUT, **raw.get("output", {})}
    cc_map = {**DEFAULT_CC, **raw.get("cc", {})}
    cc_defaults = {**DEFAULT_CC_VALUES, **raw.get("cc_defaults", {})}
    if output["fit"] not in ("contain", "cover"):
        warnings.append(f"fit must be contain or cover, got {output['fit']!r}; using contain")
        output["fit"] = "contain"
    if output["pulse_source"] not in ("kick", "beat"):
        warnings.append(f"pulse_source must be kick or beat, got {output['pulse_source']!r}; using kick")
        output["pulse_source"] = "kick"
    if output["fallback"]:
        output["fallback"] = _resolve(base, output["fallback"])
        if not output["fallback"].exists():
            warnings.append(f"missing fallback image: {output['fallback']}")

    songs = {}
    for s in raw.get("song", []):
        prog = int(s.get("program", len(songs)))
        items = {}
        for it in s.get("item", []):
            kind = it.get("type", "image")
            if kind not in ("image", "clip"):
                warnings.append(f"song {prog}: unknown type {kind!r}, skipped")
                continue
            if "note" not in it or "path" not in it:
                warnings.append(f"song {prog}: item needs note and path, skipped")
                continue
            p = _resolve(base, it["path"])
            if not p.exists():
                warnings.append(f"song {prog}: missing file {p}")
            note = int(it["note"])
            fit = it.get("fit", output["fit"])
            if fit not in ("contain", "cover"):
                warnings.append(f"song {prog}: note {note} fit must be contain or cover; using {output['fit']}")
                fit = output["fit"]
            if note in items:
                warnings.append(f"song {prog}: note {note} used twice, last one wins")
            items[note] = Item(note, kind, p, float(it.get("start", 0.0)),
                               it.get("end"), it.get("bpm"),
                               bool(it.get("hold", False)), float(it.get("gain", 1.0)), fit)
        if prog in songs:
            warnings.append(f"program {prog} defined twice, last one wins")
        songs[prog] = Song(prog, s.get("name", f"program {prog}"), items)
    return Show(midi, output, cc_map, cc_defaults, songs, base, warnings)
