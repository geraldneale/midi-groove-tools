"""Groovelab26 visuals: core logic.

MIDI parsing, show config loading and envelopes.
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
        self._log = deque(maxlen=4096)  # (seq, time, bytes) of every message, for the text scenes
        self._seq = 0

    # ---- input side (MIDI thread) ----
    def feed(self, data, t=None):
        if not data:
            return
        t = self._clock() if t is None else t
        status = data[0]
        with self._lock:
            self.last_msg_t = t
            self._seq += 1
            self._log.append((self._seq, t, tuple(data)))
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
    def log_since(self, seq):
        """Raw messages newer than seq: (newest seq, [(time, bytes), ...])."""
        with self._lock:
            new = [(t, d) for s, t, d in self._log if s > seq]
            return self._seq, new

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


# ---- surprises ----
TWISTS = ((0, True),)  # a left to right mirror only: upside down and sideways rarely work


def turn_in(beats_since, hold_bars, fade_bars):
    """How far a new picture's colors may turn (0 to 1): it shows in its original colors for
    hold_bars, then the turning eases in over fade_bars. None (no start known) turns fully."""
    if beats_since is None:
        return 1.0
    t = (beats_since / 4.0 - hold_bars) / fade_bars if fade_bars > 0 else float(beats_since / 4.0 >= hold_bars)
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def one_in(rng, n):
    """True about once in every n calls (n <= 0: never)."""
    return n > 0 and rng.random() < 1.0 / n


def roll_pulse(rng, big_chance, skip_chance, medium_chance=0):
    """For each kick: "big" now and then, "skip" (no pulse at all) now and then, "medium" a bit
    more often, else "normal" (a subtle pulse). A big one is never also skipped."""
    if one_in(rng, big_chance):
        return "big"
    if one_in(rng, skip_chance):
        return "skip"
    return "medium" if one_in(rng, medium_chance) else "normal"


def pick_twist(rng, chance, allowed=True):
    """For a new scene: (quarter_turns, mirrored), usually (0, False).
    allowed: True for any twist, "mirror" for a left to right mirror only (faces), False for none."""
    if not allowed or not one_in(rng, chance):
        return (0, False)
    return (0, True) if allowed == "mirror" else rng.choice(TWISTS)


def layer_matrix(img_aspect, out_aspect, mode="contain", turns=0, mirrored=False):
    """2x2 matrix (column major, for the shader's mat2) taking a screen offset from the centre
    (0..1 units both ways) to a texture offset, for the given fit and twist.
    contain shows the whole (turned) piece; cover fills the frame and crops."""
    odd = turns % 2 == 1
    a = 1.0 / img_aspect if odd else img_aspect          # aspect of the piece as it appears
    box_h = max(1.0, out_aspect / a) if mode == "cover" else min(1.0, out_aspect / a)
    box_w = a * box_h                                     # on screen, in screen heights
    img_w, img_h = (box_h, box_w) if odd else (box_w, box_h)
    c = (1, 0, -1, 0)[turns % 4]
    s = (0, 1, 0, -1)[turns % 4]
    f = -1.0 if mirrored else 1.0
    # texture = diag(1/img_w, 1/img_h) . mirror . rotate back . (screen x in heights)
    m00, m01 = f * c * out_aspect / img_w, f * s / img_w
    m10, m11 = -s * out_aspect / img_h, c / img_h
    return (m00, m10, m01, m11)


# ---- show config ----
DEFAULT_MIDI = {"port": "MPC", "visual_channel": 16, "drum_channel": 10,
                "kick_notes": [36], "snare_notes": [38, 40], "hat_notes": [42, 44, 46]}
DEFAULT_OUTPUT = {"width": 1920, "height": 1080, "preview_width": 1280,
                  "fullscreen": False, "display": 0, "fps": 60, "offset_ms": 0,
                  "pulse_source": "kick", "fallback": "",
                  "idle_seconds": 5.0, "crossfade": 0.25, "fit": "contain",
                  "mirror_bars": 12, "big_pulse_chance": 32, "big_pulse_size": 6.0, "skip_pulse_chance": 20,
                  "medium_pulse_chance": 8, "medium_pulse_size": 2.0,
                  "colorize": True, "colorize_saturation": 0.8, "colorize_hue": 0.62, "colorize_span": 0.55,
                  "colorize_low": 0.15, "colorize_high": 0.85, "keep_white": 0.92,
                  "original_bars": 4, "original_fade_bars": 2,
                  "twist_chance": 16, "generate": True}
DEFAULT_SHUFFLE = {"enabled": True, "note": 60, "skip_note": 61, "clips": "clips", "art": "art",
                   "image_bars": 16, "max_clip_bars": 16, "skip_fade_beats": 1.0}
DEFAULT_CC = {"zoom": 1, "hue": 2, "kaleido": 3, "feedback": 4, "rgb_split": 5,
              "brightness": 7, "drift": 8, "shimmer": 9,
              "mirror": 10, "true_angle": 11}
DEFAULT_CC_VALUES = {"zoom": 0.5, "hue": 0.0, "kaleido": 0.0, "feedback": 0.0,
                     "rgb_split": 0.0, "brightness": 1.0,
                     "drift": 0.5, "shimmer": 0.3, "mirror": 0.0, "true_angle": 0.0}


# ---- fractals (drawn live on the GPU, no file) ----
FRACTAL_KINDS = ("mandelbrot", "julia")
TEXT_KINDS = ("monitor", "hex", "binary")  # midi-monitor's looks, drawn from the live MIDI
GENERATED_KINDS = FRACTAL_KINDS + TEXT_KINDS  # made live, no file; also stand in for missing art
MANDELBROT_SPOTS = {  # points on the edge of the set that stay detailed as you dive
    "seahorse": (-0.743643887037151, 0.131825904205330),
    "spiral": (-0.761574, -0.0847596),
}
PALETTES = {  # phase offsets for the cosine color palette (red, green, blue)
    "rainbow": (0.0, 0.33, 0.67),
    "fire": (0.0, 0.1, 0.2),
    "ice": (0.5, 0.6, 0.7),
    "acid": (0.8, 0.9, 0.3),
    "okhsl": None,  # hue steps round the okhsl wheel at steady saturation and lightness (Bram Cohen's color space)
}
FRACTAL_DEFAULTS = {
    "mandelbrot": {"spot": "seahorse", "dive_bars": 16, "max_zoom": 20000.0, "iterations": 300,
                   "palette": "okhsl", "saturation": 0.9, "lightness": 0.65,
                   "color_bars": 4, "color_bands": 0.6, "log_color": True, "smoothing": 3,
                   "resolution": 0.75},
    "julia": {"radius": 0.7885, "morph_bars": 64, "start": 0.5, "zoom": 1.0, "iterations": 200,
              "palette": "okhsl", "saturation": 0.9, "lightness": 0.65,
              "color_bars": 8, "color_bands": 0.05, "log_color": False, "smoothing": 2,
              "resolution": 0.75},
}


def dive_zoom(beats, dive_bars, max_zoom):
    """Zoom factor for the Mandelbrot dive: in over dive_bars, back out over dive_bars, repeat.
    Eased at both ends, so it lingers at the top and at full depth."""
    span = max(1e-6, dive_bars * 4.0)
    phase = (beats / span) % 2.0
    u = phase if phase < 1.0 else 2.0 - phase
    u = u * u * (3.0 - 2.0 * u)  # smoothstep
    return max_zoom ** u


def julia_c(beats, morph_bars, radius, start=0.0):
    """The Julia set's constant walks a circle once every morph_bars; the shape morphs as it goes.
    start is where on the circle it begins, as a fraction of a turn (0.5 = the solid blob shape)."""
    a = 2.0 * math.pi * (start + beats / max(1e-6, morph_bars * 4.0))
    return (radius * math.cos(a), radius * math.sin(a))


@dataclass
class Item:
    note: int
    kind: str  # "image", "clip", "mandelbrot" or "julia"
    path: Path | None
    start: float = 0.0
    end: float | None = None
    bpm: float | None = None  # clip's own tempo, for tempo stretch
    hold: bool = False  # clear to black on note off
    gain: float = 1.0
    fit: str = "contain"  # contain = whole piece, black around it; cover = fill frame, crop
    params: dict = field(default_factory=dict)  # fractal settings
    twist: bool | str = True  # True: may be mirrored or turned (see twist_chance); "mirror": only
    # mirrored left to right, never turned (faces); False: always as it is
    mirror: bool = True  # follows the color mirroring; text scenes keep their true CGA colors by default
    skip_pulse_chance: int | None = None  # this piece's own beat drop rate; None: the show's setting
    colorize: bool | None = None  # color a black and white piece by lightness; None: automatic


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
    shuffle: dict = field(default_factory=dict)


def _resolve(base, p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else base / p


def _twist(value, prog, note, warnings):
    if value in (True, False, "mirror"):
        return value
    warnings.append(f'song {prog}: note {note} twist must be true, false or "mirror"; using false')
    return False


FRACTAL_SKIP = 10  # fractals drop a beat more often than art (1 in 10 vs the show's 1 in 20)


def _skip(it, kind):
    if "skip_pulse_chance" in it:
        return int(it["skip_pulse_chance"])
    return FRACTAL_SKIP if kind in FRACTAL_KINDS else None


def generated_item(kind, note=-1):
    """A scene made live (fractal or text) with default settings, e.g. to stand in for missing art."""
    params = _fractal_params(kind, {}, "-", note, []) if kind in FRACTAL_KINDS else {}
    return Item(note, kind, None, fit="cover" if kind in FRACTAL_KINDS else "contain",
                params=params, twist=kind not in TEXT_KINDS, mirror=kind not in TEXT_KINDS,
                skip_pulse_chance=FRACTAL_SKIP if kind in FRACTAL_KINDS else None)


def _fractal_params(kind, it, prog, note, warnings):
    params = dict(FRACTAL_DEFAULTS[kind])
    for k in params:
        if k in it:
            params[k] = it[k]
    where = f"song {prog}: note {note}"
    if params["palette"] not in PALETTES:
        warnings.append(f"{where} palette must be one of {', '.join(PALETTES)}; using okhsl")
        params["palette"] = "okhsl"
    params["saturation"] = min(1.0, max(0.0, float(params["saturation"])))
    params["lightness"] = min(0.95, max(0.05, float(params["lightness"])))
    if kind == "mandelbrot":
        if "center" in it:
            params["center"] = tuple(float(v) for v in it["center"])
        elif params["spot"] in MANDELBROT_SPOTS:
            params["center"] = MANDELBROT_SPOTS[params["spot"]]
        else:
            warnings.append(f"{where} spot must be one of {', '.join(MANDELBROT_SPOTS)} (or give center = [x, y]); using seahorse")
            params["center"] = MANDELBROT_SPOTS["seahorse"]
        if float(params["max_zoom"]) > 50000:
            warnings.append(f"{where} max_zoom above 50000 breaks into blocks on the GPU; capped at 50000")
            params["max_zoom"] = 50000.0
    params["resolution"] = min(1.0, max(0.25, float(params["resolution"])))
    params["iterations"] = min(2000, max(16, int(params["iterations"])))
    params["smoothing"] = min(4, max(1, int(params["smoothing"])))   # 1 = off, 2 = 4 samples, 3 = 9, 4 = 16
    params["log_color"] = bool(params["log_color"])
    return params


def load_show(path):
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    base = path.parent
    warnings = []
    midi = {**DEFAULT_MIDI, **raw.get("midi", {})}
    output = {**DEFAULT_OUTPUT, **raw.get("output", {})}
    cc_map = {**DEFAULT_CC, **raw.get("cc", {})}
    cc_defaults = {**DEFAULT_CC_VALUES, **raw.get("cc_defaults", {})}
    # the snare flash was taken out for good (photosensitivity); older show files may still mention it
    if "flash" in cc_map or "flash" in cc_defaults or "max_flash_hz" in output:
        warnings.append("the screen flash has been removed; the flash lines in show.toml are ignored")
    for d in (cc_map, cc_defaults, output):
        d.pop("flash", None)
        d.pop("max_flash_hz", None)
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
            if kind not in ("image", "clip") + GENERATED_KINDS:
                warnings.append(f"song {prog}: unknown type {kind!r}, skipped")
                continue
            if "note" not in it:
                warnings.append(f"song {prog}: item needs a note, skipped")
                continue
            note = int(it["note"])
            p, params = None, {}
            if kind in FRACTAL_KINDS:
                params = _fractal_params(kind, it, prog, note, warnings)
            elif kind in TEXT_KINDS:
                pass
            elif "path" not in it:
                warnings.append(f"song {prog}: note {note} needs a path, skipped")
                continue
            else:
                p = _resolve(base, it["path"])
                if not p.exists():
                    warnings.append(f"song {prog}: missing file {p}")
            fit = it.get("fit", output["fit"])
            if fit not in ("contain", "cover"):
                warnings.append(f"song {prog}: note {note} fit must be contain or cover; using {output['fit']}")
                fit = output["fit"]
            if note in items:
                warnings.append(f"song {prog}: note {note} used twice, last one wins")
            items[note] = Item(note, kind, p, float(it.get("start", 0.0)),
                               it.get("end"), it.get("bpm"),
                               bool(it.get("hold", False)), float(it.get("gain", 1.0)),
                               "cover" if kind in FRACTAL_KINDS else fit, params,
                               _twist(it.get("twist", kind not in TEXT_KINDS), prog, note, warnings),
                               bool(it.get("mirror", kind not in TEXT_KINDS)),
                               _skip(it, kind),
                               it.get("colorize"))
        if prog in songs:
            warnings.append(f"program {prog} defined twice, last one wins")
        songs[prog] = Song(prog, s.get("name", f"program {prog}"), items)
    shuffle = {**DEFAULT_SHUFFLE, **raw.get("shuffle", {})}
    for key in ("note", "skip_note"):
        if not 0 <= int(shuffle[key]) <= 127:
            warnings.append(f"shuffle {key} must be 0 to 127; using {DEFAULT_SHUFFLE[key]}")
            shuffle[key] = DEFAULT_SHUFFLE[key]
    shuffle["clips"], shuffle["art"] = _resolve(base, shuffle["clips"]), _resolve(base, shuffle["art"])
    if shuffle["enabled"]:
        for s in songs.values():
            for key in ("note", "skip_note"):
                if int(shuffle[key]) in s.items:
                    warnings.append(f"song {s.program}: note {shuffle[key]} is the shuffle {key.replace('_', ' ')}; "
                                    f"its item is never shown")
    return Show(midi, output, cc_map, cc_defaults, songs, base, warnings, shuffle)
