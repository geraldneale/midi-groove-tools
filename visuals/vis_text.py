"""Text scenes for the projector, after midi-monitor's three looks.

  monitor: the live MIDI monitor, one line per message, each channel in its own
           color from the 16-color IBM PC (CGA) palette, velocity as a block bar
  hex:     the raw MIDI bytes, 16 per row, status bytes in their channel's color
  binary:  the same bytes as zeroes and ones, 8 per row

Each feed turns raw MIDI bytes into colored lines; TextScreen draws the newest
lines onto an 80 by 25 character screen, like a CGA text monitor, as an RGB
numpy frame ready for the GPU. No GPU or MIDI library needed here.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# the 16-color IBM PC (CGA) palette, nothing else
CGA = {
    "black": (0, 0, 0), "blue": (0, 0, 170), "green": (0, 170, 0), "cyan": (0, 170, 170),
    "red": (170, 0, 0), "magenta": (170, 0, 170), "brown": (170, 85, 0), "lightgray": (170, 170, 170),
    "darkgray": (85, 85, 85), "lightblue": (85, 85, 255), "lightgreen": (85, 255, 85),
    "lightcyan": (85, 255, 255), "lightred": (255, 85, 85), "lightmagenta": (255, 85, 255),
    "yellow": (255, 255, 85), "white": (255, 255, 255),
}
# one color per MIDI channel, exactly as midi-monitor colors them
CHANNEL = {1: "lightcyan", 2: "cyan", 3: "lightblue", 4: "blue", 5: "lightgreen", 6: "green",
           7: "darkgray", 8: "lightgray", 9: "lightgray", 10: "lightred", 11: "white", 12: "red",
           13: "magenta", 14: "lightmagenta", 15: "brown", 16: "yellow"}
DIM = "darkgray"
BLOCK = "\u2588"  # full block, from the original PC character set
NAMES = "C C# D D# E F F# G G# A A# B".split()
TEXT_KINDS = ("monitor", "hex", "binary")


def note_name(n):
    return f"{NAMES[n % 12]}{n // 12 - 2}"  # Logic's convention: middle C = C3


class Feed:
    """Turns MIDI bytes into lines. A line is a list of (text, fg, bg) spans; bg may be None."""

    def __init__(self, keep=200):
        self.lines = deque(maxlen=keep)
        self.changed = True

    def emit(self, *spans):
        self.lines.append([s if len(s) == 3 else (s[0], s[1], None) for s in spans])
        self.changed = True

    def visible(self):
        return list(self.lines)


class MonitorFeed(Feed):
    def __init__(self, port="MIDI", keep=200):
        super().__init__(keep)
        self.port = port
        self.ticks, self.running, self.last_clock, self.gaps = 0, False, None, deque(maxlen=24)

    def bpm(self):
        return 60 / (24 * sum(self.gaps) / len(self.gaps)) if self.gaps else 0.0

    def line(self, t, *spans):
        self.emit((f"{t:9.3f}", DIM), ("  ", DIM), (f"{self.port[:14]:<14}", "cyan"), ("  ", DIM), *spans)

    def feed(self, t, data):
        if not data:
            return
        st = data[0]
        if st == 0xF8:
            if self.last_clock is not None and 0 < t - self.last_clock < 0.1:
                self.gaps.append(t - self.last_clock)
            self.last_clock = t
            if self.running:
                if self.ticks % 96 == 0:
                    self.line(t, (f" bar {self.ticks // 96 + 1:<4} {self.bpm():6.1f} BPM ", "white", "blue"))
                self.ticks += 1
            return
        if st in (0xFA, 0xFB, 0xFC):
            if st == 0xFA:
                self.ticks = 0
            self.running = st != 0xFC
            word = {0xFA: "START", 0xFB: "CONTINUE", 0xFC: "STOP"}[st]
            self.line(t, (f" {word} ", "black" if st != 0xFC else "white", "green" if st != 0xFC else "red"))
            return
        if st == 0xFE:
            return
        if st < 0xF0 and len(data) >= 2:
            kind, ch = st & 0xF0, (st & 0x0F) + 1
            col = CHANNEL[ch]
            tag = (f"ch{ch:<2} ", col)
            vel = data[2] if len(data) >= 3 else 0
            if kind == 0x90 and vel > 0:
                self.line(t, tag, (f"on  {note_name(data[1]):<4} {BLOCK * (vel // 4 + 1)}", col))
            elif kind in (0x80, 0x90):
                self.line(t, tag, (f"off {note_name(data[1])}", DIM))
            elif kind == 0xB0:
                self.line(t, tag, (f"CC {data[1]:3d} = {vel}", "magenta"))
            elif kind == 0xC0:
                self.line(t, tag, (f" program {data[1]} ", "white", "magenta"))
            elif kind == 0xE0:
                self.line(t, tag, (f"pitch bend {(vel << 7 | data[1]) - 8192}", "magenta"))
            else:
                self.line(t, tag, (" ".join(f"{b:02X}" for b in data), "lightgray"))
        elif st == 0xF0:
            self.line(t, (f"sysex {max(0, len(data) - 2)} bytes", "brown"))


def byte_color(i, status):
    """midi-monitor --hex coloring: status byte in its channel's color, clock dim, the rest white."""
    if i == 0 and status < 0xF0:
        return CHANNEL[(status & 0x0F) + 1], None
    if i == 0 and status in (0xFA, 0xFB):
        return "black", "green"
    if i == 0 and status == 0xFC:
        return "white", "red"
    if status == 0xF8:
        return DIM, None
    if status == 0xF0:
        return "brown", None
    return "lightgray", None


class ByteFeed(Feed):
    """The raw byte stream, per_row bytes a row, with a running offset like a hex dump."""

    def __init__(self, per_row=16, fmt="{:02X}", keep=200):
        super().__init__(keep)
        self.per_row, self.fmt = per_row, fmt
        self.cells, self.offset = [], 0

    def feed(self, t, data):
        if not data or data[0] == 0xFE:
            return
        for i, b in enumerate(data):
            fg, bg = byte_color(i, data[0])
            self.cells.append((self.fmt.format(b), fg, bg))
            if len(self.cells) == self.per_row:
                self.emit(*self._row(self.cells))
                self.offset += self.per_row
                self.cells = []
        self.changed = True

    def _row(self, cells):
        spans = [(f"{self.offset:06X}", DIM, None), ("  ", DIM, None)]
        for k, c in enumerate(cells):
            if k:
                spans.append((" ", DIM, None))
            spans.append(c)
        return spans

    def visible(self):
        """Finished rows plus the row still filling, so the screen moves with every byte."""
        return list(self.lines) + ([self._row(self.cells)] if self.cells else [])


def make_feed(kind, port="MIDI"):
    if kind == "monitor":
        return MonitorFeed(port)
    if kind == "hex":
        return ByteFeed(16, "{:02X}")
    if kind == "binary":
        return ByteFeed(8, "{:08b}")
    raise ValueError(kind)


FONT_PLACES = ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
               "/Library/Fonts/Courier New.ttf", "C:/Windows/Fonts/consola.ttf")


def load_font(size):
    for p in FONT_PLACES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


class TextScreen:
    """An 80 by 25 character screen. render() draws the newest lines, bottom row newest,
    with no anti-aliasing so characters are crisp pixels, like a CGA text monitor."""

    def __init__(self, cols=80, rows=25, font_size=16):
        self.cols, self.rows = cols, rows
        self.font = load_font(font_size)
        box = self.font.getbbox(BLOCK + "Mg0")
        self.cw = max(1, round(self.font.getlength("M")))
        self.ch = max(1, box[3] - min(0, box[1]) + 2)
        self.size = (cols * self.cw, rows * self.ch)  # (width, height) in pixels

    def render(self, lines):
        img = Image.new("RGB", self.size, CGA["black"])
        d = ImageDraw.Draw(img)
        d.fontmode = "1"  # no anti-aliasing: crisp pixel characters
        shown = lines[-self.rows:]
        top = self.rows - len(shown)
        for r, spans in enumerate(shown):
            y, col = (top + r) * self.ch, 0
            for span in spans:
                text, fg, bg = span if len(span) == 3 else (span[0], span[1], None)
                if col >= self.cols:
                    break
                text = text[: self.cols - col]
                x = col * self.cw
                if bg:
                    d.rectangle([x, y, x + len(text) * self.cw - 1, y + self.ch - 1], fill=CGA[bg])
                for k, chr_ in enumerate(text):  # one cell per character keeps the grid exact
                    if chr_ != " ":
                        d.text((x + k * self.cw, y + 1), chr_, font=self.font, fill=CGA[fg])
                col += len(text)
        return np.ascontiguousarray(np.flipud(np.asarray(img)))
