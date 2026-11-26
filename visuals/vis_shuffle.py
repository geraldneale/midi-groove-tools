"""Fill gaps in the show with video clips, after SoundTrack (geraldneale/SoundTrack).

SoundTrack plays a schedule of fixed songs and shuffles other songs into the time
between them, least played first, new songs entering a little below the usual play
count so they come up more at first. The same idea for the visuals:

  - fixed points: the scenes the MPC's visual track calls for
  - a gap: a shuffle note on that track; its velocity is the gap's length in bars
  - fills: video clips from the clips folder and pictures from the art folder, chained
    to fill the gap, each played for a whole number of bars (looped or cut) so every
    change lands on a bar line. Anything dropped into either folder joins the shuffle,
    no show file editing needed, and new arrivals come up more often at first
  - play counts and a never-again list live in shuffle.db in the show folder, so
    every night is different and the counts carry over between shows and machines

No GPU or MIDI library here; times are in beats, from the band's clock.
"""
from __future__ import annotations

import math
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

CLIP_TYPES = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".gif")
MODE_OFFSET = 3  # as in SoundTrack: new clips start this many plays below the usual count


@dataclass
class Pick:
    name: str     # e.g. "art/piece.png" or "clips/spin.mp4"
    path: Path
    kind: str     # "image" or "clip"
    bars: int
    start: float  # beat it starts on
    end: float    # beat it hands over on

    def __repr__(self):
        return f"Pick({self.name}, {self.bars} bars)"


def clip_seconds(path):
    """Length of a video file in seconds (0 if unreadable)."""
    try:
        import cv2
    except ImportError:
        return 0.0
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return n / fps if fps > 0 else 0.0
    finally:
        cap.release()


def bars_for(seconds, bpm, max_bars):
    """A clip's natural length rounded to whole bars (at least 1, at most max_bars)."""
    bar = 240.0 / max(bpm, 1e-6)
    return int(min(max_bars, max(1, round(seconds / bar))))


class Shuffle:
    def __init__(self, clips_folder, db_path, art_folder=None, skip_files=(), image_bars=16,
                 max_clip_bars=16, mode_offset=MODE_OFFSET, rng=None, seconds_of=clip_seconds,
                 now=time.time):
        self.clips_folder, self.db_path = Path(clips_folder), Path(db_path)
        self.art_folder = Path(art_folder) if art_folder else None
        self.skip_files = {Path(f).resolve() for f in skip_files}  # e.g. the fallback image
        self.image_bars = image_bars
        self.max_clip_bars, self.mode_offset = max_clip_bars, mode_offset
        self.rng = rng or random.Random()
        self.seconds_of, self.now = seconds_of, now
        self.db = sqlite3.connect(str(self.db_path))
        self.db.execute("CREATE TABLE IF NOT EXISTS plays (name TEXT PRIMARY KEY, count INTEGER, "
                        "seconds REAL, last TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS never (name TEXT PRIMARY KEY, since TEXT)")
        self.db.commit()
        self.current = self.upcoming = None
        self.gap_end = self.origin = None
        self.skipped = False

    # ---- the pool and its counts ----
    def _files(self):
        """(name, path, kind) for every clip and picture, top level of each folder only,
        so anything moved into a subfolder like art/misc is left out."""
        out = []
        for folder, kind, types in ((self.clips_folder, "clip", CLIP_TYPES),
                                    (self.art_folder, "image", IMAGE_TYPES)):
            if folder is None or not folder.is_dir():
                continue
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in types and p.resolve() not in self.skip_files:
                    out.append((f"{folder.name}/{p.name}", p, kind))
        return out

    def pool(self):
        """{name: (path, kind, seconds, count)} for everything not on the never list."""
        never = {r[0] for r in self.db.execute("SELECT name FROM never")}
        known = {r[0]: (r[1], r[2]) for r in self.db.execute("SELECT name, count, seconds FROM plays")}
        files = [f for f in self._files() if f[0] not in never]
        new = [f for f in files if f[0] not in known]
        if new:  # new arrivals: all a little below the usual count, so they come up more at first
            start = self._mode() - self.mode_offset
            for name, path, kind in new:
                secs = self.seconds_of(path) if kind == "clip" else 0.0
                self.db.execute("INSERT INTO plays (name, count, seconds, last) VALUES (?, ?, ?, NULL)",
                                (name, start, secs))
                known[name] = (start, secs)
            self.db.commit()
        return {name: (path, kind, known[name][1], known[name][0]) for name, path, kind in files}

    def _mode(self):
        row = self.db.execute("SELECT count FROM plays GROUP BY count "
                              "ORDER BY COUNT(*) DESC, count DESC LIMIT 1").fetchone()
        return row[0] if row else 0

    def count(self, name):
        row = self.db.execute("SELECT count FROM plays WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def _shown(self, name):
        self.db.execute("UPDATE plays SET count = count + 1, last = ? WHERE name = ?",
                        (time.strftime("%Y-%m-%d %H:%M:%S"), name))
        self.db.commit()

    # ---- choosing ----
    def _choose(self, start, room_bars, bpm, avoid=()):
        """Least shown first; among the least shown, prefer one that fits the room left."""
        pool = self.pool()
        names = [n for n in pool if n not in avoid] or list(pool)
        if not names:
            return None
        self.rng.shuffle(names)  # ties broken at random, then a stable sort by count
        names.sort(key=lambda n: pool[n][3])
        fewest = pool[names[0]][3]
        least = [n for n in names if pool[n][3] == fewest]
        fits = least if room_bars is None else [n for n in least if self._bars(pool[n], bpm) <= room_bars]
        name = (fits or least)[0]
        path, kind, _, _ = pool[name]
        bars = self._bars(pool[name], bpm)
        if room_bars is not None:
            bars = max(1, min(bars, room_bars))  # the last one in a gap is cut to fit
        return Pick(name, path, kind, bars, start, start + 4 * bars)

    def _bars(self, entry, bpm):
        _, kind, secs, _ = entry
        return self.image_bars if kind == "image" else bars_for(secs, bpm, self.max_clip_bars)

    def _bar_after(self, beat):
        """The next bar line (counted from the shuffle note) at least 2 beats away."""
        k = math.ceil((beat - self.origin) / 4 - 1e-9)
        line = self.origin + 4 * k
        return line if line - beat >= 2 else line + 4

    def _room(self, at):
        if self.gap_end is None:
            return None
        return max(0, int(round((self.gap_end - at) / 4)))

    def _plan_next(self, bpm):
        cur = self.current
        room = self._room(cur.end)
        self.upcoming = None if room == 0 else self._choose(cur.end, room, bpm, avoid={cur.name})

    # ---- the show calls these ----
    def start(self, beat, gap_bars, bpm):
        """A shuffle note: fill gap_bars from this beat (None or 0: until the next scene)."""
        self.stop()
        self.origin = beat
        self.gap_end = beat + 4 * gap_bars if gap_bars else None
        self.current = self._choose(beat, self._room(beat), bpm)
        self.skipped = False
        if self.current:
            self._plan_next(bpm)
        return self.current

    def update(self, beat, bpm):
        """Call every frame. Returns a new Pick when it's time to change clips, else None.
        When the gap is used up the last clip keeps going until the next scene note."""
        cur = self.current
        if cur is None or beat < cur.end or self.upcoming is None:
            return None
        if not self.skipped:
            self._shown(cur.name)
        self.current, self.skipped = self.upcoming, False
        self._plan_next(bpm)
        return self.current

    def skip(self, beat, bpm):
        """Next: this clip isn't working, move on now. It doesn't count as shown."""
        cur = self.current
        if cur is None:
            return None
        first = self._bar_after(beat)  # hand over on a bar line so later changes stay on the beat
        room = self._room(first)
        room = None if self.gap_end is None else room + 1
        nxt = self._choose(beat, room, bpm, avoid={cur.name})
        if nxt is None:
            return None
        nxt.end = first + 4 * (nxt.bars - 1)
        self.current, self.skipped = nxt, False
        self._plan_next(bpm)
        return nxt

    def never_again(self, beat, bpm):
        """Drop this clip from the shuffle for good, and move on."""
        cur = self.current
        if cur is None:
            return None
        self.db.execute("INSERT OR REPLACE INTO never (name, since) VALUES (?, ?)",
                        (cur.name, time.strftime("%Y-%m-%d %H:%M:%S")))
        self.db.commit()
        self.skipped = True
        return self.skip(beat, bpm)

    def stop(self):
        """A scene note took over: the clip on screen counts as shown (unless it was skipped)."""
        if self.current is not None and not self.skipped:
            self._shown(self.current.name)
        self.current = self.upcoming = self.gap_end = self.origin = None
        self.skipped = False

    @property
    def active(self):
        return self.current is not None

    def close(self):
        self.db.close()
