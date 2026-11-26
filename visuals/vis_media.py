"""Image and video sources for the visuals engine.

Each source exposes .frame, an RGB uint8 numpy array flipped bottom row first
(OpenGL texture order), and update(dt, rate) which returns True when the
frame changed and needs uploading.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:  # clips are optional
    cv2 = None

MAX_IMAGE_SIDE = 4096  # keeps big GIMP exports inside GPU texture limits


def is_colorless(frame, limit=12, share=98):
    """True only for black and white pictures: 98% of pixels are essentially gray (their
    strongest and weakest color channels within 12 of 255). Faint sepia or muted art keeps
    its own colors; a small colored signature doesn't stop a black and white piece counting."""
    f = np.asarray(frame)
    if f.ndim != 3 or f.size == 0:
        return False
    step = max(1, int((f.shape[0] * f.shape[1] / 40000) ** 0.5))  # a sample is plenty
    s = f[::step, ::step].astype(np.int16)
    chroma = s.max(axis=2) - s.min(axis=2)
    return float(np.percentile(chroma, share)) <= limit


class ArraySource:
    is_video = False

    def __init__(self, rgb):
        self.frame = np.ascontiguousarray(np.flipud(np.asarray(rgb, dtype=np.uint8)))
        self._changed = True

    @classmethod
    def from_frame(cls, frame):
        """Wrap an already flipped frame (e.g. from a preload cache) without copying."""
        src = cls.__new__(cls)
        src.frame, src._changed = frame, True
        return src

    @classmethod
    def solid(cls, rgb=(0, 0, 0), size=(2, 2)):
        return cls(np.full((size[1], size[0], 3), rgb, dtype=np.uint8))

    def update(self, dt, rate=1.0):
        changed, self._changed = self._changed, False
        return changed

    def close(self):
        pass


class ImageSource(ArraySource):
    def __init__(self, path, max_side=MAX_IMAGE_SIDE):
        # follow the file's rotation tag (phone photos and some editors store the picture turned and
        # say so in a tag), so pieces come up the way Finder and Preview show them
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)  # only ever shrinks
        super().__init__(np.asarray(img))


class ClipSource:
    """Loops a section of a video file. Playback speed follows `rate` (tempo stretch).

    Clips transcoded all keyframe (ffmpeg -g 1) seek exactly and loop cleanly.
    """
    is_video = True

    def __init__(self, path, start=0.0, end=None, max_side=1920):
        if cv2 is None:
            raise RuntimeError("opencv-python is needed for video clips")
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = n / self.fps if n > 0 else None
        self.start = max(0.0, float(start))
        self.end = float(end) if end is not None else duration
        if self.end is not None and self.end <= self.start:
            raise ValueError(f"clip end {self.end} is not after start {self.start}")
        self.max_side = max_side
        self.pos = self.start
        self._seek(self.start)
        if not self._read():
            raise RuntimeError(f"no frames at {self.start}s in {path}")
        self._changed = True

    def _seek(self, t):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        self._next_t = t  # source time of the next frame read() returns

    def _read(self):
        ok, bgr = self.cap.read()
        if not ok:
            return False
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(w, h) > self.max_side:
            k = self.max_side / max(w, h)
            rgb = cv2.resize(rgb, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        self.frame = np.ascontiguousarray(np.flipud(rgb))
        self._next_t += 1.0 / self.fps
        return True

    def _restart(self):
        self.pos = self.start
        self._seek(self.start)
        self._read()

    def update(self, dt, rate=1.0):
        changed, self._changed = self._changed, False
        self.pos += dt * rate
        if self.end is not None and self.pos >= self.end:
            length = self.end - self.start
            self.pos = self.start + (self.pos - self.end) % length
            self._seek(self.start)
            self._read()
            changed = True
        if self.pos - self._next_t > 0.5:  # far behind (high rate): jump instead of reading every frame
            self._seek(self.pos)
        while self._next_t <= self.pos + 1e-9:
            if not self._read():  # file ended before `end`
                self._restart()
                return True
            changed = True
        return changed

    def close(self):
        self.cap.release()


class ClipPool:
    """Opens video clips on a background thread ahead of time, so triggering one never
    stalls the picture while the file opens and the first frame decodes.

    prepare(key, factory) starts opening a clip; take(key, factory) hands over the ready one
    (or opens it right away if it wasn't prepared) and starts preparing a fresh copy, so the
    same note can be triggered again straight away.
    """

    def __init__(self, workers=2):
        from concurrent.futures import ThreadPoolExecutor
        self._ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="clip")
        self._ready = {}
        self._factories = {}

    def prepare(self, key, factory):
        self._discard(self._ready.pop(key, None))
        self._factories[key] = factory
        self._ready[key] = self._ex.submit(factory)

    def take(self, key, factory=None):
        factory = factory or self._factories.get(key)
        fut = self._ready.pop(key, None)
        src = fut.result() if fut is not None else factory()  # result() raises if opening failed
        self.prepare(key, factory)
        return src

    def clear(self):
        """Forget everything prepared (e.g. on a song change) and close the files."""
        for fut in self._ready.values():
            self._discard(fut)
        self._ready.clear()
        self._factories.clear()

    @staticmethod
    def _discard(fut):
        if fut is None:
            return

        def close(f):
            if not f.cancelled() and f.exception() is None:
                f.result().close()
        fut.add_done_callback(close)

    def shutdown(self):
        self.clear()
        self._ex.shutdown(wait=False, cancel_futures=True)
