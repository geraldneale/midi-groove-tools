"""Turn a YouTube playlist into show-ready video clips.

Step 1, once per playlist: list every video in a plain text file.
    python3 clip_grab.py import "https://www.youtube.com/playlist?list=..." clips.txt

Step 2, edit clips.txt. One video per line; times are optional:
    https://youtu.be/abc123                      # whole video
    https://youtu.be/def456  1:23  1:41          # just that segment
    https://youtu.be/def456  3:05  3:09  song=0 note=42 name=spin   # tag it for the show
Lines starting with # are comments. Running import again adds only new videos.

Step 3, download and cut:
    python3 clip_grab.py fetch clips.txt
Each line becomes clips/<name>.mp4: no audio, at most 1080p, 30 fps, every frame a
keyframe so it loops and jumps exactly. Clips already made are skipped, and the file
name holds the times, so changing a line's times makes a new clip. Lines tagged with
note= are printed as show.toml entries to paste in; show.toml itself is never touched.

Needs yt-dlp and ffmpeg (Mac: brew install yt-dlp ffmpeg). Only use footage you have
the right to show publicly.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_FORMAT = "bv*[height<=1080][ext=mp4]/bv*[height<=1080]/b[height<=1080]/bv*/b"
FFMPEG_ARGS = ["-an", "-vf", "scale=w=1920:h=1080:force_original_aspect_ratio=decrease:force_divisible_by=2,fps=30",
               "-c:v", "libx264", "-g", "1", "-crf", "18", "-preset", "medium",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


# ---- parsing ----
def parse_time(text):
    """'83', '83.5', '1:23', '1:02:03' -> seconds."""
    parts = text.split(":")
    if not 1 <= len(parts) <= 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise ValueError(f"not a time: {text!r} (use 1:23 or 83)")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def fmt_time(secs, sep=":"):
    """Seconds -> '1:23' (or '1m23s' for file names), keeping tenths when present."""
    m, s = divmod(round(secs, 1), 60)
    h, m = divmod(int(m), 60)
    s_txt = f"{s:04.1f}".rstrip("0").rstrip(".") if s % 1 else f"{int(s):02d}"
    if sep == ":":
        return f"{h}:{m:02d}:{s_txt}" if h else f"{m}:{s_txt}"
    return (f"{h}h" if h else "") + f"{m}m{s_txt}s"


def video_id(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", url)[-40:].strip("_") or "clip"


@dataclass
class Clip:
    line_no: int
    url: str
    start: float | None = None
    end: float | None = None
    tags: dict = field(default_factory=dict)

    @property
    def whole(self):
        return self.start is None

    def file_name(self):
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", self.tags.get("name", "")) or video_id(self.url)
        span = "whole" if self.whole else f"{fmt_time(self.start, 'm')}-{fmt_time(self.end, 'm')}"
        return f"{base}_{span}.mp4"


def parse_line(line, line_no=0):
    """One clips.txt line -> Clip, or None for blanks and comments. Raises ValueError on mistakes."""
    text = line.split("#", 1)[0].strip()
    if not text:
        return None
    words = text.split()
    url, rest = words[0], words[1:]
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"line {line_no}: should start with a video link")
    times = [w for w in rest if "=" not in w]
    tags = dict(w.split("=", 1) for w in rest if "=" in w)
    unknown = set(tags) - {"song", "note", "name", "bpm", "fit"}
    if unknown:
        raise ValueError(f"line {line_no}: unknown tag {', '.join(sorted(unknown))} (use song, note, name, bpm, fit)")
    if len(times) not in (0, 2):
        raise ValueError(f"line {line_no}: give both a start and an end time, or neither for the whole video")
    clip = Clip(line_no, url, tags=tags)
    if times:
        clip.start, clip.end = parse_time(times[0]), parse_time(times[1])
        if clip.end <= clip.start:
            raise ValueError(f"line {line_no}: end {times[1]} is not after start {times[0]}")
    for key in ("song", "note"):
        if key in tags and not tags[key].isdigit():
            raise ValueError(f"line {line_no}: {key}= needs a number")
    if "note" in tags and not 0 <= int(tags["note"]) <= 127:
        raise ValueError(f"line {line_no}: note= must be 0 to 127")
    return clip


def read_list(path):
    clips, errors = [], []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            c = parse_line(line, n)
        except ValueError as e:
            errors.append(str(e))
            continue
        if c:
            clips.append(c)
    return clips, errors


# ---- import ----
HEADER = """# Clips for the visuals. One video per line:  link  [start end]  [song=N note=N name=word]
# No times = the whole video. Times like 1:23 or 83. Lines starting with # are ignored.
# Then: python3 clip_grab.py fetch clips.txt
"""


def merge_playlist(existing_text, entries):
    """Add a line for each playlist video not already in the list. Returns (new text, number added)."""
    have = set(re.findall(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{6,})", existing_text))
    out = existing_text if existing_text.strip() else HEADER
    if out and not out.endswith("\n"):
        out += "\n"
    added = 0
    for e in entries:
        vid = e.get("id")
        if not vid or vid in have:
            continue
        title = (e.get("title") or "").replace("\n", " ").strip()
        dur = e.get("duration")
        note = f"  ({fmt_time(dur)})" if dur else ""
        out += f"https://www.youtube.com/watch?v={vid}  # {title}{note}\n"
        have.add(vid)
        added += 1
    return out, added


def cmd_import(args):
    res = subprocess.run(["yt-dlp", "--flat-playlist", "-J", args.playlist], capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"yt-dlp could not read the playlist:\n{res.stderr.strip()}")
    entries = json.loads(res.stdout).get("entries") or []
    path = Path(args.list)
    text, added = merge_playlist(path.read_text() if path.exists() else "", entries)
    path.write_text(text)
    print(f"{path}: added {added} of {len(entries)} videos. Add start and end times to any you want trimmed,")
    print(f"then: python3 clip_grab.py fetch {path}")


# ---- fetch ----
def download_command(clip, dest_stem):
    cmd = ["yt-dlp", "-q", "--no-warnings", "--no-playlist", "-f", VIDEO_FORMAT,
           "-o", f"{dest_stem}.%(ext)s"]
    if not clip.whole:
        cmd += ["--download-sections", f"*{clip.start}-{clip.end}", "--force-keyframes-at-cuts"]
    return cmd + [clip.url]


def convert_command(src, out):
    return ["ffmpeg", "-v", "error", "-y", "-i", str(src)] + FFMPEG_ARGS + ["-f", "mp4", str(out)]


def show_entry(clip, out_dir):
    lines = [f"  [[song.item]]    # song={clip.tags.get('song', '?')}, from clips.txt line {clip.line_no}",
             f"  note = {clip.tags['note']}",
             '  type = "clip"',
             f'  path = "{out_dir.name}/{clip.file_name()}"']
    if "bpm" in clip.tags:
        lines.append(f"  bpm = {clip.tags['bpm']}")
    if "fit" in clip.tags:
        lines.append(f'  fit = "{clip.tags["fit"]}"')
    return "\n".join(lines)


def fetch(clips, out_dir, redo=False, dry_run=False, run=subprocess.run):
    out_dir.mkdir(parents=True, exist_ok=True)
    made = skipped = failed = 0
    for clip in clips:
        out = out_dir / clip.file_name()
        span = "whole video" if clip.whole else f"{fmt_time(clip.start)} to {fmt_time(clip.end)}"
        if out.exists() and not redo:
            skipped += 1
            continue
        print(f"line {clip.line_no}: {span} -> {out}")
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "src"
            dl = download_command(clip, stem)
            if dry_run:
                print("   ", " ".join(dl))
                print("   ", " ".join(convert_command(str(stem) + ".<ext>", out)))
                continue
            r = run(dl, capture_output=True, text=True)
            got = sorted(Path(tmp).glob("src.*"))
            if r.returncode != 0 or not got:
                why = (r.stderr or "").strip().splitlines()
                print(f"    download failed: {why[-1] if why else 'no file came back'}")
                failed += 1
                continue
            part = out.with_suffix(".part.mp4")
            r = run(convert_command(got[0], part), capture_output=True, text=True)
            if r.returncode != 0:
                print(f"    convert failed: {(r.stderr or '').strip()[-300:]}")
                part.unlink(missing_ok=True)
                failed += 1
                continue
            part.replace(out)  # only a finished clip gets its real name
            made += 1
    wanted = {c.file_name() for c in clips}
    stale = sorted(p.name for p in out_dir.glob("*.mp4") if p.name not in wanted and ".part" not in p.name)
    return made, skipped, failed, stale


def cmd_fetch(args):
    for tool in ("yt-dlp", "ffmpeg"):
        if not args.dry_run and shutil.which(tool) is None:
            sys.exit(f"{tool} is not installed. On a Mac: brew install yt-dlp ffmpeg")
    clips, errors = read_list(args.list)
    for e in errors:
        print("skipping", e)
    out_dir = Path(args.out)
    made, skipped, failed, stale = fetch(clips, out_dir, args.redo, args.dry_run)
    print(f"\n{made} made, {skipped} already there, {failed} failed.")
    if stale:
        print(f"In {out_dir}/ but no longer in the list (left alone): {', '.join(stale)}")
    tagged = [c for c in clips if "note" in c.tags]
    if tagged:
        print("\nPaste into show.toml under the right [[song]]:\n")
        print("\n\n".join(show_entry(c, out_dir) for c in tagged))
    if failed or errors:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import", help="add a playlist's videos to the list")
    p.add_argument("playlist")
    p.add_argument("list", nargs="?", default="clips.txt")
    p.set_defaults(func=cmd_import)
    p = sub.add_parser("fetch", help="download and cut every clip in the list")
    p.add_argument("list", nargs="?", default="clips.txt")
    p.add_argument("--out", default="clips", help="folder for the clips (default clips)")
    p.add_argument("--redo", action="store_true", help="remake clips that already exist")
    p.add_argument("--dry-run", action="store_true", help="show what would run, change nothing")
    p.set_defaults(func=cmd_fetch)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
