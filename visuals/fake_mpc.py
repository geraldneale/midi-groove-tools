"""Stand-in for the MPC when rehearsing visuals alone.

Opens a virtual ALSA MIDI port called FakeMPC and plays a four on the floor
groove: clock, kick on every beat, snare on 2 and 4, 8th note hats (drum
channel), a new scene note every few bars and a slow hue sweep (visual channel).

    python3 fake_mpc.py --bpm 124 --bars 4 --notes 36 37 40 41
    python3 fake_mpc.py --bars 16 --notes 39 40:32 41:32 44    # fractals get 32 bars
    python3 fake_mpc.py --bars 16 --notes 39 60:32 40:32       # 32 bars of shuffle (clips and art)
    python3 vis_render.py show.toml     # with port = "FakeMPC" in [midi]

A note can carry its own length in bars after a colon; others use --bars.
It stops by itself after --minutes (default 500; 0 runs until Ctrl+C), so it can't be left
running for days.
"""
import argparse
import time

CLOCK, START, STOP = 0xF8, 0xFA, 0xFC


def stop_tick(minutes, bpm):
    """Clock tick to stop at after this many minutes, or None to run until Ctrl+C."""
    return None if minutes <= 0 else int(minutes * bpm * 24)


def scene_plan(notes, default_bars):
    """['39', '40:32'] -> [(39, default_bars), (40, 32)]."""
    plan = []
    for word in notes:
        note, _, bars = str(word).partition(":")
        n, b = int(note), int(bars) if bars else default_bars
        if not 0 <= n <= 127 or b < 1:
            raise ValueError(f"{word}: note must be 0 to 127 and bars at least 1")
        plan.append((n, b))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpm", type=float, default=124.0)
    ap.add_argument("--bars", type=int, default=4, help="bars per scene")
    ap.add_argument("--notes", nargs="+", default=["36", "37", "38", "39", "40", "41"],
                    help="scene notes in order; note:bars gives one its own length, e.g. 40:32")
    ap.add_argument("--shuffle-note", type=int, default=60,
                    help="the shuffle note; its velocity carries the gap length in bars, like on the MPC")
    ap.add_argument("--minutes", type=float, default=500.0,
                    help="stop by itself after this long (default 500; 0 runs until Ctrl+C)")
    ap.add_argument("--drum-ch", type=int, default=10)
    ap.add_argument("--visual-ch", type=int, default=16)
    a = ap.parse_args()
    try:
        plan = scene_plan(a.notes, a.bars)
    except ValueError as e:
        ap.error(str(e))
    import rtmidi

    on_d, off_d = 0x90 | (a.drum_ch - 1), 0x80 | (a.drum_ch - 1)
    on_v, off_v = 0x90 | (a.visual_ch - 1), 0x80 | (a.visual_ch - 1)
    cc_v = 0xB0 | (a.visual_ch - 1)

    out = rtmidi.MidiOut()
    out.open_virtual_port("FakeMPC")
    print("FakeMPC port open; starting in 3 s so the visuals app can find it")
    time.sleep(3)

    tick_s = 60.0 / (a.bpm * 24)
    pending = {}  # tick -> list of note offs
    scene_note, step, next_change = None, -1, 0
    last_tick = stop_tick(a.minutes, a.bpm)
    if last_tick:
        print(f"stops by itself after {a.minutes:g} minutes (--minutes 0 to run until Ctrl+C)")
    out.send_message([START])
    t0 = time.perf_counter()
    tick = 0
    try:
        while last_tick is None or tick < last_tick:
            delay = t0 + tick * tick_s - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            out.send_message([CLOCK])
            for msg in pending.pop(tick, []):
                out.send_message(msg)

            if tick % 12 == 0:  # 8th notes
                beat, eighth = divmod(tick // 12, 2)
                hits = [42]
                if eighth == 0:
                    hits.append(36)
                    if beat % 4 in (1, 3):
                        hits.append(38)
                    out.send_message([cc_v, 2, (beat % 96) * 127 // 96])  # hue sweep over 24 bars
                    if beat == next_change:
                        if scene_note is not None:
                            out.send_message([off_v, scene_note, 0])
                        step = (step + 1) % len(plan)
                        scene_note, bars = plan[step]
                        next_change = beat + 4 * bars
                        vel = min(127, bars) if scene_note == a.shuffle_note else 110  # shuffle: velocity = bars
                        out.send_message([on_v, scene_note, vel])
                        print(f"bar {beat // 4 + 1}: scene note {scene_note} for {bars} bars")
                for n in hits:
                    out.send_message([on_d, n, 120 if n != 42 else 70])
                    pending.setdefault(tick + 3, []).append([off_d, n, 0])
            tick += 1
    except KeyboardInterrupt:
        pass
    else:
        print(f"{a.minutes:g} minutes up: stopped")
    finally:
        out.send_message([STOP])  # a clean stop either way: time up or Ctrl+C
        if scene_note is not None:
            out.send_message([off_v, scene_note, 0])
        del out


if __name__ == "__main__":
    main()
