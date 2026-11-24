"""Stand-in for the MPC when rehearsing visuals alone.

Opens a virtual ALSA MIDI port called FakeMPC and plays a four on the floor
groove: clock, kick on every beat, snare on 2 and 4, 8th note hats (drum
channel), a new scene note every few bars and a slow hue sweep (visual channel).

    python3 fake_mpc.py --bpm 124 --bars 4 --notes 36 37 40 41
    python3 vis_render.py show.toml     # with port = "FakeMPC" in [midi]
"""
import argparse
import time

import rtmidi

CLOCK, START, STOP = 0xF8, 0xFA, 0xFC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpm", type=float, default=124.0)
    ap.add_argument("--bars", type=int, default=4, help="bars per scene")
    ap.add_argument("--notes", type=int, nargs="+", default=[36, 37, 38, 39, 40, 41])
    ap.add_argument("--drum-ch", type=int, default=10)
    ap.add_argument("--visual-ch", type=int, default=16)
    a = ap.parse_args()

    on_d, off_d = 0x90 | (a.drum_ch - 1), 0x80 | (a.drum_ch - 1)
    on_v, off_v = 0x90 | (a.visual_ch - 1), 0x80 | (a.visual_ch - 1)
    cc_v = 0xB0 | (a.visual_ch - 1)

    out = rtmidi.MidiOut()
    out.open_virtual_port("FakeMPC")
    print("FakeMPC port open; starting in 3 s so the visuals app can find it")
    time.sleep(3)

    tick_s = 60.0 / (a.bpm * 24)
    pending = {}  # tick -> list of note offs
    scene_note = None
    out.send_message([START])
    t0 = time.perf_counter()
    tick = 0
    try:
        while True:
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
                    out.send_message([cc_v, 2, (beat % 64) * 2])  # hue sweep over 16 bars
                    if beat % (4 * a.bars) == 0:
                        if scene_note is not None:
                            out.send_message([off_v, scene_note, 0])
                        scene_note = a.notes[(beat // (4 * a.bars)) % len(a.notes)]
                        out.send_message([on_v, scene_note, 110])
                        print(f"bar {beat // 4 + 1}: scene note {scene_note}")
                for n in hits:
                    out.send_message([on_d, n, 120 if n != 42 else 70])
                    pending.setdefault(tick + 3, []).append([off_d, n, 0])
            tick += 1
    except KeyboardInterrupt:
        out.send_message([STOP])
        if scene_note is not None:
            out.send_message([off_v, scene_note, 0])
    finally:
        del out


if __name__ == "__main__":
    main()
