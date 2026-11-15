"""midi-watch: print incoming MIDI with timing; clock summarized per beat.

Usage:
  midi-watch                 list input ports
  midi-watch "Port 1"        listen on the first port whose name contains "Port 1"
  midi-watch MPC --raw-clock print every clock pulse too
"""
import argparse
import statistics
import time

import mido

from .common import find_port, input_names

PPQN = 24
BEATS_PER_BAR = 4
WINDOW = PPQN  # clock intervals used for the tempo/jitter readout


def tempo_text(intervals):
    if len(intervals) < 2:
        return "(measuring tempo...)"
    mean = statistics.mean(intervals)
    bpm = 60.0 / (mean * PPQN)
    jitter_ms = statistics.pstdev(intervals) * 1000
    return f"{bpm:6.2f} BPM   jitter {jitter_ms:.3f} ms"


def line(ms, msg, text=None):
    hexb = " ".join(f"{b:02X}" for b in msg.bytes()) if msg is not None else ""
    print(f"{ms:10.2f} ms  {hexb:<8}  {text if text is not None else msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("port", nargs="?", help="substring of the input port name")
    ap.add_argument("--raw-clock", action="store_true", help="print every clock pulse")
    args = ap.parse_args()

    names = input_names()
    if not args.port:
        print("Input ports:")
        for n in names:
            print("  ", n)
        return

    name = find_port(args.port, names, "input")
    t0 = time.perf_counter()
    last_clock, intervals, pulse = None, [], 0
    print(f"Listening on: {name}   (Ctrl+C to quit)\n")

    with mido.open_input(name) as port:
        try:
            for msg in port:
                now = time.perf_counter()
                ms = (now - t0) * 1000.0

                if msg.type == "clock":
                    if last_clock is not None:
                        intervals.append(now - last_clock)
                        del intervals[:-WINDOW]
                    last_clock = now
                    if args.raw_clock:
                        line(ms, msg)
                    if pulse % PPQN == 0:
                        beat_index = pulse // PPQN
                        bar = beat_index // BEATS_PER_BAR + 1
                        beat = beat_index % BEATS_PER_BAR + 1
                        print(f"{ms:10.2f} ms  --- bar {bar:3d} beat {beat}  "
                              f"{tempo_text(intervals)}", flush=True)
                    pulse += 1
                    continue

                if msg.type == "start":
                    pulse, intervals, last_clock = 0, [], None
                elif msg.type == "songpos":
                    # Song position is in 16th notes; 6 clocks per 16th.
                    pulse = msg.pos * 6
                line(ms, msg)
        except KeyboardInterrupt:
            print("\nDone.")


if __name__ == "__main__":
    main()
