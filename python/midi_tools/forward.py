"""midi-forward: pass MIDI from one port to another (e.g. MPC clock to an S-1 via a laptop).

Usage:
  midi-forward                         list input and output ports
  midi-forward "MPC" "S-1"             forward everything
  midi-forward "MPC" "S-1" --sync-only forward only clock, start, stop, continue, song position
"""
import argparse

import mido

from .common import find_port, input_names, output_names

SYNC_TYPES = {"clock", "start", "stop", "continue", "songpos"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", nargs="?", help="substring of the input port name")
    ap.add_argument("dst", nargs="?", help="substring of the output port name")
    ap.add_argument("--sync-only", action="store_true", help="forward clock and transport only")
    args = ap.parse_args()

    ins, outs = input_names(), output_names()
    if not (args.src and args.dst):
        print("Input ports:")
        for n in ins:
            print("  ", n)
        print("Output ports:")
        for n in outs:
            print("  ", n)
        return

    src = find_port(args.src, ins, "input")
    dst = find_port(args.dst, outs, "output")
    count = 0
    with mido.open_input(src) as i, mido.open_output(dst) as o:
        print(f"Forwarding {src} -> {dst}"
              f"{' (sync only)' if args.sync_only else ''}   (Ctrl+C to stop)")
        try:
            for msg in i:
                if args.sync_only and msg.type not in SYNC_TYPES:
                    continue
                o.send(msg)
                count += 1
        except KeyboardInterrupt:
            print(f"\nDone. {count} messages forwarded.")


if __name__ == "__main__":
    main()
