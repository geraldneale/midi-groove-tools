"""midi-watch-all: listen on every input port at once and show which ports receive what.

Usage:
  midi-watch-all          all input ports
  midi-watch-all H12      only ports whose name contains "H12"

Prints the first message of each type per port (every non-clock message),
then a count per port and type on Ctrl+C.
"""
import argparse

import mido

from .common import input_names


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("filter", nargs="?", default="", help="substring of port names to include")
    args = ap.parse_args()

    names = [n for n in input_names() if args.filter.lower() in n.lower()]
    if not names:
        raise SystemExit(f"No input ports matching '{args.filter}'.")
    ports = [mido.open_input(n) for n in names]
    print(f"Listening on {len(ports)} ports. Ctrl+C to quit.")

    seen = {}
    try:
        for port, msg in mido.ports.multi_receive(ports, yield_ports=True):
            key = (port.name, msg.type)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == 1 or msg.type != "clock":
                print(port.name, msg, flush=True)
    except KeyboardInterrupt:
        print()
        for (name, kind), n in sorted(seen.items()):
            print(f"{name}: {kind} x{n}")
    finally:
        for p in ports:
            p.close()


if __name__ == "__main__":
    main()
