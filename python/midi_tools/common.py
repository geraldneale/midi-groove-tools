"""Shared helpers: port lookup by substring."""
import sys

import mido


def find_port(pattern, names, kind):
    """Return the first port name containing pattern (case-insensitive), or exit."""
    matches = [n for n in names if pattern.lower() in n.lower()]
    if not matches:
        sys.exit(f"No {kind} port matching '{pattern}'. Available: {names}")
    return matches[0]


def input_names():
    return mido.get_input_names()


def output_names():
    return mido.get_output_names()
