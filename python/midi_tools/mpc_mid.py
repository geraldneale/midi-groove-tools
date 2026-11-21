#!/usr/bin/env python3
"""mpc-mid: rewrite a Logic .mid in the layout the MPC uses for its own patterns.

What the MPC writes (from its Clip as MIDI Pattern export): format 1, 1 track, 960 PPQN.
What Logic wrote: format 0, 1 track, 480 PPQN.

This script changes only the header and multiplies every delta time by an exact whole
number (480 -> 960 is x2). No quantizing, no note changes. It then reads both files
back and checks every note matches on pitch, velocity, channel and time.

  mpc-mid HungUp_SynthBass.mid HungUp_SynthBass_960.mid
"""
import struct
import sys

TARGET_PPQN = 960


def read_vlq(b, i):
    v = 0
    while True:
        c = b[i]
        i += 1
        v = (v << 7) | (c & 0x7F)
        if not c & 0x80:
            return v, i


def write_vlq(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def parse(path):
    """Return (fmt, ppqn, tracks); each track is a list of (delta, raw_event_bytes)."""
    b = open(path, "rb").read()
    if b[:4] != b"MThd":
        sys.exit(f"{path}: not a MIDI file")
    hlen = struct.unpack(">I", b[4:8])[0]
    fmt, ntrk, div = struct.unpack(">HHH", b[8:14])
    if div & 0x8000:
        sys.exit(f"{path}: SMPTE timing, not supported")
    i = 8 + hlen
    tracks = []
    for _ in range(ntrk):
        if b[i:i + 4] != b"MTrk":
            sys.exit(f"{path}: bad track chunk")
        tlen = struct.unpack(">I", b[i + 4:i + 8])[0]
        j, end = i + 8, i + 8 + tlen
        events, status = [], None
        while j < end:
            delta, j = read_vlq(b, j)
            start = j
            c = b[j]
            if c == 0xFF:  # meta
                ln, k = read_vlq(b, j + 2)
                j = k + ln
            elif c in (0xF0, 0xF7):  # sysex
                ln, k = read_vlq(b, j + 1)
                j = k + ln
            else:
                if c & 0x80:
                    status = c
                    j += 1
                    raw_status = b""
                else:  # running status: store the status byte explicitly
                    raw_status = bytes([status])
                n = 1 if (status & 0xF0) in (0xC0, 0xD0) else 2
                j += n
                events.append((delta, raw_status + b[start:j]))
                continue
            events.append((delta, b[start:j]))
        tracks.append(events)
        i = end
    return fmt, div, tracks


def notes(ppqn, tracks):
    """Absolute note list, time in quarter-note units scaled to 3840ths so files compare."""
    out = []
    scale = 3840 // ppqn
    for events in tracks:
        t = 0
        for delta, ev in events:
            t += delta
            s = ev[0]
            if s & 0xF0 in (0x80, 0x90):
                kind = "on" if (s & 0xF0) == 0x90 and ev[2] > 0 else "off"
                out.append((t * scale, kind, s & 0x0F, ev[1], ev[2] if kind == "on" else 0))
    return sorted(out)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    fmt, ppqn, tracks = parse(src)
    if TARGET_PPQN % ppqn:
        sys.exit(f"{ppqn} PPQN does not divide evenly into {TARGET_PPQN}; refusing (would round)")
    k = TARGET_PPQN // ppqn
    events = [e for tr in tracks for e in tr] if len(tracks) == 1 else None
    if events is None:
        sys.exit("expected a single-track file")
    body = b"".join(write_vlq(d * k) + ev for d, ev in events)
    with open(dst, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, 1, TARGET_PPQN))
        f.write(b"MTrk" + struct.pack(">I", len(body)) + body)

    a = notes(ppqn, tracks)
    fmt2, ppqn2, tracks2 = parse(dst)
    b = notes(ppqn2, tracks2)
    ons = sum(1 for n in a if n[1] == "on")
    print(f"in : format {fmt}, {ppqn} PPQN, {len(events)} events, {ons} notes")
    print(f"out: format {fmt2}, {ppqn2} PPQN, ticks x{k}")
    if a == b:
        first = next(n for n in a if n[1] == "on")
        bar = first[0] / 3840 / 4 + 1  # assumes 4/4
        print(f"MATCH: all {ons} notes identical (pitch, velocity, channel, time)")
        print(f"first note at bar {bar:.4f}")
    else:
        sys.exit("MISMATCH: do not use the output")


if __name__ == "__main__":
    main()
