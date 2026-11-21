#!/usr/bin/env python3
"""mpc-pattern: make an MPC pattern (.mpcpattern) from a standard .mid file.

The MPC loads an outside .mid only as a whole new sequence, onto track 1. Only its
own pattern files load onto the track you choose (a plugin track, say). So keep
your .mid as the master copy and build the pattern from it whenever you need one;
never edit the pattern itself.

  mpc-pattern HungUp_SynthBass.mid
  mpc-pattern HungUp_SynthBass.mid HungUp_SynthBass.mpcpattern

Timing is kept exactly from the start of the file, so a bass that enters at bar 25
still enters at bar 25. The MPC counts 960 per beat; a file at 480, 960 or any
resolution that divides 960 converts without rounding (others are refused). Every
note is read back from the pattern and checked against the .mid. Notes only:
pitch bend, CCs and tempo are not carried over.
"""
import argparse, json, os, struct, sys

MPC_PPQ = 960


def read_vlq(b, i):
    v = 0
    while True:
        c = b[i]
        i += 1
        v = (v << 7) | (c & 0x7F)
        if not c & 0x80:
            return v, i


def read_notes(path):
    """(ppqn, [(start, length, channel, pitch, velocity)]) in the file's own ticks."""
    b = open(path, 'rb').read()
    if b[:4] != b'MThd':
        sys.exit(f'{path}: not a MIDI file')
    hlen = struct.unpack('>I', b[4:8])[0]
    _fmt, ntrk, div = struct.unpack('>HHH', b[8:14])
    if div & 0x8000:
        sys.exit(f'{path}: timecode-based timing is not supported')
    i, notes, hanging = 8 + hlen, [], 0
    for _ in range(ntrk):
        if b[i:i + 4] != b'MTrk':
            sys.exit(f'{path}: damaged track')
        end = i + 8 + struct.unpack('>I', b[i + 4:i + 8])[0]
        j, t, status, held = i + 8, 0, None, {}
        while j < end:
            d, j = read_vlq(b, j)
            t += d
            c = b[j]
            if c == 0xFF:
                ln, j = read_vlq(b, j + 2)
                j += ln
                continue
            if c in (0xF0, 0xF7):
                ln, j = read_vlq(b, j + 1)
                j += ln
                continue
            if c & 0x80:
                status, j = c, j + 1
            kind, ch = status & 0xF0, status & 0x0F
            n = 1 if kind in (0xC0, 0xD0) else 2
            data, j = b[j:j + n], j + n
            if kind == 0x90 and data[1] > 0:
                held.setdefault((ch, data[0]), []).append((t, data[1]))
            elif kind in (0x80, 0x90):
                q = held.get((ch, data[0]))
                if q:
                    s, v = q.pop(0)
                    notes.append((s, t - s, ch, data[0], v))
        hanging += sum(len(q) for q in held.values())
        i = end
    if hanging:
        print(f'warning: {hanging} notes never end in the file; left out')
    return div, sorted(notes)


def velocity_text(v):
    """MIDI velocity 0-127 as the MPC's 0-1 number."""
    s = repr(v / 127)
    return s[:-2] if s.endswith('.0') else s[:17]


def build(notes, k):
    nl = '\r\n'
    head = [(1, 0, '0.0'), (1, 32, '0.0'), (1, 130, '0.787401556968689')]  # as the MPC writes them
    ev = [(t, 0, 0, a, b, '0.0') for t, a, b in head]
    ev += [(2, s * k, ln * k, p, velocity_text(v), '0.5') for s, ln, _ch, p, v in notes]
    blocks = []
    for typ, time, ln, one, two, mv in ev:
        blocks.append(nl.join([
            '            {', f'                "type": {typ},', f'                "time": {time},',
            f'                "len": {ln},', f'                "1": {one},', f'                "2": {two},',
            '                "3": 0,', '                "mod": 0,', f'                "modVal": {mv}', '            }']))
    return nl.join(['{', '    "pattern": {', '        "length": 9223372036854775807,',
                    '        "events": [', (',' + nl).join(blocks), '        ]', '    }', '}']) + nl


def main():
    p = argparse.ArgumentParser(prog='mpc-pattern', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mid')
    p.add_argument('out', nargs='?', help='default: the .mid name with .mpcpattern')
    a = p.parse_args()
    out = a.out or os.path.splitext(a.mid)[0] + '.mpcpattern'

    ppqn, notes = read_notes(a.mid)
    if not notes:
        sys.exit('no notes in the file')
    if MPC_PPQ % ppqn:
        sys.exit(f'{ppqn} per beat does not divide evenly into {MPC_PPQ}; refusing (timing would be rounded)')
    k = MPC_PPQ // ppqn
    chans = sorted({n[2] + 1 for n in notes})
    if len(chans) > 1:
        print(f'note: notes on MIDI channels {chans}; a pattern has no channels, all go to one track')
    with open(out, 'w', newline='') as f:
        f.write(build(notes, k))

    back = [e for e in json.load(open(out))['pattern']['events'] if e['type'] == 2]
    want = [(s * k, ln * k, pt, v) for s, ln, _c, pt, v in notes]
    got = [(e['time'], e['len'], e['1'], round(float(e['2']) * 127)) for e in back]
    print(f'{a.mid}: {len(notes)} notes, {ppqn} per beat (x{k} to {MPC_PPQ})')
    if got != want:
        sys.exit('MISMATCH: do not use the pattern')
    bar = notes[0][0] / ppqn / 4 + 1
    print(f'MATCH: all {len(notes)} notes identical (time, length, pitch, velocity)')
    print(f'first note at bar {bar:.4f} (counting 4/4)')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
