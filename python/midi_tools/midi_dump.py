#!/usr/bin/env python3
"""midi_dump.py: show what's in a .mid file, bar by bar. No installs needed.

  python3 midi_dump.py song.mid                 # every note: bar, step, pitch, velocity, length
  python3 midi_dump.py song.mid --bars 41-44    # just those bars
  python3 midi_dump.py song.mid --grid --bars 41-44   # 1 e + a grid, one row per pitch

Bar numbers match Logic. Note names use Logic's convention (middle C = C3).
"""
import argparse, struct, sys
from collections import defaultdict

NAMES = 'C C# D D# E F F# G G# A A# B'.split()
SUBS = ['', 'e', '+', 'a']


def vlq(d, i):
    v = 0
    while True:
        b = d[i]; i += 1; v = (v << 7) | (b & 0x7F)
        if b < 0x80:
            return v, i


def read(path):
    d = open(path, 'rb').read()
    if d[:4] != b'MThd':
        sys.exit('not a MIDI file')
    hlen = struct.unpack('>I', d[4:8])[0]
    _, ntrk, tpb = struct.unpack('>HHH', d[8:14])
    p, notes, bpm = 8 + hlen, [], None
    for _ in range(ntrk):
        ln = struct.unpack('>I', d[p + 4:p + 8])[0]
        i, end, p = p + 8, p + 8 + ln, p + 8 + ln
        t, st, on = 0, None, {}
        while i < end:
            dt, i = vlq(d, i); t += dt; b = d[i]
            if b == 0xFF:
                typ = d[i + 1]; l, j = vlq(d, i + 2)
                if typ == 0x51 and bpm is None:
                    bpm = 60e6 / int.from_bytes(d[j:j + l], 'big')
                i = j + l; continue
            if b in (0xF0, 0xF7):
                l, j = vlq(d, i + 1); i = j + l; continue
            if b & 0x80:
                st = b; i += 1
            hi, ch = st & 0xF0, st & 0x0F
            if hi in (0xC0, 0xD0):
                i += 1; continue
            a, c = d[i], d[i + 1]; i += 2
            if hi == 0x90 and c:
                on.setdefault((ch, a), []).append((t, c))
            elif hi in (0x80, 0x90) and on.get((ch, a)):
                s, v = on[(ch, a)].pop(0)
                notes.append((s, t - s, a, v, ch + 1))
    return tpb, sorted(notes), bpm or 120.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file')
    ap.add_argument('--bars', help='range like 41-44, or a single bar')
    ap.add_argument('--grid', action='store_true', help='show a 1 e + a grid per bar')
    a = ap.parse_args()

    tpb, notes, bpm = read(a.file)
    bar, s16 = tpb * 4, tpb / 4
    lo, hi = 1, 10 ** 9
    if a.bars:
        lo, _, h = a.bars.partition('-'); lo = int(lo); hi = int(h or lo)
    name = lambda n: f'{NAMES[n % 12]}{n // 12 - 2}'
    print(f'{a.file}: {len(notes)} notes, {bpm:.2f} BPM, channels {sorted({n[4] for n in notes})}')

    bybar = defaultdict(list)
    for s, d, n, v, ch in notes:
        step = round(s / s16)            # nearest 16th, so a note played a hair early
        b = step // 16 + 1               # still counts in the bar it belongs to
        if lo <= b <= hi:
            bybar[b].append((step % 16 + (s / s16 - step), d / s16, n, v, ch))

    for b in sorted(bybar):
        if a.grid:
            print(f'\nbar {b}')
            print('       1 e + a 2 e + a 3 e + a 4 e + a')
            for p in sorted({x[2] for x in bybar[b]}, reverse=True):
                row = ['.'] * 16
                for st, _, n, _, _ in bybar[b]:
                    if n == p:
                        row[round(st) % 16] = 'X'
                print(f'{name(p):<6} ' + ' '.join(row))
        else:
            for st, ln, n, v, ch in bybar[b]:
                step = round(st)
                off = st - step
                pos = f'{step // 4 + 1}{SUBS[step % 4]}'
                print(f'bar {b:4d}  {pos:<3} {off:+.2f}  {name(n):<4} vel {v:3d}  len {ln:5.2f} 16ths  ch {ch}')


if __name__ == '__main__':
    main()
