#!/usr/bin/env python3
"""stem-pad: pad every stem with silence to an exact number of bars.

Logic's track export can stop each file where that track's sound stops, so the
stems of one song come out different lengths even though they all start at bar 1.
This adds silence to the end of each WAV (in place) so they are all exactly the
song length, counted from bar 1.

  stem-pad 01_SongName_Stems --bars 117 --bpm 92

Files must start at bar 1. A file already longer than the target is left alone
and reported. Run stem-check afterwards.
"""
import argparse, os, struct, sys


def chunks(data):
    """[(id, body)] for every chunk after the RIFF/WAVE header."""
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError('not a RIFF WAVE file')
    out, i = [], 12
    while i + 8 <= len(data):
        cid, size = data[i:i + 4], struct.unpack('<I', data[i + 4:i + 8])[0]
        out.append((cid, data[i + 8:i + 8 + size]))
        i += 8 + size + (size & 1)
    return out


def main():
    p = argparse.ArgumentParser(prog='stem-pad', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder')
    p.add_argument('--bars', type=int, required=True, help='song length in bars, counted from bar 1')
    p.add_argument('--bpm', type=float, required=True)
    p.add_argument('--beats', type=int, default=4, help='beats per bar (default 4)')
    a = p.parse_args()

    files = sorted(os.path.join(a.folder, n) for n in os.listdir(a.folder)
                   if n.lower().endswith('.wav') and not n.startswith('.'))
    if not files:
        sys.exit('no .wav files found')
    bad = False
    for path in files:
        name = os.path.basename(path)
        with open(path, 'rb') as f:
            data = f.read()
        try:
            cs = chunks(data)
        except ValueError as e:
            print(f'  skip {name}: {e}'); bad = True; continue
        fmt = next((b for c, b in cs if c == b'fmt '), None)
        if fmt is None or not any(c == b'data' for c, _ in cs):
            print(f'  skip {name}: no fmt or data chunk'); bad = True; continue
        sr, block = struct.unpack('<I', fmt[4:8])[0], struct.unpack('<H', fmt[12:14])[0]
        target = round(a.bars * a.beats * 60 / a.bpm * sr)
        have = len(next(b for c, b in cs if c == b'data')) // block
        if have > target:
            print(f'  {name}: {have - target} frames longer than {a.bars} bars, left alone'); bad = True; continue
        if have == target:
            print(f'  {name}: already {a.bars} bars'); continue
        body = b''
        for c, b in cs:
            if c == b'data':
                b = b[:have * block] + bytes((target - have) * block)
            body += c + struct.pack('<I', len(b)) + b + (b'\0' if len(b) & 1 else b'')
        with open(path, 'wb') as f:
            f.write(b'RIFF' + struct.pack('<I', 4 + len(body)) + b'WAVE' + body)
        print(f'  {name}: padded {(target - have) / sr:.3f} s')
    print(f'target {a.bars} bars at {a.bpm:g} BPM')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
