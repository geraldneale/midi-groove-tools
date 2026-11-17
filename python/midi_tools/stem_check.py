#!/usr/bin/env python3
"""stem-check: do all the stems of a song line up?

Reads the header of every WAV and checks they all have the same length (to the
sample), sample rate and bit depth, so they start and end together when loaded
as audio tracks on the MPC.

  stem-check 01_SongName_Stems              every .wav in the folder
  stem-check Bass.wav Drums.wav Pads.wav    just these files

Exit code 0 when everything matches, 1 when something differs.

Only the headers are read, so it is quick on long files. WAVE_FORMAT_EXTENSIBLE
(Logic writes it for some exports) and 32-bit float are read the same as plain PCM.
"""
import argparse, os, struct, sys
from collections import Counter

FORMATS = {1: 'int', 3: 'float'}
EXTENSIBLE = 0xFFFE


def wav_info(path):
    """(sample rate, bits, channels, frames, kind) from the RIFF header."""
    with open(path, 'rb') as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b'RIFF' or head[8:12] != b'WAVE':
            raise ValueError('not a RIFF WAVE file')
        fmt = None
        while True:
            h = f.read(8)
            if len(h) < 8:
                raise ValueError('no data chunk')
            cid, size = h[:4], struct.unpack('<I', h[4:])[0]
            if cid == b'fmt ':
                body = f.read(size + (size & 1))                # chunks are padded to even sizes
                tag, ch, sr, _, block, bits = struct.unpack('<HHIIHH', body[:16])
                if tag == EXTENSIBLE and size >= 40:
                    tag = struct.unpack('<H', body[24:26])[0]   # first two bytes of the subformat GUID
                fmt = (tag, ch, sr, block, bits)
            elif cid == b'data':
                if fmt is None:
                    raise ValueError('data chunk before fmt chunk')
                tag, ch, sr, block, bits = fmt
                if tag not in FORMATS:
                    raise ValueError(f'unsupported format tag {tag}')
                if size == 0xFFFFFFFF:
                    raise ValueError('data size not set (file still being written, or RF64)')
                return sr, bits, ch, size // block, FORMATS[tag]
            else:
                f.seek(size + (size & 1), 1)


def length(frames, sr):
    s = frames / sr
    return f'{int(s // 60)}:{s % 60:06.3f}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('paths', nargs='+', help='a folder of stems, or WAV files')
    a = p.parse_args()

    files = []
    for path in a.paths:
        if os.path.isdir(path):
            files += sorted(os.path.join(path, n) for n in os.listdir(path)
                            if n.lower().endswith('.wav') and not n.startswith('.'))
        else:
            files.append(path)
    if not files:
        sys.exit('no .wav files found')

    rows, bad = [], []
    for path in files:
        try:
            rows.append((os.path.basename(path),) + wav_info(path))
        except (OSError, ValueError, struct.error) as e:
            bad.append(f'{os.path.basename(path)}: {e}')

    w = max([len(r[0]) for r in rows] + [4])
    print(f'{len(files)} WAV files')
    print(f'  {"file":<{w}}  {"rate":>6}  {"bits":>9}  {"ch":>2}  {"frames":>11}  {"length":>10}')
    for name, sr, bits, ch, frames, kind in rows:
        print(f'  {name:<{w}}  {sr:>6}  {f"{bits} {kind}":>9}  {ch:>2}  {frames:>11}  {length(frames, sr):>10}')

    problems = list(bad)
    if rows:
        common = {k: Counter(r[i] for r in rows).most_common(1)[0][0]
                  for k, i in (('rate', 1), ('bits', 2), ('kind', 5), ('frames', 4))}
        sr = common['rate']
        for name, r, bits, ch, frames, kind in rows:
            if r != common['rate']:
                problems.append(f'{name}: {r} Hz, the others are {common["rate"]} Hz')
            if (bits, kind) != (common['bits'], common['kind']):
                problems.append(f'{name}: {bits} bit {kind}, the others are {common["bits"]} bit {common["kind"]}')
            d = frames - common['frames']
            if d:
                problems.append(f'{name}: {abs(d)} frames {"longer" if d > 0 else "shorter"} '
                                f'({abs(d) / r * 1000:.1f} ms) than the others')

    print()
    if problems:
        for msg in problems:
            print(f'  {msg}')
        print('verdict  MISMATCH')
        sys.exit(1)
    print(f'verdict  MATCH  ({len(rows)} files, {sr} Hz, {common["bits"]} bit {common["kind"]}, '
          f'{length(common["frames"], sr)})')


if __name__ == '__main__':
    main()
