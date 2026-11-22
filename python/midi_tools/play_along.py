#!/usr/bin/env python3
"""play-along: play a song's stems from any bar, loop a few bars, leave your part out.

Mixes the stems in a folder (or the WAVs you name), cuts out the bars you ask for,
and plays them on a loop until Ctrl+C, so you can pick up an instrument and play.

  play-along mpc44 --bpm 125 --from 25 --to 33 --minus Bass
  play-along mpc44 --bpm 125 --from 25 --to 33 --minus Bass --count-in 1
  play-along mpc44 --bpm 125 --from 17 --minus Bass --minus Kick --once
  play-along mpc44 --bpm 125 --from 25 --to 33 --out loop.wav     # write it, don't play

Bars are counted from bar 1 at the start of the files (the standard for stems
exported from Logic and padded with stem-pad). --to is the last bar played, so
--from 25 --to 32 is 8 bars. Without --to it plays to the end of the song, once.
--minus leaves out every file whose name contains that word (any case); give it
more than once to leave out several. --count-in adds bars of click before the
first pass. Plays with afplay on a Mac, or aplay / paplay / ffplay on Linux.
"""
import argparse, os, shutil, signal, struct, subprocess, sys, tempfile, time
import numpy as np


def wav_info(path):
    """(channels, rate, bits, float?, data offset, frames) from a RIFF WAV header."""
    with open(path, 'rb') as f:
        head = f.read(12)
        if head[:4] != b'RIFF' or head[8:12] != b'WAVE':
            sys.exit(f'{path}: not a WAV file')
        fmt = None
        while True:
            h = f.read(8)
            if len(h) < 8:
                sys.exit(f'{path}: no audio found in the file')
            cid, size = h[:4], struct.unpack('<I', h[4:])[0]
            if cid == b'fmt ':
                body = f.read(size + (size & 1))
                tag, ch, rate = struct.unpack('<HHI', body[:8])
                bits = struct.unpack('<H', body[14:16])[0]
                if tag == 0xFFFE and len(body) >= 26:
                    tag = struct.unpack('<H', body[24:26])[0]
                if tag not in (1, 3):
                    sys.exit(f'{path}: compressed WAV; convert it first: afconvert -f WAVE -d LEI24@44100 in.wav out.wav')
                fmt = (ch, rate, bits, tag == 3)
            elif cid == b'data':
                if not fmt:
                    sys.exit(f'{path}: damaged header')
                ch, rate, bits, flt = fmt
                return ch, rate, bits, flt, f.tell(), size // (ch * bits // 8)
            else:
                f.seek(size + (size & 1), 1)


def read_frames(path, start, count):
    """float32 (frames, channels) for frames start..start+count, padded with silence past the end."""
    ch, rate, bits, flt, off, total = wav_info(path)
    width = bits // 8
    n = max(0, min(count, total - start))
    with open(path, 'rb') as f:
        f.seek(off + start * ch * width)
        raw = f.read(n * ch * width)
    if flt:
        x = np.frombuffer(raw, '<f4' if width == 4 else '<f8').astype(np.float32)
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = (np.pad(b, ((0, 0), (1, 0))).view('<i4').ravel() >> 8).astype(np.float32) / 2 ** 23
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    else:
        x = np.frombuffer(raw, {2: '<i2', 4: '<i4'}[width]).astype(np.float32) / (2 ** (bits - 1))
    x = x.reshape(-1, ch)
    if len(x) < count:
        x = np.vstack([x, np.zeros((count - len(x), ch), np.float32)])
    return x


def click_bars(bars, beats, bpm, rate, ch):
    """Count-in: short blips on every beat, higher on beat 1."""
    beat = 60 / bpm * rate
    n = int(round(bars * beats * beat))
    out = np.zeros(n, np.float32)
    blip = int(0.03 * rate)
    t = np.arange(blip) / rate
    env = np.exp(-t * 120)
    for k in range(bars * beats):
        f = 1600 if k % beats == 0 else 1000
        i = int(round(k * beat))
        out[i:i + blip] += 0.5 * env[:n - i] * np.sin(2 * np.pi * f * t[:n - i])
    return np.repeat(out[:, None], ch, axis=1)


def write_wav(path, x, rate):
    with open(path, 'wb') as f:
        data = np.round(np.clip(x, -1, 1 - 2 ** -15) * 2 ** 15).astype('<i2').tobytes()
        ch = x.shape[1]
        f.write(b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVE')
        f.write(b'fmt ' + struct.pack('<IHHIIHH', 16, 1, ch, rate, rate * ch * 2, ch * 2, 16))
        f.write(b'data' + struct.pack('<I', len(data)) + data)


def player():
    for cmd in (['afplay'], ['aplay', '-q'], ['paplay'], ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet']):
        if shutil.which(cmd[0]):
            return cmd
    sys.exit('no audio player found (afplay, aplay, paplay or ffplay); use --out FILE instead')


def main():
    p = argparse.ArgumentParser(prog='play-along', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stems', nargs='+', help='a folder of WAVs, or the WAV files')
    p.add_argument('--bpm', type=float, required=True)
    p.add_argument('--beats', type=int, default=4, help='beats per bar (default 4)')
    p.add_argument('--from', dest='first', type=int, default=1, help='first bar (default 1)')
    p.add_argument('--to', dest='last', type=int, help='last bar played; loops unless --once')
    p.add_argument('--minus', action='append', default=[], help='leave out files whose name contains this')
    p.add_argument('--count-in', type=int, default=0, metavar='BARS', help='bars of click before the first pass')
    p.add_argument('--once', action='store_true', help='play once instead of looping')
    p.add_argument('--out', help='write the mix to this WAV instead of playing it')
    a = p.parse_args()

    files = []
    for s in a.stems:
        if os.path.isdir(s):
            files += sorted(os.path.join(s, n) for n in os.listdir(s)
                            if n.lower().endswith('.wav') and not n.startswith('.'))
        else:
            files.append(s)
    if not files:
        sys.exit('no WAV files found')
    left = [f for f in files if any(m.lower() in os.path.basename(f).lower() for m in a.minus)]
    keep = [f for f in files if f not in left]
    for m in a.minus:
        if not any(m.lower() in os.path.basename(f).lower() for f in files):
            print(f'note: no file name contains "{m}"')
    if not keep:
        sys.exit('every file was left out; nothing to play')

    infos = {f: wav_info(f) for f in keep}
    rates = {i[1] for i in infos.values()}
    if len(rates) > 1:
        sys.exit(f'the files have different sample rates {sorted(rates)}; run stem-check')
    rate = rates.pop()
    ch = max(i[0] for i in infos.values())
    total = max(i[5] for i in infos.values())
    bar = a.beats * 60 / a.bpm * rate
    song_bars = int(total // round(bar)) or 1
    last = a.last or song_bars
    if a.first < 1 or last < a.first:
        sys.exit(f'bars must run forward from 1 (got --from {a.first} --to {last})')
    if a.first > song_bars + 1:
        sys.exit(f'the song is only about {song_bars} bars long at {a.bpm:g} BPM')
    start = int(round((a.first - 1) * bar))
    count = int(round(last * bar)) - start
    loop = a.last is not None and not a.once

    mix = np.zeros((count, ch), np.float32)
    for f in keep:
        x = read_frames(f, start, count)
        mix += x if x.shape[1] == ch else np.repeat(x.mean(axis=1, keepdims=True), ch, axis=1)
    peak = float(np.abs(mix).max())
    if peak > 0.99:
        mix *= 0.99 / peak
        print(f'mix peaked above full scale; turned down {20 * np.log10(peak / 0.99):.1f} dB')

    names = ', '.join(os.path.basename(f) for f in left) or 'nothing'
    bars = last - a.first + 1
    print(f'bars {a.first}-{last} ({bars} bar{"s" if bars != 1 else ""}, {count / rate:.1f} s) '
          f'at {a.bpm:g} BPM, {len(keep)} stems; left out: {names}')

    lead = click_bars(a.count_in, a.beats, a.bpm, rate, ch) if a.count_in else np.zeros((0, ch), np.float32)
    if a.out:
        write_wav(a.out, np.vstack([lead, mix]), rate)
        print(f'wrote {a.out}')
        return

    cmd = player()
    tmp = tempfile.mkdtemp(prefix='play-along-')
    try:
        if loop:
            # One long file of back-to-back passes, so the loop point has no gap.
            reps = max(1, int(np.ceil(300 * rate / count)))
            write_wav(os.path.join(tmp, 'a.wav'), np.vstack([lead] + [mix] * reps), rate)
            print('looping; Ctrl+C to stop')
        else:
            write_wav(os.path.join(tmp, 'a.wav'), np.vstack([lead, mix]), rate)
        while True:
            proc = subprocess.Popen(cmd + [os.path.join(tmp, 'a.wav')])
            proc.wait()
            if not loop:
                break
            if a.count_in:  # later passes start straight on the downbeat
                write_wav(os.path.join(tmp, 'a.wav'), np.vstack([mix] * reps), rate)
                a.count_in = 0
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=2)
        except Exception:
            pass
        print()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
