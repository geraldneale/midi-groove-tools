#!/usr/bin/env python3
"""song_minus_stems.py: what did the stem splitter drop?

Adds the stems together, lines them up with the original mix, and subtracts them.
What is left (the residual) is what no stem got. Writes it as a WAV and lists the
bars where the most went missing.

  song_minus_stems.py song.wav drums.wav bass.wav vocals.wav other.wav
  song_minus_stems.py song.wav stems/*.wav --bpm 109.45 --first 0.401
  song_minus_stems.py song.wav stems/*.wav --solo drums.wav     # mix minus everything but drums

All files must be WAV at the same sample rate (afconvert or ffmpeg first if not).
--first is the time of beat 1 in seconds (beatmap prints it); with --bpm it turns
times into bar numbers.
"""
import argparse, os, sys, wave
import numpy as np


def read(path):
    """float32 array (frames, channels) and the sample rate."""
    try:
        w = wave.open(path, 'rb')
    except wave.Error as e:
        sys.exit(f'{path}: {e}. Convert it first: afconvert -f WAVE -d LEI24 "{path}" out.wav')
    with w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = (np.pad(b, ((0, 0), (1, 0))).view('<i4').ravel() >> 8).astype(np.float32) / 2 ** 23
    else:
        x = np.frombuffer(raw, {1: np.uint8, 2: '<i2', 4: '<i4'}[width]).astype(np.float32)
        x = (x - 128) / 128 if width == 1 else x / {2: 2 ** 15, 4: 2 ** 31}[width]
    return x.reshape(-1, ch), sr


def write(path, x, sr):
    peak = np.abs(x).max()
    if peak > 1:
        x = x / peak
        print(f'  residual peaked above full scale; scaled down {20 * np.log10(peak):.1f} dB')
    with wave.open(path, 'wb') as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(3); w.setframerate(sr)
        i = np.round(np.clip(x, -1, 1 - 2 ** -23) * 2 ** 23).astype('<i4').ravel()
        w.writeframes(i.view(np.uint8).reshape(-1, 4)[:, :3].tobytes())


def to_channels(x, ch):
    if x.shape[1] == ch:
        return x
    if x.shape[1] == 1:
        return np.repeat(x, ch, axis=1)
    return np.repeat(x.mean(axis=1, keepdims=True), ch, axis=1)


def best_lag(a, b, sr, max_s=0.5, span_s=90):
    """Samples to shift b so it lines up with a (positive: b is late)."""
    n = min(len(a), len(b), int(span_s * sr))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    size = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    m = int(max_s * sr)
    c = np.concatenate([c[-m:], c[:m + 1]])
    return int(np.argmax(c)) - m


def db(v):
    return 20 * np.log10(max(v, 1e-12))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mix')
    p.add_argument('stems', nargs='+')
    p.add_argument('--bpm', type=float, help='song tempo, to report bars instead of seconds')
    p.add_argument('--first', type=float, default=0.0, help='time of beat 1 in seconds (default 0)')
    p.add_argument('--solo', help='one stem: also write mix minus all the other stems')
    p.add_argument('--top', type=int, default=12, help='how many bars to list (default 12)')
    p.add_argument('-o', help='residual file name (default <mix>_minus_stems.wav)')
    a = p.parse_args()

    mix, sr = read(a.mix)
    ch = mix.shape[1]
    stems = {}
    for s in a.stems:
        x, r = read(s)
        if r != sr:
            sys.exit(f'{s} is {r} Hz but the mix is {sr} Hz. Convert one so they match.')
        stems[s] = to_channels(x, ch)
    n = max(len(mix), *(len(x) for x in stems.values()))
    total = np.zeros((n, ch), np.float32)
    for x in stems.values():
        total[:len(x)] += x
    mix = np.pad(mix, ((0, n - len(mix)), (0, 0)))

    lag = best_lag(mix.mean(axis=1), total.mean(axis=1), sr)
    if lag:
        total = np.roll(total, lag, axis=0)
        for k in stems:
            stems[k] = np.roll(np.pad(stems[k], ((0, n - len(stems[k])), (0, 0))), lag, axis=0)
    g = float(np.sum(mix * total, dtype=np.float64) / max(np.sum(total * total, dtype=np.float64), 1e-12))
    res = mix - g * total

    print(f'mix {os.path.basename(a.mix)}: {n / sr:.1f} s, {sr} Hz, {ch} ch, {len(stems)} stems')
    print(f'stems shifted {lag / sr * 1000:+.1f} ms to line up; level match {db(g):+.2f} dB')
    print(f'residual is {db(rms(res)) - db(rms(mix)):.1f} dB below the mix '
          '(about -30 or lower is a clean split; -15 or higher means a lot went missing)')
    out = a.o or os.path.splitext(a.mix)[0] + '_minus_stems.wav'
    write(out, res, sr)
    print(f'wrote {out}')

    if a.solo:
        key = next((k for k in stems if os.path.abspath(k) == os.path.abspath(a.solo)), None)
        if key is None:
            sys.exit(f'--solo {a.solo} is not one of the stems given')
        solo = mix - g * (total - stems[key])
        sout = os.path.splitext(a.mix)[0] + '_minus_all_but_' + os.path.splitext(os.path.basename(key))[0] + '.wav'
        write(sout, solo, sr)
        print(f'wrote {sout} (the mix with every other stem taken out)')

    # where it went missing: residual level against mix level, per bar or per 2 s
    step = 240 / a.bpm if a.bpm else 2.0
    t0 = a.first % step if a.bpm else 0.0
    rows = []
    t = t0
    while t < n / sr:
        i, j = int(t * sr), int(min(n, (t + step) * sr))
        m, r = rms(mix[i:j]), rms(res[i:j])
        if m > 1e-4:
            rows.append((db(r) - db(m), t, db(r)))
        t += step
    rows.sort(reverse=True)
    unit = 'bar' if a.bpm else 'time'
    print(f'\nmost missing, by {unit} (residual compared with the mix there):')
    for rel, t, lvl in rows[:a.top]:
        where = f'bar {int(round((t - a.first) / step)) + 1:>3}' if a.bpm else f'{int(t // 60)}:{t % 60:04.1f}'
        print(f'  {where}  at {int(t // 60)}:{t % 60:04.1f}   {rel:6.1f} dB')


if __name__ == '__main__':
    main()
