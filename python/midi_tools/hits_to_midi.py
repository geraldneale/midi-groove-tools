#!/usr/bin/env python3
"""hits_to_midi.py: turn drum hits in audio into MIDI, keeping the human timing.

Finds every hit in one frequency band (snare by default), measures when its attack
starts and how hard it is, and writes one MIDI note per hit. Nothing is quantized.
Load the file on an MPC drum track and pick your own sample.

  hits_to_midi.py Drums.wav --bpm 109.45 --first 0.401 --bars 15-22
  hits_to_midi.py Drums.wav --bpm 109.45 --first 0.401 --band kick
  hits_to_midi.py HumanBehaviour_minus_stems.wav --bpm 109.45 --first 0.401 --bars 15-22

--first is the time of beat 1 in seconds (beatmap prints it). Bars are counted from
beat 1, the same way beatmap counts them. Beat 1 lands on bar 3 of the MIDI file
(--lead-bars 2), matching beatmap's two-bar lead-in, so it lines up in Logic and on
the MPC when the audio sits at bar 3 too.
"""
import argparse, os, sys, wave
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from midi_clean import write_midi

BANDS = {                  # Hz: where the band's attack shows best
    'kick': (40, 120),
    'snare': (180, 6000),
    'hat': (6000, 16000),
}
NOTES = {'kick': 36, 'snare': 38, 'hat': 42}     # General MIDI; MPC pads A01 etc. follow these


def read_mono(path):
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
    return x.reshape(-1, ch).mean(axis=1), sr


def band_pass(x, sr, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    # gentle edges so hits don't ring
    g = np.clip((f - lo * 0.7) / (lo * 0.3), 0, 1) * np.clip((hi * 1.3 - f) / (hi * 0.3), 0, 1)
    return np.fft.irfft(X * g, len(x)).astype(np.float32)


def find_hits(x, sr, lo, hi, sens=1.0, min_gap=0.025):
    """(attack time s, peak level) for each hit in the band."""
    y = band_pass(x, sr, lo, hi)
    hop = max(1, int(0.001 * sr))                      # 1 ms envelope
    n = len(y) // hop
    env = np.abs(y[:n * hop]).reshape(n, hop).max(axis=1)
    sm = np.convolve(env, np.ones(3) / 3, 'same')
    other = None
    if lo >= 150:                                      # a kick leaks its click into higher bands:
        yl = band_pass(x, sr, 30, 120)                 # keep the low band to tell kicks apart
        other = np.convolve(np.abs(yl[:n * hop]).reshape(n, hop).max(axis=1), np.ones(3) / 3, 'same')
    rise = np.maximum(np.diff(sm, prepend=sm[0]), 0)
    # adaptive threshold: local median of the rise plus a margin from the local spread
    local = np.convolve(rise, np.ones(601) / 601, 'same')       # 0.6 s moving average
    floor = 0.02 * sm.max()
    thr = (2.5 * local + 1.5 * rise.std()) / sens
    gap = int(min_gap * 1000)
    peaks = np.nonzero((rise[1:-1] > thr[1:-1]) & (rise[1:-1] >= rise[:-2]) & (rise[1:-1] >= rise[2:]))[0] + 1

    def jumps(i):
        j = min(n, i + 12)
        b = sm[i:j].max() - sm[max(0, i - 15):i + 1].min()
        lj = other[i:j].max() - other[max(0, i - 15):i + 1].min() if other is not None else 0.0
        return b, lj
    # how much of a kick leaks into this band: measured on hits where the low band clearly leads
    leak = 0.0
    if other is not None and len(peaks):
        r = [b / lj for b, lj in map(jumps, peaks) if lj > 3 * b and lj > 0]
        leak = float(np.median(r)) if len(r) >= 3 else 0.0
    hits, last = [], -gap
    i = 1
    while i < n - 1:
        if rise[i] > thr[i] and rise[i] >= rise[i - 1] and rise[i] >= rise[i + 1] and i - last >= gap:
            j = min(n, i + 30)
            peak = sm[i:j].max()
            trough = sm[last:i + 1].min() if hits else 0.0
            new_attack = peak > 1.8 * trough                # not just a bump in the last hit's decay
            base = sm[max(0, i - 15):i + 1].min()
            kick_only = False
            if other is not None:          # no more in this band than a kick leaks into it: a kick, not a hit
                b, lj = jumps(i)
                kick_only = lj > 3 * b and (leak == 0 or b < 1.3 * leak * lj)
            if peak > floor and new_attack and not kick_only:
                # attack: walk back to where the level was under 20% of the way up
                k = i
                while k > 0 and sm[k] > base + 0.2 * (peak - base) and i - k < 15:
                    k -= 1
                s0, s1 = k * hop, (i + 1) * hop
                a = np.abs(y[s0:s1])
                if len(a):
                    lvl = base + 0.2 * (peak - base)
                    over = np.nonzero(a > lvl)[0]
                    t = (s0 + (over[0] if len(over) else 0)) / sr
                    hits.append((t, float(peak)))
                    last = i
        i += 1
    return hits


def velocities(levels, lo=20, hi=127):
    """Peak levels to MIDI velocity on a log scale, loudest hit = hi."""
    lv = np.array(levels)
    d = 20 * np.log10(np.maximum(lv / lv.max(), 1e-4))     # dB below the loudest
    v = hi + d * (hi - lo) / 36                               # 36 dB range
    return np.clip(np.round(v), 1, 127).astype(int)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('wav')
    p.add_argument('--bpm', type=float, required=True)
    p.add_argument('--first', type=float, default=0.0, help='time of beat 1 in seconds (default 0)')
    p.add_argument('--bars', help='only these bars, e.g. 15-22 (counted from beat 1 = bar 1)')
    p.add_argument('--band', default='snare', help='kick, snare, hat, or LOW-HIGH in Hz (default snare)')
    p.add_argument('--note', type=int, help='MIDI note to write (default 38 snare, 36 kick, 42 hat)')
    p.add_argument('--sens', type=float, default=1.0, help='higher finds quieter hits (default 1.0)')
    p.add_argument('--lead-bars', type=int, default=2, help='empty bars before beat 1 (default 2)')
    p.add_argument('-o', help='output file (default <wav>_<band>.mid)')
    a = p.parse_args()

    if a.band in BANDS:
        lo, hi = BANDS[a.band]; note = a.note or NOTES[a.band]
    else:
        try:
            lo, hi = (float(v) for v in a.band.split('-'))
        except ValueError:
            sys.exit('--band must be kick, snare, hat, or LOW-HIGH in Hz')
        note = a.note or 38
    x, sr = read_mono(a.wav)
    beat = 60 / a.bpm
    bar = 4 * beat
    t0, t1 = 0.0, len(x) / sr
    if a.bars:
        b0, b1 = (int(v) for v in a.bars.split('-'))
        t0 = max(0.0, a.first + (b0 - 1) * bar - 0.05)
        t1 = min(t1, a.first + b1 * bar)
    seg = x[int(t0 * sr):int(t1 * sr)]
    hits = [(t + t0, lv) for t, lv in find_hits(seg, sr, lo, hi, a.sens)]
    if a.bars:
        hits = [h for h in hits if h[0] >= a.first + (b0 - 1) * bar - 0.02]
    if not hits:
        sys.exit('no hits found; try --sens 2 or another --band')

    tpb = 960
    vel = velocities([lv for _, lv in hits])
    notes = []
    for (t, _), v in zip(hits, vel):
        tick = round(((t - a.first) / beat + a.lead_bars * 4) * tpb)
        if tick < 0:
            continue
        notes.append({'start': tick, 'end': tick + tpb // 8, 'pitch': note, 'vel': int(v), 'ch': 9})
    tempo = round(60_000_000 / a.bpm).to_bytes(3, 'big')
    out = a.o or os.path.splitext(a.wav)[0] + f'_{a.band}.mid'
    write_midi(out, tpb, notes, [(0, 0x51, tempo)], [])

    print(f'{len(notes)} hits in {lo:.0f} to {hi:.0f} Hz, note {note}, written to {out}')
    print(f'beat 1 is at bar {a.lead_bars + 1} of the file; times are not quantized')
    print('\n  bar beat  offset from the 16th grid   vel')
    for n in notes:
        q = n['start'] - a.lead_bars * 4 * tpb
        s16 = tpb // 4
        g = round(q / s16) * s16                      # nearest 16th, for the label only
        b = g // (4 * tpb) + 1
        bt = g % (4 * tpb) / tpb + 1
        off = (q - g) / tpb * beat * 1000
        print(f'  {b:>3} {bt:5.2f}  {off:+6.1f} ms  {n["vel"]:>3}')


if __name__ == '__main__':
    main()
