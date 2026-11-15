#!/usr/bin/env python3
"""test_midi_tools.py: check midi_align and midi_clean still do the right thing.

Run after changing either script:  python3 test_midi_tools.py
Builds a fake synth bass stem (pattern B from Hung Up) and a fake Flex Pitch
export with the usual faults, runs the tools, and checks the results.
"""
import os, subprocess, sys, tempfile, wave
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from midi_clean import read_midi, write_midi

SR, BPM = 48000, 125
S16 = 60 / BPM / 4
TPB = 480
TICK = 60 / BPM / TPB
# pattern B: (16th step, MIDI pitch). Low notes on the beat, octave on + and a.
PAT = [(0, 26), (2, 38), (3, 38), (4, 26), (6, 38), (7, 38), (8, 29), (10, 41), (11, 41), (12, 24), (14, 36), (15, 36)]
FIRST, LAST = 8, 16                 # synth bass plays bars 9 to 16; bars 1-8 empty


def build(d):
    rng = np.random.default_rng(0)
    x = np.zeros(int(SR * S16 * 16 * (LAST + 2)))
    notes = []
    for bar in range(FIRST, LAST):
        for st, p in PAT:
            t = (bar * 16 + st) * S16 + rng.uniform(-0.005, 0.008)
            f = 440 * 2 ** ((p - 69) / 12)
            n = int(S16 * 1.8 * SR) if p < 36 else int(S16 * 0.9 * SR)
            tt = np.arange(n) / SR
            env = np.minimum(1, tt / 0.003) * np.exp(-tt * 6)
            if p < 36:                               # filter-envelope bump on the "e" (fools the attack finder)
                env *= 1 + 0.8 * np.exp(-((tt - S16) / 0.01) ** 2)
            i = int(t * SR)
            x[i:i + n] += 0.5 * (2 * ((tt * f) % 1) - 1) * env
            if p == 24:
                continue                             # fault: Flex Pitch misses the low C0
            late = 0.033 + (0.05 if rng.random() < 0.1 else rng.uniform(0, 0.015))   # fault: late detection
            s = (t + late) / TICK
            vel = 127 if bar < LAST - 2 else 20                   # fade-out in the last two bars
            notes.append({'start': round(s), 'end': round(s + (1.8 if p < 36 else 0.8) * S16 / TICK),
                          'pitch': p, 'vel': vel, 'ch': 0})
    for k in range(4):                               # drum bleed in bar 3: noise bursts, no pitch
        i = int((32 + k * 4) * S16 * SR); n = int(0.1 * SR); tt = np.arange(n) / SR
        x[i:i + n] += 0.6 * rng.normal(size=n) * np.exp(-tt * 40)
    x /= np.abs(x).max() * 1.1
    with wave.open(os.path.join(d, 'b.wav'), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype('<i2').tobytes())
    write_midi(os.path.join(d, 'b.mid'), TPB, notes, [(0, 0x51, (60_000_000 // BPM).to_bytes(3, 'big'))], [])


def run(d, *args):
    r = subprocess.run([sys.executable, *args], cwd=d, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)
    return r.stdout


def beat_test(check):
    """beatmap.py on a fake song whose tempo drifts from 112.3 to 108.7 BPM."""
    import shutil
    from beatmap import read_wav, find_beats
    rng = np.random.default_rng(3)
    sr, nb = 44100, 160
    bpm = np.linspace(112.3, 108.7, nb) + 0.6 * np.sin(np.arange(nb) / 9)
    truth = 1.37 + np.concatenate([[0], np.cumsum(60 / bpm[:-1])])
    x = np.zeros((int((truth[-1] + 3) * sr), 2))
    tt = np.arange(int(0.3 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
    hat = rng.normal(size=len(tt)) * np.exp(-tt * 120) * 0.15
    timp = np.sin(2 * np.pi * 90 * tt) * np.exp(-tt * 4) * 0.5
    def add(t, s):
        i = int(t * sr); x[i:i + len(s)] += s[:len(x) - i, None]
    for i, b in enumerate(truth):
        if rng.random() < 0.85: add(b, kick)
        add(b + 0.27, hat)
        if rng.random() < 0.3: add(b + rng.choice([0.13, 0.4]), timp)     # quirky off-beat hits
    x += 0.01 * rng.normal(size=x.shape)
    x /= np.abs(x).max() * 1.1
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'song.wav')
        with wave.open(path, 'wb') as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((x * 32767).astype('<i2').tobytes())
        from beatmap import logic_position
        print('beatmap Logic region position')
        check(logic_position(0.401, 109.45, 2) == '2 4 2 19', 'Human Behaviour: first beat 0.401 s at 109.45 -> 2 4 2 19')
        check(logic_position(60 / 125 * 8, 125, 2) == '1 1 1 1', 'two bars of silence already in the file -> 1 1 1 1')
        print('beatmap on a fake song drifting from 112.3 to 108.7 BPM')
        b = find_beats(read_wav(path)[5], sr, 111)
        check(abs(len(b) - nb) <= 2, f'found the beats ({len(b)} of {nb}), none in the silence')
        err = np.array([np.abs(truth - v).min() for v in b]) * 1000
        check(np.median(err) < 5, f'beats where the kicks are: median {np.median(err):.1f} ms off')
        if shutil.which('rubberband'):
            run(d, os.path.join(HERE, 'beatmap.py'), 'song.wav', '--bpm', '111', '--stretch')
            b2 = find_beats(read_wav(os.path.join(d, 'song_111bpm.wav'))[5], sr, 111)
            grid = 8 * 60 / 111 + np.arange(len(b2)) * 60 / 111
            dev = np.abs(b2 - grid) * 1000
            check(np.median(dev) < 8, f'stretched file on the 111 grid: median {np.median(dev):.1f} ms off')
            check(abs(b2[0] - 8 * 60 / 111) < 0.01, 'first beat lands on bar 3')
        else:
            print('  skip rubberband not installed, stretch not tested')


def main():
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(('  ok   ' if cond else '  FAIL ') + msg)
        ok &= bool(cond)

    with tempfile.TemporaryDirectory() as d:
        build(d)
        run(d, os.path.join(HERE, 'midi_align.py'), 'b.mid', 'b.wav', '--window', '40')
        run(d, os.path.join(HERE, 'midi_clean.py'), 'b_aligned.mid', '--no-offset', '--fix-late-lows', '--fix-outliers', '0.25')
        tpb, notes, metas, other = read_midi(os.path.join(d, 'b_aligned_clean.mid'))
        q = tpb // 4
        cell = {}
        for n in notes:
            cell.setdefault((round(n['start'] / q), n['pitch']), []).append(n['start'] - round(n['start'] / q) * q)
        want = {((bar * 16 + st), p) for bar in range(FIRST, LAST) for st, p in PAT}

        print('midi_align + midi_clean on a fake pattern B stem')
        check(any(t == 0 and r[0] & 0xF0 == 0xB0 and r[1] == 110 for t, r in other), 'bar 1 marker (CC 110) at tick 0')
        c0 = sum(1 for k in want if k[1] == 24 and k in cell)
        check(c0 >= (LAST - FIRST) - 1, f'missing C0 on beat 4 added: {c0} of {LAST - FIRST} bars')
        extra = sorted(k for k in cell if k not in want)
        check(len(extra) <= 1, f'no extra notes (found {len(extra)}: {extra[:6]})')
        bleed = [k for k in cell if k[0] < FIRST * 16]
        check(not bleed, f'nothing added for drum bleed (found {len(bleed)})')
        offs = [abs(o) for v in cell.values() for o in v]
        check(max(offs) <= q * 0.25, f'every note within 1/4 of a 16th of the grid (worst {max(offs) / q:.2f})')
        loud = [n for n in notes if n['start'] >= (LAST - 2) * 16 * q + q and n['vel'] > 40]
        check(not loud, f'added notes in a fade match the fade (found {len(loud)} loud ones)')
        dup = [k for k, v in cell.items() if len(v) > 1]
        check(not dup, f'no doubled notes (found {len(dup)})')
    beat_test(check)
    print('all good' if ok else 'SOMETHING BROKE')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
