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
        err = np.array([np.abs(truth - v).min() for v in b]) * 1000
        check(nb - 4 <= len(b) <= nb and err.max() < 120,
              f'found the beats ({len(b)} of {nb}; a few with no kick may be left out), none in the silence')
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


def break_test(check):
    """beatmap on a steady 92 BPM song with an 8 bar break: no kick, loud off-beat hats over soft on-beat ones.
    The tracker must not slide onto the off-beats there (Lebanese Blonde did)."""
    from beatmap import find_beats, beat_numbers
    rng = np.random.default_rng(5)
    sr, bpm, nbars = 44100, 92.0, 40
    T = 60 / bpm
    truth = 0.5 + np.arange(nbars * 4) * T
    x = np.zeros(int((truth[-1] + 2) * sr))
    tt = np.arange(int(0.3 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
    hat = rng.normal(size=len(tt)) * np.exp(-tt * 90)
    def add(t, s, g):
        i = int(t * sr); x[i:i + len(s)] += g * s[:len(x) - i]
    for i, b in enumerate(truth):
        brk = 16 * 4 <= i < 24 * 4
        if not brk:
            add(b + rng.normal(0, 0.004), kick, 0.9)
        else:
            add(b, hat, 0.12)                                # the break: a soft hat on the beat,
        add(b + T / 2, hat, 0.6 if brk else 0.15)          # loud hats on the off-beats
    x += 0.005 * rng.normal(size=len(x))
    b = find_beats(x / np.abs(x).max() / 1.1, sr, bpm)
    k = beat_numbers(b)
    slope, first = np.polyfit(k, b, 1)
    print('beatmap on a steady 92 song with an 8 bar break of off-beat hats')
    near = np.array([np.abs(truth - v).min() for v in b])
    check((near < 0.05).all(), f'no beats slid onto the off-beats in the break (worst {near.max() * 1000:.0f} ms from a real beat)')
    check(abs(60 / slope - bpm) < 0.02, f'tempo fit stays {60 / slope:.2f} (92.00 wanted)')


def minus_test(check):
    """song_minus_stems.py on a fake split that lost a snare roll in bars 5 and 6."""
    rng = np.random.default_rng(7)
    sr, bpm = 44100, 110
    bar = 240 / bpm
    n = int(12 * bar * sr)
    t = np.arange(n) / sr
    bass = 0.3 * np.sin(2 * np.pi * 55 * t)
    drums = np.zeros(n)
    hit = rng.normal(size=int(0.08 * sr)) * np.exp(-np.arange(int(0.08 * sr)) / sr * 60)
    for k in range(48):                                   # a hit on every beat
        i = int(k * bar / 4 * sr); drums[i:i + len(hit)] += 0.4 * hit
    roll = np.zeros(n)
    for k in range(32):                                   # 16th note roll, bars 5 and 6
        i = int((4 * bar + k * bar / 16) * sr); roll[i:i + len(hit)] += 0.3 * hit
    mix = bass + drums + roll
    def save(path, x, shift=0):
        x = np.roll(x, shift)
        with wave.open(path, 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())
    with tempfile.TemporaryDirectory() as d:
        save(os.path.join(d, 'mix.wav'), mix / 1.5)
        save(os.path.join(d, 'drums.wav'), drums / 1.5, int(0.012 * sr))    # stems come out 12 ms late
        save(os.path.join(d, 'bass.wav'), bass / 1.5, int(0.012 * sr))
        out = run(d, os.path.join(HERE, 'song_minus_stems.py'), 'mix.wav', 'drums.wav', 'bass.wav',
                  '--bpm', str(bpm), '--top', '2')
        print('song_minus_stems on a fake split that lost a snare roll')
        check('-12.0 ms' in out or '-12.' in out.split('shifted')[1][:8], 'found the 12 ms offset')
        top = [l for l in out.splitlines() if l.strip().startswith('bar')]
        bars = sorted(int(l.split()[1]) for l in top)
        check(bars == [5, 6], f'lost roll found in bars 5 and 6 (listed {bars})')
        r, _ = (lambda w: (np.frombuffer(w.readframes(w.getnframes()), np.uint8), 0))(wave.open(os.path.join(d, 'mix_minus_stems.wav'), 'rb'))
        check(len(r) > 0, 'residual WAV written')


def hits_test(check):
    """hits_to_midi.py on a fake drum stem: a human snare roll over a kick on every beat."""
    from midi_clean import read_midi
    rng = np.random.default_rng(11)
    sr, bpm, first = 44100, 109.45, 0.401
    beat = 60 / bpm
    n = int((first + 10 * 4 * beat) * sr)
    x = np.zeros(n)
    tt = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 60 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 12)
    snare = (rng.normal(size=len(tt)) * 0.6 + np.sin(2 * np.pi * 200 * tt)) * np.exp(-tt * 30)
    def add(t, s, g):
        i = int(t * sr); x[i:i + len(s)] += g * s[:n - i]
    for k in range(40):
        add(first + k * beat, kick, 0.8)
    truth, gains = [], []
    for k in range(32):                                   # bars 5 and 6: 16ths, crescendo, human timing
        t = first + 4 * 4 * beat + k * beat / 4 + rng.normal(0, 0.006)
        g = 0.15 + 0.6 * k / 31
        add(t, snare, g); truth.append(t); gains.append(g)
    for t in (first + 4 * 4 * beat + 3 * beat / 4 + beat / 8, first + 5 * 4 * beat + 9 * beat / 4 + beat / 8):
        add(t, snare, 0.3); truth.append(t); gains.append(0.3)    # two trill notes between 16ths
    truth = np.sort(truth)
    x /= np.abs(x).max() * 1.1
    with tempfile.TemporaryDirectory() as d:
        with wave.open(os.path.join(d, 'drums.wav'), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((x * 32767).astype('<i2').tobytes())
        run(d, os.path.join(HERE, 'hits_to_midi.py'), 'drums.wav', '--bpm', str(bpm),
            '--first', str(first), '--bars', '4-7')
        tpb, notes, metas, other = read_midi(os.path.join(d, 'drums_snare.mid'))
        print('hits_to_midi on a fake snare roll over a kick')
        check(len(notes) == len(truth), f'one note per snare hit, no kicks ({len(notes)} of {len(truth)})')
        beat_ticks = tpb
        got = np.array(sorted(first + (nn['start'] / beat_ticks - 8) * beat for nn in notes))
        if len(got) == len(truth):
            err = np.abs(got - truth) * 1000
            check(np.median(err) < 2 and err.max() < 12, f'hit times kept, not quantized: median {np.median(err):.1f} ms, worst {err.max():.1f} ms off')
            v = [nn['vel'] for nn in sorted(notes, key=lambda m: m['start'])]
            check(v[-1] > v[0] + 30, f'crescendo kept in velocities ({v[0]} up to {v[-1]})')
        check(min(nn['start'] for nn in notes) >= 8 * tpb + 4 * 4 * tpb - tpb // 8, 'first hit lands at bar 5 after the 2 bar lead-in (file bar 7)')


def clock_test(check):
    """clock_check.py on a fake MPC clock: 120 BPM, 0.3 ms jitter, a stop and restart,
    then a change to 126 BPM, with one dropped tick and one doubled tick."""
    from clock_check import analyze, write_csv
    rng = np.random.default_rng(5)
    sd = 0.0003
    events, t = [(0.0, 'start')], 0.0
    def ticks(bpm, count, drop=None, double=None):
        nonlocal t
        step = 60 / bpm / 24
        for k in range(count):
            t += step
            if k == drop:
                continue
            events.append((t + rng.normal(0, sd), 'clock'))
            if k == double:
                events.append((t + rng.normal(0, sd) + 0.0004, 'clock'))
    ticks(120, 24 * 32, drop=400)
    t += 3.0                                              # stopped for 3 s: not jitter, not missing
    events.append((t, 'stop')); events.append((t + 0.001, 'start'))
    ticks(120, 24 * 8)
    ticks(126, 24 * 32, double=300)
    print('midi-clock-check on a fake 120 then 126 BPM clock with 0.3 ms jitter')
    s = analyze(events)
    want = 0.3 * np.sqrt(2)                               # interval jitter from 0.3 ms per tick
    check(abs(s['sd'] - want) < 0.05, f"jitter sd found: {s['sd']:.3f} ms (made with {want:.3f})")
    check(s['missing'] == 1 and s['doubled'] == 1, f"one tick dropped, one doubled (found {s['missing']} and {s['doubled']})")
    check(abs(s['bpm_min'] - 120) < 0.3 and abs(s['bpm_max'] - 126) < 0.3,
          f"tempo range {s['bpm_min']:.2f} to {s['bpm_max']:.2f} (made 120 and 126)")
    check(s['worst'] < 2, f"the stop and the tempo change are not counted as jitter (worst {s['worst']:.3f} ms)")
    with tempfile.TemporaryDirectory() as d:
        write_csv(os.path.join(d, 'ticks.csv'), events)
        r = subprocess.run([sys.executable, os.path.join(HERE, 'clock_check.py'), '--replay', 'ticks.csv'],
                           cwd=d, capture_output=True, text=True)
        check('NOT STEADY' in r.stdout and r.returncode == 1, 'replay of the saved run: verdict NOT STEADY (a tick went missing)')
        steady = [e for i, e in enumerate(events) if i < 300]
        write_csv(os.path.join(d, 'steady.csv'), steady)
        r = subprocess.run([sys.executable, os.path.join(HERE, 'clock_check.py'), '--replay', 'steady.csv'],
                           cwd=d, capture_output=True, text=True)
        check('verdict      STEADY' in r.stdout and r.returncode == 0, 'clean stretch at 0.3 ms: verdict STEADY')
        r = subprocess.run([sys.executable, os.path.join(HERE, 'clock_check.py'), '--replay', 'steady.csv',
                            '--threshold', '0.1'], cwd=d, capture_output=True, text=True)
        check('sd above 0.1 ms' in r.stdout, '--threshold 0.1 makes the same run fail')


def stem_test(check):
    """stem_check.py on fake Logic stems: 24-bit extensible headers with a bext chunk,
    then one stem a few frames long and one at the wrong sample rate."""
    import struct
    def save(path, frames, sr=44100, bits=24, extensible=True, odd_chunk=True):
        ch, block = 2, 2 * bits // 8
        if extensible:
            fmt = struct.pack('<HHIIHHHHI', 0xFFFE, ch, sr, sr * block, block, bits, 22, bits, 3) + \
                  struct.pack('<H', 1) + bytes(14)
        else:
            fmt = struct.pack('<HHIIHH', 1, ch, sr, sr * block, block, bits)
        extra = b'bext' + struct.pack('<I', 5) + b'abcde' + b'\0' if odd_chunk else b''
        data = bytes(frames * block)
        body = b'WAVE' + extra + b'fmt ' + struct.pack('<I', len(fmt)) + fmt + b'data' + struct.pack('<I', len(data)) + data
        with open(path, 'wb') as f:
            f.write(b'RIFF' + struct.pack('<I', len(body)) + body)
    print('stem-check on fake Logic stems')
    with tempfile.TemporaryDirectory() as d:
        for name in ('Bass', 'Drums', 'Pads'):
            save(os.path.join(d, name + '.wav'), 88200)
        save(os.path.join(d, 'Vox.wav'), 88200, extensible=False, odd_chunk=False)
        r = subprocess.run([sys.executable, os.path.join(HERE, 'stem_check.py'), d], capture_output=True, text=True)
        check(r.returncode == 0 and 'MATCH  (4 files, 44100 Hz, 24 bit int, 0:02.000)' in r.stdout,
              'four matching stems, extensible and plain headers: verdict MATCH')
        save(os.path.join(d, 'Pads.wav'), 88209)
        save(os.path.join(d, 'Vox.wav'), 96000, sr=48000)
        r = subprocess.run([sys.executable, os.path.join(HERE, 'stem_check.py'), d], capture_output=True, text=True)
        check(r.returncode == 1 and 'Pads.wav: 9 frames longer' in r.stdout, 'a stem 9 frames long is caught')
        check('Vox.wav: 48000 Hz, the others are 44100 Hz' in r.stdout, 'a 48 kHz stem among 44.1 kHz ones is caught')


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
    break_test(check)
    minus_test(check)
    hits_test(check)
    clock_test(check)
    stem_test(check)
    print('all good' if ok else 'SOMETHING BROKE')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
