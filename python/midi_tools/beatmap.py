#!/usr/bin/env python3
"""beatmap.py: find the beats in a song that drifts, and stretch it onto a steady grid.

  python3 beatmap.py song.wav --bpm 111                 # report: tempo bar by bar, steady or drifting,
                                                        # and for a steady song the Logic tempo and region position
  python3 beatmap.py song.wav --bpm 111 --stretch       # write song_111bpm.wav, steady at 111
  python3 beatmap.py song_111bpm.wav --bpm 111 --check  # prove the result is on the grid

The stretched file has a 2-bar lead-in: import it at bar 1 in Logic, Smart Tempo Off,
and the song's first beat lands on bar 3. --lead-bars changes that.

Beats are found from the audio itself (no Logic Smart Tempo). --bpm is a hint, so the
tracker doesn't lock onto half or double time. If the first detected beat isn't the
real first downbeat, give it with --first (seconds into the file).

Needs numpy, and the rubberband command line tool for --stretch:
  Mac:   brew install rubberband
  Linux: sudo apt install rubberband-cli
Input must be WAV. From an m4a on the Mac: afconvert -f WAVE -d LEI24 song.m4a song.wav
"""
import argparse, os, shutil, statistics, subprocess, sys, tempfile, wave
import numpy as np

HOP = 256


# ---------- audio ----------

def read_wav(path):
    """(raw bytes, channels, sample width, rate, frames) exactly as stored, plus mono float."""
    with wave.open(path, 'rb') as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = (np.pad(b, ((0, 0), (1, 0))).view('<i4').ravel() >> 8).astype(np.float32) / 2 ** 23
    else:
        x = np.frombuffer(raw, {1: np.uint8, 2: '<i2', 4: '<i4'}[width]).astype(np.float32)
        x /= {1: 128, 2: 2 ** 15, 4: 2 ** 31}[width]
    return raw, ch, width, sr, n, x.reshape(-1, ch).mean(axis=1)


def write_wav(path, raw, ch, width, sr):
    with wave.open(path, 'wb') as w:
        w.setnchannels(ch); w.setsampwidth(width); w.setframerate(sr)
        w.writeframes(raw)


# ---------- beats ----------

def onset_envelope(x, sr, low=False):
    """How much new sound starts in each 256-sample frame (spectral flux)."""
    step = max(1, sr // 22050)                     # work at about 22 kHz
    x, fs = x[::step], sr / step
    n_fft = 2048
    pad = np.pad(x, (n_fft // 2, n_fft // 2))
    n = 1 + (len(pad) - n_fft) // HOP
    fr = np.lib.stride_tricks.as_strided(pad, (n, n_fft), (pad.strides[0] * HOP, pad.strides[0]))
    win = np.hanning(n_fft).astype(np.float32)
    env = np.empty(n, np.float32)
    top = int((200 if low else 8000) * n_fft / fs)
    prev = None
    for i in range(0, n, 2048):                    # in chunks, to keep memory small
        m = np.log1p(100 * np.abs(np.fft.rfft(fr[i:i + 2048] * win, axis=1))[:, 1:top])
        d = np.diff(np.vstack([m[:1] if prev is None else prev, m]), axis=0)
        env[i:i + len(m)] = np.maximum(d, 0).sum(axis=1)
        prev = m[-1:]
    env -= np.convolve(env, np.ones(32) / 32, 'same')   # remove slow loudness changes
    env = np.maximum(env, 0)
    return env / (env.std() + 1e-9), HOP * step / sr


def track_beats(env, dt, bpm, tight=100.0, period=None, hold=0.0):
    """Best chain of beats: each beat sits on strong onsets, and each gap stays near
    the beat length (dynamic programming, after Ellis 2007). With `period` (frames per beat
    at each frame) the expected gap follows the song's own tempo curve instead of --bpm.
    `hold` adds a cost for every bit a gap differs from the expected one, so sliding half a
    beat costs the same however slowly it's done (the squared `tight` cost alone lets the
    chain creep onto the off-beats a little per beat through a break with no kick)."""
    P = 60 / bpm / dt
    if period is None:
        period = np.full(len(env), P)
    lo, hi = int(period.min() * 0.8), int(period.max() * 1.25) + 1
    taus = np.arange(lo, hi)
    score = env.astype(np.float64).copy()
    back = np.full(len(env), -1)
    for t in range(lo, len(env)):
        lg = np.log(taus / period[t])
        prev = score[t - taus] - tight * lg ** 2 - hold * np.abs(lg)
        k = np.argmax(prev)
        if prev[k] > 0:
            score[t] += prev[k]; back[t] = t - taus[k]
    t = len(env) - 1 - int(np.argmax(score[::-1][:int(P)]))
    beats = []
    while t >= 0:
        beats.append(t); t = back[t]
    beats = np.array(beats[::-1], float)
    # the chain also runs through silence at the start and end: trim beats with no onset under them
    strength = env[beats.astype(int)]
    ok = np.where(strength > 0.3 * np.median(strength))[0]
    beats = beats[ok[0]:ok[-1] + 1]
    # sub-frame position: centre of the onset peak
    out = []
    for b in beats.astype(int):
        if 0 < b < len(env) - 1:
            a, c, d = env[b - 1], env[b], env[b + 1]
            den = a - 2 * c + d
            out.append(b + (np.clip(0.5 * (a - d) / den, -0.5, 0.5) if den < 0 else 0))   # a flat peak can't move it more than half a frame
        else:
            out.append(b)
    return np.array(out) * dt


def tempo_curve(beats, n, dt, half=8):
    """Frames per beat at every frame, from the median gap of the nearby beats."""
    g = np.diff(beats)
    loc = np.array([np.median(g[max(0, i - half):i + half + 1]) for i in range(len(g))])
    mid = (beats[:-1] + beats[1:]) / 2
    return np.interp(np.arange(n) * dt, mid, loc) / dt


def refine(x, sr, beats):
    """Move each beat to the attack in the waveform: where the peak level (5 ms windows)
    first climbs halfway up, looking from 40 ms before to 80 ms after the onset frame."""
    w = max(1, int(0.005 * sr))
    out = []
    for b in beats:
        a0, a1 = int((b - 0.04) * sr), int((b + 0.08) * sr)
        seg = np.abs(x[max(0, a0):max(0, a1)])
        if len(seg) < 4 * w:
            out.append(b); continue
        env = seg[:len(seg) // w * w].reshape(-1, w).max(axis=1)
        k = int(np.argmax(env >= env.min() + 0.5 * (env.max() - env.min())))
        out.append(max(0, a0) / sr + k * w / sr)
    return np.array(out)


def find_beats(x, sr, bpm, low=False, first=None):
    # the beat is carried by the low end (kick, bass); hats and shakers on the off-beats
    # can pull a full-band tracker half a beat off, so the low band counts double
    lo, dt = onset_envelope(x, sr, True)
    env = lo if low else 2 * lo + onset_envelope(x, sr, False)[0]
    if first is not None:
        # Anchor the count on the given downbeat. Dropping the beats before it is not enough:
        # a crowd clapping in time before the band comes in (The In Crowd, live) set the chain's
        # tempo and phase, and it carried on half a beat off for 25 beats. So ignore everything
        # before it, and make the chain start exactly there.
        k = int(round(first / dt))
        env = env.copy()
        env[:max(0, k - 1)] = 0
        env[k] = env.max() * 50 + 1
    # pass 1, loose: follows drift, but where a section has no kick it can slide onto the
    # off-beats for a few bars (Lebanese Blonde's break did, for 8 bars)
    loose = track_beats(env, dt, bpm, 100.0)
    # pass 2, held to the song's own tempo curve from pass 1: sliding half a beat now costs
    # more than a break's off-beats can pay back, while the drift is already in the expected gap
    rough = track_beats(env, dt, bpm, 100.0, tempo_curve(loose, len(env), dt), hold=300.0)
    fine = refine(x, sr, rough)
    # the onset frames sit a fixed amount early; where no clear attack is near the beat
    # (no kick on it), keep the rough position plus that fixed amount instead
    lag = np.median(fine - rough)
    beats = np.where(np.abs(fine - (rough + lag)) <= 0.015, fine, rough + lag)
    if first is not None:                          # drop anything before the given first beat
        beats = beats[beats >= first - 0.05]
        if len(beats) and abs(beats[0] - first) < 0.1:
            beats[0] = first                       # it was given, so it is exact
    return beats


def beat_numbers(beats, half=8):
    """Count beats: how many beat lengths each gap spans, using the local median gap.
    A missed beat makes a gap of 2; an extra beat squeezed in makes a gap of 0 and is dropped."""
    d = np.diff(beats)
    k = [0]
    for i, g in enumerate(d):
        p = np.median(d[max(0, i - half):i + half + 1])
        k.append(k[-1] + max(0, int(round(g / p))))
    return np.array(k)


def bar_report(beats, per_bar=4, cols=8):
    bpms = [60 * per_bar / (beats[i + per_bar] - beats[i]) for i in range(0, len(beats) - per_bar, per_bar)]
    lines = []
    for r in range(0, len(bpms), cols):
        row = '  '.join(f'{v:6.1f}' for v in bpms[r:r + cols])
        lines.append(f'  bars {r + 1:>3}-{min(r + cols, len(bpms)):<3} {row}')
    return bpms, lines


def logic_position(first, bpm, lead_bars):
    """Where the region must start in Logic (bar beat 16th tick) so the beat at `first`
    seconds into the file lands on bar lead_bars+1. Logic counts from 1, 240 ticks per 16th."""
    t = round(((lead_bars * 4) - first * bpm / 60) * 960)
    if t < 0:
        return f'1 1 1 1 and move it later; the intro is longer than {lead_bars} bars, try --lead-bars {lead_bars + (-t + 3839) // 3840}'
    return f'{t // 3840 + 1} {t % 3840 // 960 + 1} {t % 960 // 240 + 1} {t % 240 + 1}'


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wav')
    ap.add_argument('--bpm', type=float, required=True, help='rough tempo of the song (a hint) and the steady target tempo')
    ap.add_argument('--stretch', action='store_true', help='write <name>_<bpm>bpm.wav on a steady grid (needs rubberband)')
    ap.add_argument('--check', action='store_true', help='measure how far each beat is from a steady --bpm grid')
    ap.add_argument('--first', type=float, help='time in seconds of the first downbeat, if the tracker picks the wrong one')
    ap.add_argument('--lead-bars', type=int, default=2, help='bars of lead-in before the first beat in the output (default 2)')
    ap.add_argument('--low', action='store_true', help='follow only the low end (below 200 Hz); try it if beats come out wrong')
    ap.add_argument('-o', '--out', help='output file for --stretch')
    a = ap.parse_args()

    raw, ch, width, sr, n, x = read_wav(a.wav)
    found = find_beats(x, sr, a.bpm, a.low, a.first)
    if len(found) < 8:
        sys.exit('found too few beats; check --bpm')
    beats = found
    k = beat_numbers(beats)                        # a missed beat counts as 2, so the grid can't slip
    T = 60 / a.bpm
    print(f'{a.wav}: {n / sr:.1f} s, {k[-1] + 1} beats, first beat at {beats[0]:.3f} s')
    full = np.interp(np.arange(k[-1] + 1), k, beats)     # every beat, misread ones filled in from their neighbours

    if a.check:
        off = np.median(beats - k * T)
        dev = (beats - (off + k * T)) * 1000
        print(f'Distance from a steady {a.bpm:g} BPM grid: median {np.median(np.abs(dev)):.1f} ms, '
              f'95% within {np.percentile(np.abs(dev), 95):.1f} ms, worst {np.abs(dev).max():.1f} ms')
        bad = [(kk // 4 + 1, d) for kk, d in zip(k, dev) if abs(d) > 20]
        if bad:
            print('Beats more than 20 ms off (bar counted from the first beat): '
                  + ', '.join(f'bar {b} {d:+.0f} ms' for b, d in bad[:12]) + (' ...' if len(bad) > 12 else ''))
            print('A single odd beat between good ones is usually a beat with no kick on it, '
                  'where the beat finder guessed. Listen there before worrying.')
        else:
            print('Every beat is within 20 ms of the grid.')
        return

    bpms, lines = bar_report(full)
    print(f'Tempo bar by bar (counted from the first beat): {min(bpms):.1f} to {max(bpms):.1f}, '
          f'median {statistics.median(bpms):.1f}')
    print('\n'.join(lines))
    if not a.stretch:
        q = max(1, len(bpms) // 4)
        early, late = statistics.median(bpms[:q]), statistics.median(bpms[-q:])
        # straight-line fit through every good beat: steadier than using just the first and last
        slope, first = np.polyfit(k, beats, 1)
        avg = 60 / slope
        if abs(early - late) / avg > 0.01:
            print(f'DRIFTS: about {early:.1f} at the start, {late:.1f} at the end. '
                  f'Add --stretch to write a steady version at --bpm.')
            return
        print(f'Steady: bar-to-bar changes are feel, not drift (start {early:.1f}, end {late:.1f}).')
        print(f'Logic: tempo {avg:.2f}, region position {logic_position(first, avg, a.lead_bars)} '
              f'(no stretching; first beat on bar {a.lead_bars + 1})')
        return

    rb = shutil.which('rubberband')
    if not rb:
        sys.exit('rubberband not found. Mac: brew install rubberband   Linux: sudo apt install rubberband-cli')

    # 1. trim or pad the start so the first beat sits exactly lead-bars in
    lead = a.lead_bars * 4 * T
    fsz = ch * width
    shift = int(round((lead - beats[0]) * sr))     # >0: add silence, <0: cut
    if shift >= 0:
        raw2 = bytes(shift * fsz) + raw
    else:
        raw2 = raw[-shift * fsz:]
    n2 = len(raw2) // fsz
    src = beats + shift / sr                       # beats in the prepared file; first = lead

    # 2. time map: every good beat goes to its place on the steady grid (misread ones are skipped,
    #    so the audio between their neighbours is stretched evenly)
    tgt = lead + k * T
    pairs = [(0, 0)] + [(int(round(s * sr)), int(round(t * sr))) for s, t in zip(src, tgt)]
    tail = (n2 / sr - src[-1]) * T / np.median(np.diff(src[-9:]))   # after the last beat: its local tempo
    end_t = int(round((tgt[-1] + tail) * sr))
    pairs.append((n2, end_t))
    pairs = [p for i, p in enumerate(pairs) if i == 0 or (p[0] > pairs[i - 1][0] and p[1] > pairs[i - 1][1])]

    out = a.out or os.path.splitext(a.wav)[0] + f'_{a.bpm:g}bpm.wav'
    with tempfile.TemporaryDirectory() as d:
        prep, tmap = os.path.join(d, 'in.wav'), os.path.join(d, 'map.txt')
        write_wav(prep, raw2, ch, width, sr)
        with open(tmap, 'w') as f:
            f.writelines(f'{s} {t}\n' for s, t in pairs)
        base = [rb, '-q', '--timemap', tmap, '--time', f'{end_t / n2:.9f}']
        # R2 engine by default: tested with rubberband 4.0, the R3 (-3) engine drifted about 20 ms
        # off the time map by the end of a song; R2 stayed within a few ms. diag_stretch.py compares them.
        r3 = os.environ.get('BEATMAP_RB_ENGINE', 'r2') == 'r3'
        r = subprocess.run(base + (['-3'] if r3 else []) + [prep, out], capture_output=True, text=True)
        if r.returncode:
            sys.exit('rubberband failed:\n' + r.stderr)
    print(f'Wrote {out}')
    print(f'Logic: tempo {a.bpm:g}, region position 1 1 1 1 (first beat on bar {a.lead_bars + 1})')
    print(f'Check it: beatmap "{out}" --bpm {a.bpm:g} --check')


if __name__ == '__main__':
    main()
