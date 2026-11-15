#!/usr/bin/env python3
"""midi_align.py: move each MIDI note's start to the attack in the audio it came from.

Audio-to-MIDI (Logic Flex Pitch, basic-pitch) finds pitch well but starts notes
late, because pitch takes time to recognize. Attacks are easy to find. This keeps
each note's pitch and slides its start to the nearest attack in the stem.

Usage:
  python3 midi_align.py bass.mid bass_stem.wav
  python3 midi_align.py bass.mid bass_stem.wav -o bass_aligned.mid --window 60

Attacks in the audio with no MIDI note get a note added, pitch read from the audio
(--no-fill to skip). The bars where notes were added are listed so you can check them.

The WAV is assumed to start at bar 1. If its region starts later in Logic,
give that start with --start, as Logic shows it in the Event List (e.g. "1 1 2 117").
Needs numpy (pip install numpy). Keep it in the same folder as midi_clean.py.
"""
import argparse, os, statistics, sys, wave
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from midi_clean import read_midi, write_midi, tempo_and_meter


def load_wav(path):
    with wave.open(path, 'rb') as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 3:  # 24-bit: pad to 32-bit
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = (np.pad(b, ((0, 0), (1, 0))).view('<i4').ravel() >> 8).astype(np.float32)
    else:
        x = np.frombuffer(raw, {1: np.uint8, 2: '<i2', 4: '<i4'}[width]).astype(np.float32)
    return x.reshape(-1, ch).mean(axis=1), sr


def frames(x, win, hop):
    n = 1 + (len(x) - win) // hop
    return np.lib.stride_tricks.as_strided(x, (n, win), (x.strides[0] * hop, x.strides[0]))


def onsets(x, sr, gap=0.06):
    """Attack times in seconds. Coarse: where loudness jumps. Fine: where the
    envelope (peak level over one low-note cycle) first climbs halfway up."""
    hop, win = 64, 1024
    e = np.log(np.sqrt((frames(x, win, hop) ** 2).mean(axis=1)) + 1e-5)
    k = int(0.012 * sr / hop)
    rise = np.concatenate([np.zeros(k), np.maximum(0, e[k:] - e[:-k])])
    thr = np.median(rise) + 4 * np.median(np.abs(rise - np.median(rise))) + 0.2
    mind = int(gap * sr / hop)
    peaks = []
    for i in range(1, len(rise) - 1):
        if rise[i] >= thr and rise[i] >= rise[i - 1] and rise[i] > rise[i + 1]:
            if peaks and i - peaks[-1] < mind:
                if rise[i] > rise[peaks[-1]]:
                    peaks[-1] = i
                continue
            peaks.append(i)
    cyc, out = int(0.03 * sr), []
    for i in peaks:
        c = i * hop + win // 2
        a, b = max(0, c - int(0.06 * sr)), min(len(x) - cyc, c + int(0.03 * sr))
        env = frames(np.ascontiguousarray(np.abs(x[a:b + cyc])), cyc, 32).max(axis=1)
        j = np.argmax(env >= env.min() + 0.5 * (env.max() - env.min()))
        out.append((a + j * 32 + cyc) / sr)
    return np.array(out)


def pitch_at(x, sr, t, fmin=28.0, fmax=500.0):
    """MIDI pitch of the note starting at t seconds (YIN on ~90 ms just after the attack),
    or None when the sound has no clear pitch (drum bleed, noise)."""
    step = max(1, sr // 12000)                       # work at about 12 kHz
    i0 = int((t + 0.015) * sr)                       # skip the click of the attack
    seg = x[i0:i0 + int(0.2 * sr):step]
    fs = sr / step
    W, tmax, tmin = int(0.085 * fs), int(fs / fmin), int(fs / fmax)
    if len(seg) < W + tmax or not seg.any():
        return None
    seg = seg - seg.mean()
    d = np.array([np.sum((seg[:W] - seg[k:k + W]) ** 2) for k in range(tmax + 1)])
    cm = d[1:] * np.arange(1, tmax + 1) / np.maximum(np.cumsum(d[1:]), 1e-12)
    cm = np.concatenate([[1.0], cm])
    below = np.where(cm[tmin:] < 0.2)[0]
    if not len(below):
        return None
    k = tmin + below[0]
    while k + 1 <= tmax and cm[k + 1] < cm[k]:      # walk down to the bottom of the dip
        k += 1
    if 0 < k < tmax:                                  # parabolic fine-tune
        a_, b_, c_ = cm[k - 1], cm[k], cm[k + 1]
        k = k + 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_ or 1)
    f = fs / k
    return int(round(69 + 12 * np.log2(f / 440)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('midi')
    ap.add_argument('wav')
    ap.add_argument('-o', '--out', help='output .mid (default: <midi>_aligned.mid)')
    ap.add_argument('--window', type=float, default=25, help='max ms a note may move after the lag is removed (default 25)')
    ap.add_argument('--no-fill', action='store_true', help="don't add notes for attacks the MIDI missed")
    ap.add_argument('--start', default='1 1 1 1', help='where the WAV region starts in Logic: "bar beat div tick" (default 1 1 1 1)')
    a = ap.parse_args()

    tpb, notes, metas, other = read_midi(a.midi)
    bpm = tempo_and_meter(metas)[0]
    sec_per_tick = 60 / bpm / tpb
    x, sr = load_wav(a.wav)
    bar_, beat_, div_, tick_ = (int(v) for v in a.start.split())
    num = 4                                             # Logic: 4 divisions (16ths) per beat, 240 ticks per 16th
    start_beats = (bar_ - 1) * num + (beat_ - 1) + (div_ - 1) / 4 + (tick_ - 1) / 960
    att_wav = onsets(x, sr)                              # attack times in the WAV
    att = att_wav + start_beats * 60 / bpm               # attack times on the project timeline
    if not len(att):
        sys.exit('no attacks found in the audio')

    starts = np.array([n['start'] * sec_per_tick for n in notes])
    nearest = lambda t: att[np.argmin(np.abs(att - t))]
    # typical delay of the MIDI behind the audio: the most common note-to-attack distance
    d = np.concatenate([t - att[np.abs(att - t) < 0.15] for t in starts])
    hist, edges = np.histogram(d, bins=np.arange(-0.15, 0.1501, 0.005))
    lag = edges[np.argmax(hist)] + 0.0025

    moved, kept, shifts = 0, 0, []
    for n, t in zip(notes, starts):
        target = nearest(t - lag)
        if abs(target - (t - lag)) * 1000 <= a.window:
            new = round(target / sec_per_tick); moved += 1
        else:
            new = round((t - lag) / sec_per_tick); kept += 1   # no attack close by: just remove the lag
        shifts.append((n['start'] - new) * sec_per_tick * 1000)
        ln = n['end'] - n['start']
        n['start'] = max(0, new)
        n['end'] = n['start'] + ln

    # missed notes: an attack in the audio with no MIDI note on it means the
    # audio-to-MIDI step didn't hear that note. Read its pitch from the audio and add it.
    added = []
    if not a.no_fill:
        s16 = 60 / bpm / 4
        have = np.sort(np.array([n['start'] * sec_per_tick for n in notes]))
        vel = int(statistics.median(n['vel'] for n in notes))
        ch = statistics.mode(n['ch'] for n in notes)
        lo = min(n['pitch'] for n in notes) - 12
        hi = max(n['pitch'] for n in notes) + 12
        for i, (t, tw) in enumerate(zip(att, att_wav)):
            j = np.searchsorted(have, t)
            near = min(abs(have[j - 1] - t) if j else 9, abs(have[j] - t) if j < len(have) else 9)
            if near < 0.08:                    # 16ths are 120 ms apart at 125 BPM: anything closer is that note
                continue
            p = pitch_at(x, sr, tw)
            if p is None or not lo <= p <= hi:
                continue
            tk = t / sec_per_tick
            if any(n['pitch'] == p and n['start'] < tk < n['end'] + 0.02 / sec_per_tick for n in notes):
                continue                       # that pitch is still sounding: a bump in the same note, not a new one
            nxt = have[j] if j < len(have) else t + 2 * s16       # next note in the MIDI
            same = [n['end'] - n['start'] for n in notes if n['pitch'] == p] or [round(1.8 * s16 / sec_per_tick)]
            end = min(nxt - 0.005, t + statistics.median(same) * sec_per_tick)
            st = round(t / sec_per_tick)
            # velocity from the notes around it (within a bar), so fades stay fades
            bar_t = 4 * 60 / bpm / sec_per_tick
            near_v = [n['vel'] for n in notes if abs(n['start'] - st) <= bar_t]
            v = int(statistics.median(near_v)) if near_v else vel
            notes.append({'start': st, 'end': max(st + 1, round(end / sec_per_tick)), 'pitch': p, 'vel': v, 'ch': ch})
            added.append((t, p))
            have = np.sort(np.append(have, t))

    out = a.out or a.midi.rsplit('.', 1)[0] + '_aligned.mid'
    write_midi(out, tpb, notes, metas, other)
    print(f'{len(att)} attacks found in {a.wav}, {len(notes)} notes in {a.midi}')
    print(f'MIDI was typically {lag * 1000:.0f} ms behind the audio')
    print(f'{moved} notes moved to their attack, {kept} had no attack within {a.window:.0f} ms (lag removed only)')
    print(f'Shift per note: median {statistics.median(shifts):.0f} ms, range {min(shifts):.0f} to {max(shifts):.0f} ms')
    if added:
        nm = 'C C# D D# E F F# G G# A A# B'.split()
        bars = defaultdict(list)
        for t, p in added:
            bars[int(t / (240 / bpm)) + 1].append(f'{nm[p % 12]}{p // 12 - 2}')
        print(f'{len(added)} notes added where the audio has an attack but the MIDI had no note:')
        for b in sorted(bars):
            print(f'  bar {b}: {" ".join(bars[b])}')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
