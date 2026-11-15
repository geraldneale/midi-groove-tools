#!/usr/bin/env python3
"""midi_clean.py: clean up audio-to-MIDI output (Logic Flex Pitch, basic-pitch, etc.).

Keeps the human timing by default (no quantize). Removes a constant early/late
offset, duplicate and overlapping notes, finds the repeating bar patterns,
reports the bars that don't match any pattern so you can check them by ear,
and shows the feel (how far each 16th position sits ahead of or behind the grid).
Pure Python standard library, no installs needed.

Usage:
  python3 midi_clean.py in.mid                      # keep feel, report only fixes
  python3 midi_clean.py in.mid --strength 100       # full quantize (machine parts)
  python3 midi_clean.py in.mid --strength 50        # tighten but keep some feel
  python3 midi_clean.py in.mid --drop-short --report report.txt
  python3 midi_clean.py in.mid --strength 100 --fix-late-lows   # synth bass from Flex Pitch
  python3 midi_clean.py in.mid --no-offset --fix-late-lows --fix-outliers 0.25   # keep feel, fix stragglers

Bar numbers match Logic (first bar of the file = bar 1).
Note names use Logic's convention (middle C = C3).
"""
import argparse, struct, statistics, sys
from collections import Counter, defaultdict

NAMES = 'C C# D D# E F F# G G# A A# B'.split()
SUBS = ['', 'e', '+', 'a']


# ---------- MIDI read ----------

def read_vlq(d, i):
    v = 0
    while True:
        b = d[i]; i += 1
        v = (v << 7) | (b & 0x7F)
        if b < 0x80:
            return v, i


def read_midi(path):
    d = open(path, 'rb').read()
    if d[:4] != b'MThd':
        sys.exit(f'{path}: not a MIDI file (missing MThd header)')
    hlen = struct.unpack('>I', d[4:8])[0]
    fmt, ntrk, tpb = struct.unpack('>HHH', d[8:14])
    if tpb & 0x8000:
        sys.exit('SMPTE time division is not supported')
    p = 8 + hlen
    notes, metas, other = [], [], []
    for _ in range(ntrk):
        if d[p:p + 4] != b'MTrk':
            sys.exit(f'bad track chunk at byte {p}')
        ln = struct.unpack('>I', d[p + 4:p + 8])[0]
        i, end, p = p + 8, p + 8 + ln, p + 8 + ln
        tick, status, on = 0, None, {}
        while i < end:
            dt, i = read_vlq(d, i)
            tick += dt
            b = d[i]
            if b == 0xFF:
                typ = d[i + 1]
                ln2, j = read_vlq(d, i + 2)
                data = d[j:j + ln2]
                i = j + ln2
                if typ != 0x2F:                      # drop End of Track, rewritten later
                    metas.append((tick, typ, data))
                continue
            if b in (0xF0, 0xF7):
                ln2, j = read_vlq(d, i + 1)
                i = j + ln2
                continue
            if b & 0x80:
                status = b; i += 1
            hi, ch = status & 0xF0, status & 0x0F
            if hi in (0xC0, 0xD0):
                other.append((tick, bytes([status, d[i]]))); i += 1
                continue
            a, c = d[i], d[i + 1]; i += 2
            if hi == 0x90 and c > 0:
                on.setdefault((ch, a), []).append((tick, c))
            elif hi in (0x80, 0x90):
                if on.get((ch, a)):
                    s, v = on[(ch, a)].pop(0)
                    notes.append({'start': s, 'end': tick, 'pitch': a, 'vel': v, 'ch': ch})
            else:
                other.append((tick, bytes([status, a, c])))   # CC, pitch bend, aftertouch
    return tpb, notes, metas, other


# ---------- MIDI write (format 0, one track) ----------

def vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80); n >>= 7
    return bytes(reversed(out))


def write_midi(path, tpb, notes, metas, other):
    ev = []  # (tick, order, bytes)  order: meta 0, note-off 1, other 2, note-on 3
    for t, typ, data in metas:
        ev.append((t, 0, b'\xFF' + bytes([typ]) + vlq(len(data)) + data))
    for t, raw in other:
        ev.append((t, 2, raw))
    # Silent anchor at bar 1 (CC 110 = 0, an unused controller) so Logic and the MPC
    # import the file at bar 1 instead of trimming the empty bars before the first note.
    if notes and not any(t == 0 and len(raw) == 3 and raw[0] & 0xF0 == 0xB0 and raw[1] == 110 for t, raw in other):
        ev.append((0, 2, bytes([0xB0 | notes[0]['ch'], 110, 0])))
    for n in notes:
        ev.append((n['start'], 3, bytes([0x90 | n['ch'], n['pitch'], n['vel']])))
        ev.append((n['end'], 1, bytes([0x80 | n['ch'], n['pitch'], 0])))
    ev.sort(key=lambda e: (e[0], e[1]))
    body, last = bytearray(), 0
    for t, _, raw in ev:
        body += vlq(t - last) + raw; last = t
    body += b'\x00\xFF\x2F\x00'
    with open(path, 'wb') as f:
        f.write(b'MThd' + struct.pack('>IHHH', 6, 0, 1, tpb))
        f.write(b'MTrk' + struct.pack('>I', len(body)) + body)


# ---------- helpers ----------

def name(p):
    return f'{NAMES[p % 12]}{p // 12 - 2}'


def step_name(step):
    """16th-note step in bar (0-15) -> Ethan Hein style '1', '1e', '1+', '1a'."""
    return f'{step // 4 + 1}{SUBS[step % 4]}'


def tempo_and_meter(metas):
    bpm = meter = None
    for t, typ, data in sorted(metas, key=lambda m: m[0]):
        if typ == 0x51 and len(data) == 3 and bpm is None:
            bpm = 60e6 / int.from_bytes(data, 'big')
        if typ == 0x58 and len(data) >= 2 and meter is None:
            meter = (data[0], 2 ** data[1])
    num, den = meter or (4, 4)
    return bpm or 120.0, num, den


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('infile')
    ap.add_argument('-o', '--out', help='output .mid (default: <in>_clean.mid)')
    ap.add_argument('--grid', type=int, default=16, help='quantize grid: 8, 16 or 32 (default 16)')
    ap.add_argument('--strength', type=int, default=0, help='quantize strength %%: 0 keeps human timing (default), 100 locks to the grid')
    ap.add_argument('--min-repeat', type=int, default=3, help='bars that repeat this often count as a pattern (default 3)')
    ap.add_argument('--no-offset', action='store_true',
                    help='skip the automatic correction of a constant early/late offset (e.g. Flex Pitch lag)')
    ap.add_argument('--fix-late-lows', action='store_true',
                    help='move low notes (below C1) that sit on an "e" back onto the beat before it')
    ap.add_argument('--fix-outliers', type=float, metavar='F',
                    help='move only notes more than F of a step off the grid (e.g. 0.25) to the typical feel of their position')
    ap.add_argument('--drop-short', action='store_true', help='delete fragments shorter than 1/3 of a grid step')
    ap.add_argument('--report', help='also write the report to this text file')
    a = ap.parse_args()

    tpb, notes, metas, other = read_midi(a.infile)
    if not notes:
        sys.exit('no notes found')
    bpm, num, den = tempo_and_meter(metas)
    q = tpb * 4 // a.grid                 # ticks per grid step
    bar = tpb * 4 * num // den            # ticks per bar
    steps_per_bar = bar // q
    ms_per_tick = 60000 / bpm / tpb
    out = []
    say = out.append

    say(f'File: {a.infile}')
    say(f'Tempo {bpm:.2f} BPM, {num}/{den}, {tpb} ticks per beat, grid 1/{a.grid}, strength {a.strength}%')
    say(f'Notes in: {len(notes)}')

    # timing as recorded
    offs = [((n['start'] + q / 2) % q - q / 2) for n in notes]
    off_ms = [o * ms_per_tick for o in offs]
    far = sum(1 for o in offs if abs(o) > q / 4)
    say(f'Timing as recorded: mean {statistics.mean(off_ms):+.1f} ms '
        f'({"late" if statistics.mean(off_ms) > 0 else "early"}), spread {statistics.pstdev(off_ms):.1f} ms, '
        f'{far} notes more than 1/4 step off the grid')

    # constant offset: audio-to-MIDI often places every note a little late.
    # Shift everything by the median offset first, so quantize snaps to the right step.
    shift = 0 if a.no_offset else round(statistics.median(offs))
    if shift:
        for n in notes:
            n['start'] = max(0, n['start'] - shift); n['end'] = max(n['start'] + 1, n['end'] - shift)
        say(f'Removed a constant offset of {shift * ms_per_tick:+.1f} ms (detection lag; use --no-offset to keep it)')

    # short fragments
    short = [n for n in notes if n['end'] - n['start'] < q / 3]
    if a.drop_short:
        notes = [n for n in notes if n['end'] - n['start'] >= q / 3]
        say(f'Dropped {len(short)} short fragments (< 1/3 step)')

    # nearest grid step for every note (used for duplicates, patterns and the report,
    # whether or not the output is quantized)
    for n in notes:
        n['step'] = round(n['start'] / q)

    # feel: average distance from the grid for each 16th position in the beat
    if a.strength < 100:
        pos = defaultdict(list)
        for n in notes:
            st16 = (n['step'] * 16 // a.grid) % 4 if a.grid >= 16 else 0
            pos[st16].append((n['start'] - n['step'] * q) * ms_per_tick)
        feel = ', '.join(f'{["1", "e", "+", "a"][k]}: {statistics.mean(v):+.0f} ms ({len(v)})'
                         for k, v in sorted(pos.items()))
        say(f'Feel by position in the beat (+ = behind the grid, count): {feel}')

    # quantize starts, keep lengths
    s = a.strength / 100
    for n in notes if s else []:
        ln = max(n['end'] - n['start'], q // 4)
        target = round(n['start'] / q) * q
        n['start'] = round(n['start'] + (target - n['start']) * s)
        n['end'] = n['start'] + ln

    # late low notes: audio-to-MIDI detects low notes late, and quantize can snap a
    # note meant for a beat onto the "e" after it. Move it back if that beat is free.
    if a.fix_late_lows:
        e_step = a.grid // 16 if a.grid >= 16 else 0      # grid steps per 16th
        beat = a.grid // 4
        taken = {(n['step'], n['pitch']) for n in notes}
        lows = {}
        for n in notes:
            if n['pitch'] < 36 and e_step and n['step'] % beat == e_step and (n['step'] - e_step, n['pitch']) not in taken:
                end = n['end']
                n['step'] -= e_step
                n['start'] = n['step'] * q
                n['end'] = max(end, n['start'] + 2 * q * e_step)
                taken.add((n['step'], n['pitch']))
                lows.setdefault(n['step'] // steps_per_bar + 1, []).append(name(n['pitch']))
        if lows:
            say(f'Moved {sum(len(v) for v in lows.values())} late low notes onto the beat, bars: '
                + ', '.join(str(b) for b in sorted(lows)))

    # outliers: notes far off the grid (missed attacks, late detection). Move just those
    # to where the other notes at the same 16th position typically sit, so the feel stays.
    if a.fix_outliers:
        lim = a.fix_outliers * q
        per = a.grid // 4 if a.grid >= 4 else 1
        good = defaultdict(list)
        for n in notes:
            o = n['start'] - n['step'] * q
            if abs(o) <= lim:
                good[n['step'] % per].append(o)
        typ = {k: round(statistics.median(v)) for k, v in good.items()}
        fixed = defaultdict(int)
        for n in notes:
            o = n['start'] - n['step'] * q
            if abs(o) > lim:
                ln = n['end'] - n['start']
                n['start'] = max(0, n['step'] * q + typ.get(n['step'] % per, 0))
                n['end'] = n['start'] + ln
                fixed[n['step'] // steps_per_bar + 1] += 1
        say(f'Moved {sum(fixed.values())} notes more than {a.fix_outliers:g} of a step off the grid '
            f'to the typical feel of their position')

    # duplicates: same pitch, same start
    notes.sort(key=lambda n: (n['step'], n['pitch'], -n['vel']))
    seen, kept, dups = set(), [], []
    for n in notes:
        k = (n['step'], n['pitch'], n['ch'])
        if k in seen:
            dups.append(n); continue
        seen.add(k); kept.append(n)
    notes = kept

    # overlaps: same pitch, a note still sounding when the next one starts -> trim it
    trimmed = 0
    byp = defaultdict(list)
    for n in notes:
        byp[(n['ch'], n['pitch'])].append(n)
    for lst in byp.values():
        lst.sort(key=lambda n: n['start'])
        for x, y in zip(lst, lst[1:]):
            if x['end'] > y['start']:
                x['end'] = max(y['start'] - 1, x['start'] + 1); trimmed += 1
    say(f'Removed {len(dups)} duplicate notes, trimmed {trimmed} overlapping notes. Notes out: {len(notes)}')

    # bars and patterns
    bars = defaultdict(list)
    for n in notes:
        b = n['step'] // steps_per_bar + 1
        bars[b].append((n['step'] % steps_per_bar, n['pitch']))
    sig = {b: frozenset(v) for b, v in bars.items()}
    counts = Counter(sig.values())
    patterns = [p for p, c in counts.most_common() if c >= a.min_repeat]
    # label patterns by first appearance
    label, nxt = {}, 0
    for b in sorted(sig):
        if sig[b] in patterns and sig[b] not in label:
            label[sig[b]] = f'P{nxt + 1}'; nxt += 1

    def describe(p):
        roots = sorted({pt for _, pt in p})
        return ' '.join(name(x) for x in roots)

    say('')
    say(f'Patterns found (repeat {a.min_repeat}+ times):')
    for p in sorted(label, key=lambda p: int(label[p][1:])):
        say(f'  {label[p]}: {counts[p]} bars, {len(p)} notes, pitches {describe(p)}')

    # patterns that use the same pitches but differ slightly (e.g. a section played twice)
    labeled = sorted(label, key=lambda p: int(label[p][1:]))
    for i, p1 in enumerate(labeled):
        for p2 in labeled[i + 1:]:
            d = p1 ^ p2
            if 0 < len(d) <= 4:
                mis = ', '.join(f'{name(pt)} on {step_name(st * 16 // a.grid)}' for st, pt in sorted(p1 - p2))
                ext = ', '.join(f'{name(pt)} on {step_name(st * 16 // a.grid)}' for st, pt in sorted(p2 - p1))
                say(f'  note: {label[p2]} is {label[p1]} with ' + '; '.join(x for x in [f"no {mis}" if mis else "", f"added {ext}" if ext else ""] if x))

    # bar map, 8 bars per row
    first, last = min(sig), max(sig)
    say('')
    say('Bar map (8 bars per row, . = silent, ? = no pattern match):')
    for r in range((first - 1) // 8 * 8 + 1, last + 1, 8):
        cells = []
        for b in range(r, r + 8):
            if b not in sig:
                cells.append(' .  ')
            elif sig[b] in label:
                cells.append(f'{label[sig[b]]:<4}')
            else:
                cells.append(' ?  ')
        say(f'  {r:4d}-{r + 7:<4d} ' + ' '.join(cells))

    # flagged bars
    say('')
    say('Bars to check by ear:')
    flagged = 0
    for b in sorted(sig):
        if sig[b] in label:
            continue
        flagged += 1
        best = min(patterns, key=lambda p: len(p ^ sig[b])) if patterns else None
        if best is None or len(best ^ sig[b]) > max(len(best), len(sig[b])) // 2:
            say(f'  bar {b}: unique ({len(sig[b])} notes, pitches {describe(sig[b])}), no close pattern')
            continue
        missing = sorted(best - sig[b])
        extra = sorted(sig[b] - best)
        parts = []
        if missing:
            parts.append('missing ' + ', '.join(f'{name(pt)} on {step_name(st * 16 // a.grid)}' for st, pt in missing))
        if extra:
            parts.append('extra ' + ', '.join(f'{name(pt)} on {step_name(st * 16 // a.grid)}' for st, pt in extra))
        say(f'  bar {b}: closest to {label[best]}; ' + '; '.join(parts))
    if not flagged:
        say('  none, every bar matches a pattern')
    if short and not a.drop_short:
        say('')
        say(f'Short fragments kept ({len(short)}); rerun with --drop-short to delete them:')
        for n in short[:40]:
            say(f'  bar {n["start"] // bar + 1}: {name(n["pitch"])} near {step_name((n["start"] % bar) // (tpb // 4))}')
        if len(short) > 40:
            say(f'  ... and {len(short) - 40} more')

    outfile = a.out or a.infile.rsplit('.', 1)[0] + '_clean.mid'
    write_midi(outfile, tpb, notes, metas, other)
    say('')
    say(f'Wrote {outfile}')
    text = '\n'.join(out)
    print(text)
    if a.report:
        open(a.report, 'w').write(text + '\n')


if __name__ == '__main__':
    main()
