#!/usr/bin/env python3
"""midi-clock-check: how steady is a MIDI clock?

Listens on one input port, timestamps every clock tick (24 per beat), and reports
tempo and tick jitter about once a second, then a summary with a verdict.

  midi-clock-check                          list input ports
  midi-clock-check H12 --seconds 60         listen for a minute on the first port named *H12*
  midi-clock-check MPC --csv mpc.csv        also save every tick time
  midi-clock-check --replay mpc.csv         analyze a saved run again (no MIDI needed)
  midi-clock-check H12 --threshold 0.5      stricter verdict (default 1 ms)

Jitter is each tick interval minus the typical interval around it (median of the
48 nearby ticks), so a tempo change is not counted as jitter. It is measured on
intervals, the same way midi-watch does: if each tick lands up to x ms early or
late, intervals vary about 1.4 times as much. Tempo is measured per beat. At 120 BPM a tick
should arrive every 20.83 ms. Gaps across Stop/Start are not counted. A gap of about
two ticks or more counts as missing ticks; an interval under half the typical one
counts as a doubled tick.

Jitter is measured on the receiving computer, so it includes USB and OS timing:
the source is at least that steady. To see whether a router adds anything, run it
once on the source's direct USB port and once through the router, and compare.
"""
import argparse, csv, sys, threading, time
import numpy as np

PPQN = 24
NEAR = 48                 # ticks either side used for the typical interval


def segments(events):
    """Split (time_s, kind) events into runs of clock tick times between transport events."""
    runs, cur = [], []
    for t, kind in events:
        if kind == 'clock':
            cur.append(t)
        elif kind in ('start', 'stop', 'continue', 'songpos'):
            if len(cur) > 1:
                runs.append(cur)
            cur = []
    if len(cur) > 1:
        runs.append(cur)
    return [np.array(r) for r in runs]


def analyze(events):
    """Stats for a list of (time_s, kind) events. Times in seconds, results in ms."""
    devs, ivals, bpms = [], [], []
    missing = doubled = ticks = 0
    for r in segments(events):
        ticks += len(r)
        iv = np.diff(r) * 1000
        k = min(NEAR, len(iv))
        # typical interval around each tick: rolling median
        pad = np.pad(iv, (k, k), mode='edge')
        typ = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(iv))])
        big = iv > 1.5 * typ
        small = iv < 0.5 * typ
        missing += int(np.sum(np.round(iv[big] / typ[big]) - 1))
        doubled += int(np.sum(small))
        good = ~(big | small)
        devs.append(iv[good] - typ[good])
        ivals.append(iv[good])
        # tempo per beat: time across 24 clean intervals, so a stray tick doesn't skew it
        for i in range(0, len(iv) - PPQN + 1, PPQN):
            if good[i:i + PPQN].all():
                bpms.append([60000 / iv[i:i + PPQN].sum()])
    if not devs or not sum(len(d) for d in devs):
        return None
    d, iv = np.concatenate(devs), np.concatenate(ivals)
    bpm = np.concatenate(bpms) if bpms else 60000 / (np.array([np.median(iv)]) * PPQN)
    a = np.abs(d)
    return {
        'ticks': ticks, 'bpm': float(np.mean(bpm)), 'bpm_min': float(np.min(bpm)),
        'bpm_max': float(np.max(bpm)), 'interval': float(np.mean(iv)),
        'sd': float(np.std(d)), 'p95': float(np.percentile(a, 95)), 'worst': float(a.max()),
        'missing': missing, 'doubled': doubled,
    }


def line(s):
    return (f"{s['bpm']:7.2f} BPM  tick {s['interval']:6.3f} ms  jitter sd {s['sd']:.3f}"
            f"  p95 {s['p95']:.3f}  worst {s['worst']:.3f} ms"
            f"  missing {s['missing']}  doubled {s['doubled']}")


def summary(s, threshold):
    if s is None:
        print('\nNo clock ticks received. Is the source sending clock, and is it playing?')
        return 1
    print('\nSummary')
    print(f"  ticks        {s['ticks']}")
    print(f"  tempo        {s['bpm']:.2f} BPM average, {s['bpm_min']:.2f} to {s['bpm_max']:.2f}")
    print(f"  tick         {s['interval']:.3f} ms average")
    print(f"  jitter       sd {s['sd']:.3f} ms, 95% within {s['p95']:.3f} ms, worst {s['worst']:.3f} ms")
    print(f"  missing      {s['missing']}    doubled {s['doubled']}")
    ok = s['sd'] <= threshold and not s['missing'] and not s['doubled']
    why = '' if ok else (' (ticks missing or doubled)' if s['sd'] <= threshold
                         else f" (sd above {threshold:g} ms)")
    print(f"  verdict      {'STEADY' if ok else 'NOT STEADY'}{why}")
    return 0 if ok else 1


def read_csv(path):
    with open(path, newline='') as f:
        return [(float(r['time_ms']) / 1000, r['kind']) for r in csv.DictReader(f)]


def write_csv(path, events):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_ms', 'kind'])
        for t, kind in events:
            w.writerow([f'{t * 1000:.4f}', kind])


def listen(name, seconds, threshold):
    import mido
    events, lock = [], threading.Lock()
    t0 = time.perf_counter()

    def on_msg(msg):
        now = time.perf_counter() - t0
        if msg.type in ('clock', 'start', 'stop', 'continue', 'songpos'):
            with lock:
                events.append((now, msg.type))
            if msg.type != 'clock':
                extra = f' pos={msg.pos}' if msg.type == 'songpos' else ''
                print(f'{now * 1000:10.2f} ms  {msg.type}{extra}', flush=True)

    print(f'Listening on: {name}   (Ctrl+C to stop)\n', flush=True)
    with mido.open_input(name, callback=on_msg):
        try:
            end = time.perf_counter() + seconds if seconds else None
            while end is None or time.perf_counter() < end:
                time.sleep(1)
                with lock:
                    recent = [e for e in events if e[0] > time.perf_counter() - t0 - 1.0]
                s = analyze(recent)
                if s:
                    print(f'{(time.perf_counter() - t0) * 1000:10.2f} ms  {line(s)}', flush=True)
        except KeyboardInterrupt:
            pass
    with lock:
        return list(events)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('port', nargs='?', help='substring of the input port name')
    ap.add_argument('--seconds', type=float, help='stop after this long (default: Ctrl+C)')
    ap.add_argument('--threshold', type=float, default=1.0, help='jitter sd in ms for the verdict (default 1)')
    ap.add_argument('--csv', help='save every tick and transport event to this file')
    ap.add_argument('--replay', help='analyze a CSV saved with --csv instead of listening')
    a = ap.parse_args()

    if a.replay:
        events = read_csv(a.replay)
        print(f'Replaying {a.replay}: {len(events)} events')
    else:
        from .common import find_port, input_names
        names = input_names()
        if not a.port:
            print('Input ports:')
            for n in names:
                print('  ', n)
            return 0
        events = listen(find_port(a.port, names, 'input'), a.seconds, a.threshold)
        if a.csv:
            write_csv(a.csv, events)
            print(f'\nwrote {a.csv}')
    return summary(analyze(events), a.threshold)


if __name__ == '__main__':
    sys.exit(main())
