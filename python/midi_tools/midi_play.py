#!/usr/bin/env python3
"""midi-play: play a .mid file to a MIDI port, from any bar, on any channel.

For trying a part on a hardware synth straight from the computer, without Logic
or the MPC.

  midi-play                                              list the output ports
  midi-play HungUp_SynthBass.mid --port S-1
  midi-play HungUp_SynthBass.mid --port S-1 --channel 3 --from 25

--port matches any part of the port name, in any case. --channel sends every note
on that channel (1-16) instead of the file's own. --from starts at that bar
(counted from bar 1 at the start of the file, using the file's tempo and time
signature), so a part that comes in at bar 25 starts at once. Ctrl+C stops it and
turns off any note still sounding.
"""
import argparse, os, struct, sys, time


def read_vlq(b, i):
    v = 0
    while True:
        c = b[i]
        i += 1
        v = (v << 7) | (c & 0x7F)
        if not c & 0x80:
            return v, i


def read_events(path):
    """(ppqn, [(tick, track, kind, data)]) for every track; kind is 'tempo', 'meter' or 'midi'."""
    b = open(path, 'rb').read()
    if b[:4] != b'MThd':
        sys.exit(f'{path}: not a MIDI file')
    hlen = struct.unpack('>I', b[4:8])[0]
    _fmt, ntrk, div = struct.unpack('>HHH', b[8:14])
    if div & 0x8000:
        sys.exit(f'{path}: timecode-based timing is not supported')
    i, events = 8 + hlen, []
    for trk in range(ntrk):
        if b[i:i + 4] != b'MTrk':
            sys.exit(f'{path}: damaged track')
        end = i + 8 + struct.unpack('>I', b[i + 4:i + 8])[0]
        j, t, status = i + 8, 0, None
        while j < end:
            d, j = read_vlq(b, j)
            t += d
            c = b[j]
            if c == 0xFF:
                typ = b[j + 1]
                ln, k = read_vlq(b, j + 2)
                body = b[k:k + ln]
                j = k + ln
                if typ == 0x51:
                    events.append((t, trk, 'tempo', int.from_bytes(body, 'big')))
                elif typ == 0x58:
                    events.append((t, trk, 'meter', (body[0], 2 ** body[1])))
                continue
            if c in (0xF0, 0xF7):
                ln, j = read_vlq(b, j + 1)
                j += ln
                continue
            if c & 0x80:
                status, j = c, j + 1
            n = 1 if (status & 0xF0) in (0xC0, 0xD0) else 2
            events.append((t, trk, 'midi', bytes([status]) + b[j:j + n]))
            j += n
        i = end
    events.sort(key=lambda e: (e[0], e[1]))
    return div, events


def schedule(path, from_bar=1, channel=None):
    """[(seconds after the start bar, raw MIDI bytes)], the BPM at the start bar, and the meter."""
    ppqn, events = read_events(path)
    meter = next((d for t, _, k, d in events if k == 'meter' and t == 0), (4, 4))
    bar_ticks = ppqn * 4 * meter[0] // meter[1]
    start = (from_bar - 1) * bar_ticks
    tempo, last_t, sec, start_sec = 500000, 0, 0.0, None
    out = []
    for t, _, kind, data in events:
        if start_sec is None and t >= start:
            start_sec = sec + (start - last_t) * tempo / ppqn / 1e6
        sec += (t - last_t) * tempo / ppqn / 1e6
        last_t = t
        if kind == 'tempo':
            tempo = data
        elif kind == 'midi' and t >= start:
            if channel is not None and data[0] < 0xF0:
                data = bytes([(data[0] & 0xF0) | (channel - 1)]) + data[1:]
            out.append((sec - start_sec, data))
    if start_sec is None:
        sys.exit(f'the file ends before bar {from_bar}')
    start_tempo = ([d for t, _, k, d in events if k == 'tempo' and t <= start] or [500000])[-1]
    return out, 60e6 / start_tempo, meter


def main():
    p = argparse.ArgumentParser(prog='midi-play', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mid', nargs='?')
    p.add_argument('--port', help='part of the output port name, e.g. S-1')
    p.add_argument('--channel', type=int, choices=range(1, 17), metavar='1-16',
                   help="send on this channel instead of the file's own")
    p.add_argument('--from', dest='first', type=int, default=1, help='bar to start at (default 1)')
    a = p.parse_args()

    import mido
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import find_port
    names = mido.get_output_names()
    if not a.mid or not a.port:
        print('Output ports:')
        for n in names:
            print('  ', n)
        if a.mid:
            print('give one with --port')
        return
    if not os.path.exists(a.mid):
        sys.exit(f'{a.mid}: no such file on this machine')
    if a.first < 1:
        sys.exit('--from must be 1 or later')
    events, bpm, meter = schedule(a.mid, a.first, a.channel)
    name = find_port(a.port, names, 'output')
    if not events:
        sys.exit(f'no notes from bar {a.first} on')
    ch = f'channel {a.channel}' if a.channel else "the file's channels"
    print(f'playing {os.path.basename(a.mid)} on {name}, {ch}, from bar {a.first} '
          f'({bpm:g} BPM, {meter[0]}/{meter[1]}); Ctrl+C to stop')
    with mido.open_output(name) as out:
        try:
            t0 = time.time()
            for t, data in events:
                wait = t0 + t - time.time()
                if wait > 0:
                    time.sleep(wait)
                out.send(mido.Message.from_bytes(list(data)))
            print('done')
        except KeyboardInterrupt:
            print()
        finally:
            out.reset()


if __name__ == '__main__':
    main()
