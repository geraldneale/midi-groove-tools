#!/usr/bin/env python3
"""midi_monitor.py: watch MIDI live in the terminal.

  python3 midi_monitor.py                  # every message, one line each
  python3 midi_monitor.py --port H12       # only ports whose name contains "H12"
  python3 midi_monitor.py --no-notes       # control and transport only
  python3 midi_monitor.py --grid           # a 1 e + a grid for each finished bar
  python3 midi_monitor.py --grid --bpm 125 # grid without MIDI clock (bar 1 = first note)
  python3 midi_monitor.py --hex            # the raw bytes, 16 per row

Note names use Logic's convention (middle C = C3). Ctrl+C to stop.
Colors: the 16-color IBM PC (CGA) palette, nothing else.
Needs mido, python-rtmidi and colorama (use your venv).
"""
import argparse, sys, threading, time
from collections import defaultdict
import mido
from colorama import Back, Fore, Style, init

init(strip=not sys.stdout.isatty())   # plain text when piped to a file
R = Style.RESET_ALL
DIM = Fore.LIGHTBLACK_EX               # CGA dark gray
# one color per MIDI channel from the CGA palette. Black is invisible, leaving 15
# colors for 16 channels, so the two middle channels (least used) share light gray.
CH = {1: Fore.LIGHTCYAN_EX, 2: Fore.CYAN, 3: Fore.LIGHTBLUE_EX, 4: Fore.BLUE,
      5: Fore.LIGHTGREEN_EX, 6: Fore.GREEN, 7: Fore.LIGHTBLACK_EX, 8: Fore.WHITE,
      9: Fore.WHITE, 10: Fore.LIGHTRED_EX, 11: Fore.LIGHTWHITE_EX, 12: Fore.RED,
      13: Fore.MAGENTA, 14: Fore.LIGHTMAGENTA_EX, 15: Fore.YELLOW, 16: Fore.LIGHTYELLOW_EX}
BAR = Back.BLUE + Fore.LIGHTWHITE_EX
LOOK = {'start': Back.GREEN + Fore.BLACK, 'continue': Back.GREEN + Fore.BLACK,
        'stop': Back.RED + Fore.LIGHTWHITE_EX}
BLOCK = '█'                       # full block, from the original PC character set

NAMES = 'C C# D D# E F F# G G# A A# B'.split()
lock = threading.Lock()
t0 = time.perf_counter()
clock = {'ticks': 0, 'running': False, 'last': None, 'gaps': []}
grid = {'first': None, 'hits': defaultdict(dict), 'printed': 0}
hexrow = {'cells': [], 'offset': 0}


def note(n):
    return f'{NAMES[n % 12]}{n // 12 - 2}'


def bpm_now():
    g = clock['gaps']
    return 60 / (24 * sum(g) / len(g)) if g else 0


def out(port, text):
    with lock:
        print(f'{DIM}{time.perf_counter() - t0:9.3f}{R}  {Fore.CYAN}{port[:14]:<14}{R}  {text}{R}', flush=True)


# ---------- --hex ----------

def hex_out(msg):
    b = msg.bytes()
    status = b[0]
    for i, x in enumerate(b):
        if i == 0 and status < 0xF0:
            col = CH[(status & 0x0F) + 1]           # status byte in its channel's color
        elif i == 0 and status in (0xFA, 0xFB):
            col = LOOK['start']
        elif i == 0 and status == 0xFC:
            col = LOOK['stop']
        elif status == 0xF8:
            col = DIM
        elif status == 0xF0:
            col = Fore.YELLOW
        else:
            col = Fore.WHITE
        hexrow['cells'].append(f'{col}{x:02X}{R}')
        if len(hexrow['cells']) == 16:
            with lock:
                print(f'{DIM}{hexrow["offset"]:06X}{R}  ' + ' '.join(hexrow['cells']), flush=True)
            hexrow['offset'] += 16
            hexrow['cells'] = []


# ---------- --grid ----------

def position(a):
    """Where we are, in 16th notes since bar 1. From MIDI clock if running, else from --bpm."""
    if clock['running']:
        return clock['ticks'] / 6
    if a.bpm and grid['first'] is not None:
        return (time.perf_counter() - grid['first']) / (60 / a.bpm / 4)
    return None


def grid_note(msg, a):
    if a.bpm and not clock['running'] and grid['first'] is None:
        grid['first'] = time.perf_counter()
    p = position(a)
    if p is None:
        return
    step = round(p)
    row = grid['hits'][step // 16].setdefault((msg.channel + 1, msg.note), {})
    row[step % 16] = max(row.get(step % 16, 0), msg.velocity)


def grid_render(bar):
    rows = grid['hits'].pop(bar, {})
    if not rows:                              # skip silent bars
        return
    lines = [f'{BAR} bar {bar + 1:<4} {bpm_now():6.1f} BPM {R}' if clock['gaps'] else f'{BAR} bar {bar + 1:<4} {R}',
             f'{DIM}            1 e + a 2 e + a 3 e + a 4 e + a{R}']
    for (ch, n) in sorted(rows, key=lambda k: (k[0], -k[1])):
        col = CH[ch]
        cells = ' '.join(f'{col}{BLOCK}{R}' if s in rows[(ch, n)] else f'{DIM}·{R}' for s in range(16))
        lines.append(f'{col}ch{ch:<2} {note(n):<5}{R} {cells}')
    with lock:
        print('\n'.join(lines) + '\n', flush=True)


def grid_ticker(a):
    while True:
        time.sleep(0.01)
        p = position(a)
        if p is None:
            continue
        now_bar = int((p + 0.5) // 16)       # a note a hair early still counts in the next bar
        while grid['printed'] < now_bar:
            grid_render(grid['printed'])
            grid['printed'] += 1


# ---------- all modes ----------

def handle(port, msg, a):
    if a.hex:
        hex_out(msg)
    c = clock
    if msg.type == 'clock':
        now = time.perf_counter()
        if c['last'] is not None:
            c['gaps'] = (c['gaps'] + [now - c['last']])[-24:]
        c['last'] = now
        if c['running']:
            if c['ticks'] % 96 == 0 and not (a.grid or a.hex):
                out(port, f'{BAR} bar {c["ticks"] // 96 + 1:<4} {bpm_now():6.1f} BPM {R}')
            c['ticks'] += 1
        return
    if msg.type in ('start', 'continue', 'stop'):
        if msg.type == 'start':
            c['ticks'] = 0
            grid['hits'].clear(); grid['printed'] = 0
        c['running'] = msg.type != 'stop'
        if not a.hex:
            out(port, f'{LOOK[msg.type]} {msg.type.upper()} ')
        return
    if a.hex or msg.type == 'active_sensing':
        return
    if a.grid:
        if msg.type == 'note_on' and msg.velocity > 0:
            grid_note(msg, a)
        return
    col = CH[msg.channel + 1] if hasattr(msg, 'channel') else Fore.WHITE
    ch = f'{col}ch{msg.channel + 1:<2}{R}' if hasattr(msg, 'channel') else '    '
    if msg.type in ('note_on', 'note_off'):
        if a.no_notes:
            return
        on = msg.type == 'note_on' and msg.velocity > 0
        bar = BLOCK * (msg.velocity // 4 + 1)       # velocity as a block bar
        text = f'{col}on  {note(msg.note):<4} {bar}' if on else f'{DIM}off {note(msg.note)}'
    elif msg.type == 'control_change':
        text = f'{Fore.MAGENTA}CC {msg.control:3d} = {msg.value}'
    elif msg.type == 'program_change':
        text = f'{Back.MAGENTA}{Fore.LIGHTWHITE_EX} program {msg.program} '
    elif msg.type == 'pitchwheel':
        text = f'{Fore.MAGENTA}pitch bend {msg.pitch}'
    elif msg.type == 'sysex':
        text = f'{Fore.YELLOW}sysex {len(msg.data)} bytes'
    else:
        text = str(msg)
    out(port, f'{ch} {text}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', help='only watch ports whose name contains this text')
    ap.add_argument('--no-notes', action='store_true', help='hide note on/off')
    ap.add_argument('--grid', action='store_true', help='print a 1 e + a grid for each finished bar')
    ap.add_argument('--bpm', type=float, help='tempo for --grid when no MIDI clock is coming in')
    ap.add_argument('--hex', action='store_true', help='show the raw bytes')
    a = ap.parse_args()

    names = [n for n in mido.get_input_names() if not a.port or a.port.lower() in n.lower()]
    if not names:
        sys.exit('no matching MIDI input ports. Available: ' + ', '.join(mido.get_input_names()))
    ports = []
    for n in names:
        p = mido.open_input(n, callback=lambda m, n=n: handle(n, m, a))
        # rtmidi drops clock and sysex by default; we want them
        if hasattr(p, '_rt'):
            p._rt.ignore_types(sysex=False, timing=False, active_sense=True)
        ports.append(p)
        print(f'{Fore.LIGHTGREEN_EX}watching:{R} {n}')
    if a.grid:
        threading.Thread(target=grid_ticker, args=(a,), daemon=True).start()
        print(f'{DIM}grid: waiting for {"MIDI clock or the first note" if a.bpm else "MIDI clock (START)"}{R}')
    elif not a.hex:
        print(f'{DIM}     time  port            message{R}')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for p in ports:
            p.close()


if __name__ == '__main__':
    main()
