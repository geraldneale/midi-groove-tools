# MIDI Groove Tools

Small command-line utilities for watching and routing MIDI on macOS and Linux. Built while syncing an MPC Live III, a CME H12MIDI Pro, Roland AIRA Compact gear and Logic Pro.

## Install

```
curl -fsSL https://raw.githubusercontent.com/geraldneale/midi-groove-tools/main/install.sh | sh
```

Or from a clone:

```
git clone https://github.com/geraldneale/midi-groove-tools.git
cd midi-groove-tools
./install.sh
```

Run the same command again to update. `./install.sh --uninstall` removes everything.

Commands go in `~/.local/bin`; everything else lives in `~/.local/share/midi-groove-tools`. Requires Python 3.9+, and on Linux the ALSA library (`alsa-lib` on Arch, `libasound2` on Debian/Ubuntu).

## Commands

| Command | What it does |
| --- | --- |
| `midi-watch [PORT] [--raw-clock]` | Print incoming MIDI with timestamps. Clock is summarized once per beat with BPM and jitter. No argument lists input ports. |
| `midi-watch-all [FILTER]` | Listen on every input port at once; shows which port receives what, with counts on Ctrl+C. |
| `midi-forward [SRC] [DST] [--sync-only]` | Pass MIDI from one port to another, e.g. a drum machine's clock to a synth through the laptop. No arguments lists ports. |

Port names match by substring, case-insensitive: `midi-watch "Port 1"` listens on the first input whose name contains "Port 1".

### Example: checking a clock source

```
$ midi-watch "MPC Live III Port 1"
Listening on: MPC Live III Port 1   (Ctrl+C to quit)

   4640.59 ms  FA        start time=0
   4640.79 ms  F2 00 00  songpos pos=0 time=0
   4645.26 ms  --- bar   1 beat 1  (measuring tempo...)
   5245.37 ms  --- bar   1 beat 2   99.98 BPM   jitter 0.583 ms
```

Jitter is measured on the receiving computer, so it includes USB and OS timing: the source is at least that steady.

## Design

The command names are the stable interface. Tools start in Python (`python/`) and may move to Rust (`rust/`). `install.sh` installs a Rust binary under the same name when one is built, so scripts and habits keep working.

## License

MIT. See [LICENSE](LICENSE).
