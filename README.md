# MIDI Groove Tools

Tools for a band that plays with a machine and still wants to sound human.

Real bands drift. Old records drift. A drum machine doesn't. These command-line tools sit in between: they find the beats in a song that speeds up and slows down, stretch it onto a steady grid without touching the feel inside each beat, turn stems and drum hits into MIDI that keeps its human timing, show what a stem splitter threw away, and measure how steady your MIDI clock really is, down to the fraction of a millisecond.

Built while getting a live rig of an MPC Live III, a CME H12MIDI Pro, Roland AIRA Compact gear and Logic Pro to lock together. Runs on macOS and Linux.

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

Run the same command again to update. No git needed: on a Mac without the developer tools (or any machine without git) the installer downloads the archive instead. `./install.sh --uninstall` removes everything.

Commands go in `~/.local/bin`; everything else lives in `~/.local/share/midi-groove-tools`. Requires Python 3.9+ (the song tools also use numpy; `beatmap --stretch` needs `rubberband`), and on Linux the ALSA library (`alsa-lib` on Arch, `libasound2` on Debian/Ubuntu).

## Commands

| Command | What it does |
| --- | --- |
| `midi-watch [PORT] [--raw-clock]` | Print incoming MIDI with timestamps. Clock is summarized once per beat with BPM and jitter. No argument lists input ports. |
| `midi-watch-all [FILTER]` | Listen on every input port at once; shows which port receives what, with counts on Ctrl+C. |
| `midi-forward [SRC] [DST] [--sync-only]` | Pass MIDI from one port to another, e.g. a drum machine's clock to a synth through the laptop. No arguments lists ports. |
| `beatmap SONG.wav --bpm N` | Find the beats in a song: steady or drifting, and for a steady song the Logic tempo and region position. `--stretch` writes a steady version (needs `rubberband`). |
| `midi-align MIDI WAV --start "BAR BEAT DIV TICK"` | Move audio-to-MIDI notes onto their attacks in the stem; add notes the conversion missed. |
| `midi-clean MIDI [--no-offset --fix-late-lows --fix-outliers 0.25]` | Clean converted MIDI, keep the human feel, report bars to check. |
| `midi-dump MIDI [--bars 41-44] [--grid]` | Show a .mid file bar by bar, or as a 1 e + a grid. |
| `midi-monitor [--grid] [--hex]` | Live colored MIDI monitor; `--grid` prints each bar on a 1 e + a grid. |
| `midi-clock-check [PORT] [--seconds 60] [--csv FILE]` | Measure how steady a MIDI clock is: tempo, tick jitter (sd, 95th percentile, worst), missing and doubled ticks, and a verdict (`--threshold`, default 1 ms). `--replay FILE` re-analyzes a saved run. |
| `song-minus-stems MIX.wav STEMS.wav... [--bpm N --first S]` | Subtract the stems from the original mix to hear what the splitter dropped; lists the bars where the most went missing. `--solo STEM` writes the mix minus every other stem. |
| `hits-to-midi DRUMS.wav --bpm N [--first S --bars 15-22 --band snare]` | Turn drum hits in one band (kick, snare, hat or LOW-HIGH Hz) into MIDI notes with their human timing and velocity, unquantized. |
| `stem-check FOLDER` | Check every WAV in a folder (or the files given) has the same length to the sample, sample rate and bit depth, so the stems start and end together on the MPC. Exit code 1 on any mismatch. |
| `stem-pad FOLDER --bars N --bpm N` | Pad every WAV in a folder with silence at the end so all are exactly N bars from bar 1. For Logic exports that stop each track where its sound stops. |
| `mpc-pattern MIDI [OUT]` | Make an MPC pattern (.mpcpattern) from a .mid so it loads onto the track you choose (the MPC loads a .mid only as a whole new sequence). Timing exact from the start of the file, never rounded; every note checked after. Keep the .mid as the master. |
| `play-along STEMS --bpm N --from 25 --to 32 [--minus Bass --count-in 1]` | Play a song's stems from any bar and loop a few bars until Ctrl+C, leaving out the part you want to play yourself. `--first SECONDS` (the band's first downbeat) follows a live band's own beats instead of a steady grid. `--once` plays through once; `--out FILE` writes the mix instead. |
| `midi-tools-test` | Check the song tools still do the right thing. |

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
