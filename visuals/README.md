# Groovelab26 visuals

The MPC drives the visuals over MIDI. A Python app on the iMac (Xubuntu) renders a fixed 1920x1080 frame: in a window for rehearsal, fullscreen on the projector at the gig.

## Setup (Xubuntu)

Python 3.9 or newer. Not installed by the main `install.sh`.

    sudo apt install python3-venv mesa-utils
    cd ~/midi-groove-tools/visuals
    python3 -m venv ~/vis-env && source ~/vis-env/bin/activate
    pip install -r requirements.txt
    python3 -m unittest test_vis          # logic tests, no hardware needed

## Files

| File | What it does |
|---|---|
| vis_render.py | The show app: MIDI in, shaders, output |
| vis_core.py | MIDI parsing, clock and tempo, show file loader |
| vis_media.py | Image and video clip sources |
| vis_shuffle.py | Fills gaps with clips and art, least shown first (after SoundTrack) |
| show.toml | The show: MIDI channels, CC map, songs and their art and clips |
| fake_mpc.py | Virtual MPC for rehearsing visuals without the band |
| run_show.sh | Gig launcher: fullscreen, auto restart, no screen blanking |

## How the MPC talks to it

On the visual channel (default 16):
* **Note on** selects the image or clip mapped to that note in the current song. Velocity sets its intensity. About 1 scene in 4 comes up mirrored, upside down or turned sideways (`twist_chance` in `[output]`, 0 = never); add `twist = false` to an item to keep it exactly as it is, or `twist = "mirror"` to allow only a left to right mirror, never turned sideways or upside down (good for faces).
* **Live scenes from the MIDI itself:** `type = "monitor"`, `"hex"` or `"binary"` puts one of midi-monitor's looks on the projector, drawn from the incoming MIDI on an 80 by 25 character screen like a CGA monitor: the colored message monitor (one CGA color per channel, velocity as a block bar), the raw bytes in hex, or the same bytes as zeroes and ones. They keep their true CGA colors and stay upright; add `mirror = true` or `twist = true` to an item to change that. `mirror = false` on any item keeps its true colors: no color mirroring and no hue knob.
* **No art? Live scenes instead.** A scene note with no item, or whose image or clip file is missing, gets the next live scene in turn: Mandelbrot, Julia, monitor, hex, binary. With no fallback image, the Julia set plays instead of a black screen. `generate = false` in `[output]` turns this off.
* **Shuffle, after SoundTrack:** note 60 on the visual track fills a gap, with its velocity giving the gap's length in bars. It chains video clips from `clips/` and pictures from `art/`, least shown first; anything newly dropped into either folder comes up more often at first so you can check it. Pictures stay up 16 bars and clips up to 16 (`image_bars`, `max_clip_bars`), long enough for the colors to develop; each piece plays whole bars, so changes land on bar lines. Note 61 (or N in the visuals window) skips the piece on screen with a gentle fade and doesn't count it as shown; X drops it for good. Play counts and the never-again list live in `shuffle.db` beside the show file. A shuffled piece keeps any settings its own show.toml entry has (faces, true colors). Settings in `[shuffle]`.
* **Fractals:** an item can be `type = "mandelbrot"` (a tempo locked dive that floats back out and repeats) or `type = "julia"` (a shape that morphs on a loop). They're drawn live, need no file, and start fresh on each trigger. Settings are in the example show.toml. Fine detail is smoothed with several samples per pixel (`smoothing`, 1 to 4; the Mandelbrot uses 3 and colors by the log of its detail so it doesn't shimmer as it dives). If a laptop stutters, lower `smoothing` or `resolution`. The Julia set's shape and colors are slow by default (`morph_bars = 64`, `color_bars = 8`).
* **Note off** clears to black only if the item has `hold = true`; otherwise the scene stays until the next note.
* **CCs** control the effects: 1 zoom punch, 2 hue, 3 kaleidoscope, 4 feedback, 5 RGB split, 7 brightness, 8 drift, 9 hat shimmer, 10 color mirroring, 11 true angle. There is no screen flash: it was taken out for good, because full screen flashes can trigger seizures. See `[cc]` in show.toml.
* **Black and white art gets color:** a picture or clip with almost no color of its own is colored by lightness (each pixel keeps its exact lightness and takes a hue from how light it is), so the mirroring has color to turn. Automatic; `colorize = true` or `false` on an item forces it, `colorize = false` under `[output]` turns it off, and true colors pieces (`mirror = false`) are left alone. `colorize_hue`, `colorize_span` and `colorize_saturation` in `[output]` set the colors.
* **Color mirroring** (CC 10 amount, CC 11 true angle) is Bram Cohen's okhsl mirror from his color_cycling code, run on every pixel of everything on screen. Two opposite hues stay true, the ones 90 degrees from them swap, and the true pair travels round the color wheel once every `mirror_bars`, locked to tempo. Lightness never changes, so pictures keep their shape. With `mirror_bars = 0` it holds still and CC 11 aims the true hues on cue. The math is in vis_color.py, with a plain Python version the tests check the GPU against.
* **Program change** switches song.

Drum channel (default 10): kick gives a subtle zoom pulse; about 1 kick in 8 pulses twice as much (`medium_pulse_chance`, `medium_pulse_size`) and about 1 in 32 six times as much (`big_pulse_chance`, `big_pulse_size`); about 1 in 20 gives no punch at all, like a dropped beat: `skip_pulse_chance`, with the Mandelbrot and Julia at 1 in 10, and any item can set its own `skip_pulse_chance`; all in `[output]`, 0 turns one off), hats add shimmer.

MIDI clock gives the tempo, used for clip tempo stretch and for `pulse_source = "beat"`.

In MPC MIDI settings, enable clock out to the USB port the iMac sees.

## Rehearsing without the MPC

    python3 fake_mpc.py --bpm 124          # terminal 1
    python3 vis_render.py show.toml        # terminal 2, with port = "FakeMPC"

Keys: 1 to 9 trigger items, Space fake kick, Esc quit.

Scene length: every scene gets `--bars` bars; give a note its own length with a colon, e.g. `--notes 39 40:32 41:32 44` keeps the Mandelbrot (40) and Julia (41) up for 32 bars. At the gig, a scene lasts as long as its note on the MPC's visual track. The fake MPC stops by itself after 500 minutes (`--minutes`, 0 = until Ctrl+C), so it can't be left running for days.

## Gig

    python3 vis_render.py --list-ports     # confirm the MPC port name
    ./run_show.sh                          # projector on display 1; DISPLAY_INDEX=0 ./run_show.sh to change

**On a Mac** (for example a 2016 MacBook Pro, Intel): install Python 3.9 or newer from python.org, then the same `pip install -r requirements.txt`. The 2016 has only USB-C/Thunderbolt 3 ports, so bring a USB-C to HDMI adapter for the projector and a USB-C cable or adapter for the MPC. Projector is usually display 1. If pip finds no prebuilt OpenCV for your macOS version, try `pip install "opencv-python<4.11"`.

On Xubuntu, before the show, turn off the Xfce compositor (Settings, Window Manager Tweaks, Compositor).

**Projector lag.** With `pulse_source = "beat"`, set `offset_ms` to a positive value (try 30 to 60) until the zoom punch lands on the kick you hear. With `pulse_source = "kick"` the app can only add delay. To get note driven scenes earlier, nudge the visual track earlier in the MPC.

## Preparing material

GIMP art: export PNG. Anything over 4096 px is scaled down on load. For projection, a punchier contrast and saturation pass usually helps.

Video clips: cut and transcode every frame as a keyframe, so loops and jumps are exact:

    yt-dlp -f "bv*[height<=1080]" -o "raw/%(id)s.%(ext)s" <url>
    ffmpeg -ss 00:01:23 -to 00:01:41 -i raw/<id>.webm -an \
      -vf "scale=1920:-2,fps=30" -c:v libx264 -g 1 -crf 18 -pix_fmt yuv420p clips/name.mp4

Only use footage you have the right to show publicly.

## Known gaps (next steps)

* After a Song Position jump the app doesn't know which scene should be showing until the next scene note. The fix is to read each song's visual MIDI file and look up the scene at that position.
* GIMP layer parallax, tile reveals and video masked through art are not built yet.
