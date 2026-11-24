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
| vis_core.py | MIDI parsing, clock and tempo, show file loader, strobe limiter |
| vis_media.py | Image and video clip sources |
| show.toml | The show: MIDI channels, CC map, songs and their art and clips |
| fake_mpc.py | Virtual MPC for rehearsing visuals without the band |
| run_show.sh | Gig launcher: fullscreen, auto restart, no screen blanking |

## How the MPC talks to it

On the visual channel (default 16):
* **Note on** selects the image or clip mapped to that note in the current song. Velocity sets its intensity.
* **Note off** clears to black only if the item has `hold = true`; otherwise the scene stays until the next note.
* **CC 1 to 9** control the effects (zoom punch, hue, kaleidoscope, feedback, RGB split, snare flash, brightness, drift, hat shimmer). See `[cc]` in show.toml.
* **Program change** switches song.

Drum channel (default 10): kick punches the zoom, snare flashes (only if CC 6 is up, capped at 3 per second), hats add shimmer.

MIDI clock gives the tempo, used for clip tempo stretch and for `pulse_source = "beat"`.

In MPC MIDI settings, enable clock out to the USB port the iMac sees.

## Rehearsing without the MPC

    python3 fake_mpc.py --bpm 124          # terminal 1
    python3 vis_render.py show.toml        # terminal 2, with port = "FakeMPC"

Keys: 1 to 9 trigger items, Space fake kick, S fake snare, Esc quit.

## Gig

    python3 vis_render.py --list-ports     # confirm the MPC port name
    ./run_show.sh                          # projector on display 1; DISPLAY_INDEX=0 ./run_show.sh to change

Before the show, turn off the Xfce compositor (Settings, Window Manager Tweaks, Compositor).

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
* Video clips open when triggered, which can cause a small hitch. Preloading the next clip is planned.
* GIMP layer parallax, tile reveals and video masked through art are not built yet.
