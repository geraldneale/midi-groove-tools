# Working on MIDI Groove Tools

Gerald's open source command-line tools for a live band rig: MIDI clock and monitoring,
and getting songs from stems onto a steady grid for the MPC Live III. MIT licensed and
public, so nothing private (names of venues, bandmates' details, work matters) goes in.
Several chats change this repo, often on the same day. These rules keep them from undoing
each other's work.

## Before changing anything

- Start from the current main: `git fetch origin main && git reset --hard origin/main`
  (or a fresh clone). Never apply files written earlier in a chat or a Project doc over
  main without checking what changed on main since; another chat may have added to them.
- Run `python3 python/midi_tools/test_midi_tools.py` first, so you know it passed before
  your change.

## Making a change

- Tools live in `python/midi_tools/`, one module per command. The command name in
  `install.sh` (`TOOLS=`, as `command-name:module`) is the stable interface; keep it.
- A new command needs all three: the module, an entry in `TOOLS=` in `install.sh`
  (before `midi-tools-test`), and a row in the README command table.
- `test_midi_tools.py` is shared. Add a test function for your tool and a call in
  `main()`. Never replace the file wholesale and never delete another tool's test.
  Tests use fixed random seeds and synthetic data, so results are the same everywhere.
- Must run on Python 3.9 (an old Mac runs 3.9), macOS and Linux. Dependencies are in
  `python/requirements.txt`; keep to numpy, mido, python-rtmidi and colorama unless
  there's a strong reason. Tools that don't need MIDI ports must not import mido at the
  top level, so they work on machines without MIDI.
- Write help text and output in plain words for a musician, not a programmer.

## Before pushing

- `python3 python/midi_tools/test_midi_tools.py` must end with `all good`. If rubberband
  isn't installed, the stretch test is skipped; say so, since it runs on Gerald's Mac.
- `sh -n install.sh` must pass.
- Commit to main as Gerald:
  `git -c user.name="Gerald Neale" -c user.email="53587595+geraldneale@users.noreply.github.com" commit`
  with a message saying what changed and why, ending with the session's attribution lines.
- If the push is rejected because main moved, fetch, rebase onto `origin/main`, rerun the
  tests, and push again. Never force push.

## After pushing

Tell Gerald what changed and that machines update with:
`cd ~ && curl -fsSL https://raw.githubusercontent.com/geraldneale/midi-groove-tools/main/install.sh | sh`
(on the Linux laptop, whose install runs from its clone: `cd ~/midi-groove-tools && git pull && sh ./install.sh`).
