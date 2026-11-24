"""Groovelab26 MIDI driven visuals: the show app.

Listens to the MPC (visual track on one channel, drums on another, MIDI clock)
and renders a fixed 1920x1080 frame: in a window on the iMac for rehearsal,
fullscreen on the projector at the gig.

Usage:
    python3 vis_render.py show.toml               # rehearsal window
    python3 vis_render.py show.toml --fullscreen --display 1
    python3 vis_render.py --list-ports

Keys: Esc quit, 1-9 trigger the current song's items in note order,
Space fake kick, S fake snare.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import moderngl
import numpy as np
import pygame
import rtmidi

from vis_core import (FRACTAL_KINDS, PALETTES, MidiState, StrobeLimiter, beat_pulse,
                      decay, dive_zoom, julia_c, load_show)
from vis_media import ArraySource, ClipSource, ImageSource

VERT = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() { uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FRAG = """
#version 330
uniform sampler2D tex_cur;
uniform sampler2D tex_prev;
uniform sampler2D tex_fb;
uniform vec2 fit_cur;
uniform vec2 fit_prev;
uniform float gain_cur, gain_prev, xfade, zoom, hue, kaleido, feedback;
uniform float rgb_split, brightness, flash, drift, time;
in vec2 uv;
out vec4 frag;

vec2 kal(vec2 p, float n) {
    if (n < 1.5) return p;
    vec2 c = p - 0.5;
    float a = atan(c.y, c.x);
    float r = length(c);
    float seg = 6.2831853 / n;
    a = mod(a, seg);
    a = abs(a - seg * 0.5);
    return vec2(cos(a), sin(a)) * r + 0.5;
}

vec3 hue_rotate(vec3 c, float h) {
    const mat3 to_yiq = mat3(0.299, 0.596, 0.211, 0.587, -0.274, -0.523, 0.114, -0.322, 0.312);
    const mat3 to_rgb = mat3(1.0, 1.0, 1.0, 0.956, -0.272, -1.106, 0.621, -0.647, 1.703);
    vec3 yiq = to_yiq * c;
    float a = h * 6.2831853;
    yiq.yz = mat2(cos(a), sin(a), -sin(a), cos(a)) * yiq.yz;
    return to_rgb * yiq;
}

vec3 layer(sampler2D t, vec2 fit, vec2 p, float split) {
    vec2 q = (p - 0.5) * fit + 0.5;
    vec2 o = vec2(split * fit.x, 0.0);
    vec2 inside = step(vec2(0.0), q) * step(q, vec2(1.0));  // black outside the piece
    return vec3(texture(t, q + o).r, texture(t, q).g, texture(t, q - o).b) * inside.x * inside.y;
}

void main() {
    vec2 p = (uv - 0.5) / zoom;
    p += drift * 0.03 * vec2(sin(time * 0.11), cos(time * 0.07));
    p = kal(p + 0.5, kaleido);
    float s = rgb_split * 0.012;
    vec3 a = layer(tex_cur, fit_cur, p, s) * gain_cur;
    vec3 b = layer(tex_prev, fit_prev, p, s) * gain_prev;
    vec3 col = mix(b, a, xfade);
    vec3 fb = texture(tex_fb, (uv - 0.5) * 0.985 + 0.5).rgb;
    col = mix(col, max(col, fb * 0.96), feedback);
    col = clamp(hue_rotate(col, hue) * brightness, 0.0, 1.0);
    col = mix(col, vec3(1.0), clamp(flash, 0.0, 1.0));
    frag = vec4(col, 1.0);
}
"""

FRAG_FRACTAL = """
#version 330
uniform int mode;          // 0 mandelbrot, 1 julia
uniform vec2 center;
uniform float scale;       // height of the view in the complex plane
uniform float aspect;
uniform vec2 jc;           // julia constant
uniform int max_iter;
uniform float color_phase, color_bands;
uniform vec3 palette;
in vec2 uv;
out vec4 frag;
void main() {
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0) * scale + center;
    vec2 z = mode == 0 ? vec2(0.0) : p;
    vec2 c = mode == 0 ? p : jc;
    float n = 0.0;
    float r2 = 0.0;
    for (int i = 0; i < 2000; i++) {
        if (i >= max_iter) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        r2 = dot(z, z);
        if (r2 > 256.0) break;
        n += 1.0;
    }
    if (n >= float(max_iter)) { frag = vec4(0.0, 0.0, 0.0, 1.0); return; }  // inside the set
    float smooth_n = n + 1.0 - log2(0.5 * log(r2));                          // no banding
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (smooth_n * color_bands + color_phase + palette));
    frag = vec4(col, 1.0);
}
"""

BLIT = """
#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 frag;
void main() { frag = texture(tex, uv); }
"""


def set_u(prog, name, value):
    try:
        prog[name].value = value
    except KeyError:  # uniform optimized out by the driver
        pass


class Layer:
    def __init__(self, ctx, source, item=None, gain=1.0, mode="cover"):
        self.src, self.item, self.gain = source, item, gain
        self.mode = item.fit if item else mode
        f = source.frame
        h, w = f.shape[:2]
        self.size = (w, h)
        self.tex = ctx.texture((w, h), 3, f.tobytes())
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = self.tex.repeat_y = False
        source.update(0.0)  # clear the initial changed flag

    def fit(self, out_aspect):
        """Texture scale for the shader. cover fills the frame and crops; contain shows it all."""
        a = self.size[0] / self.size[1]
        wider = a > out_aspect
        if self.mode == "cover":
            return (out_aspect / a, 1.0) if wider else (1.0, a / out_aspect)
        return (1.0, a / out_aspect) if wider else (out_aspect / a, 1.0)

    def update(self, dt, rate=1.0):
        if self.src.update(dt, rate):
            f = self.src.frame
            if (f.shape[1], f.shape[0]) == self.size:
                self.tex.write(f.tobytes())

    def release(self):
        self.tex.release()
        self.src.close()


class FractalLayer(Layer):
    """A Mandelbrot or Julia set drawn live into its own texture every frame.

    Each trigger starts fresh: the Mandelbrot dive begins at the top, the Julia morph at its start.
    Timing follows the band's tempo; the kick punch and hue knob apply on top as for art.
    """

    def __init__(self, app, item, gain=1.0):
        self.app, self.item, self.gain, self.src = app, item, gain, None
        self.mode = "cover"
        s = item.params["resolution"]
        self.size = (max(64, round(app.W * s)), max(36, round(app.H * s)))
        self.tex = app.ctx.texture(self.size, 4)
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = self.tex.repeat_y = False
        self.fbo = app.ctx.framebuffer(color_attachments=[self.tex])
        self.beats = 0.0

    def update(self, dt, rate=1.0):
        self.beats += dt * self.app.state.bpm / 60.0
        p, prm = self.app.fractal_prog, self.item.params
        if self.item.kind == "mandelbrot":
            set_u(p, "mode", 0)
            set_u(p, "center", tuple(prm["center"]))
            set_u(p, "scale", 3.0 / dive_zoom(self.beats, float(prm["dive_bars"]), float(prm["max_zoom"])))
        else:
            set_u(p, "mode", 1)
            set_u(p, "center", (0.0, 0.0))
            set_u(p, "scale", 3.0 / max(0.05, float(prm["zoom"])))
            set_u(p, "jc", julia_c(self.beats, float(prm["morph_bars"]), float(prm["radius"]), float(prm["start"])))
        set_u(p, "aspect", self.size[0] / self.size[1])
        set_u(p, "max_iter", int(prm["iterations"]))
        set_u(p, "color_phase", self.beats / max(1e-6, float(prm["color_bars"]) * 4.0))
        set_u(p, "color_bands", float(prm["color_bands"]))
        set_u(p, "palette", PALETTES[prm["palette"]])
        self.fbo.use()
        self.app.fractal_vao.render(moderngl.TRIANGLE_STRIP)

    def release(self):
        self.fbo.release()
        self.tex.release()


class MidiConnection:
    """Finds the MPC port by name, and reconnects if the cable is pulled and replugged."""

    def __init__(self, state, port_match):
        self.state, self.match = state, port_match.lower()
        self.midi_in, self.name = None, None
        self._probe = rtmidi.MidiIn()
        self._next_poll = 0.0

    def _cb(self, event, _data=None):
        self.state.feed(event[0])

    def poll(self, now):
        if now < self._next_poll:
            return
        self._next_poll = now + 2.0
        ports = self._probe.get_ports()
        if self.midi_in is not None:
            if self.name not in ports:
                print(f"MIDI port lost: {self.name}", file=sys.stderr)
                self.midi_in.close_port()
                self.midi_in, self.name = None, None
            return
        for i, name in enumerate(ports):
            if self.match in name.lower():
                m = rtmidi.MidiIn()
                m.ignore_types(sysex=True, timing=False, active_sense=True)
                m.set_callback(self._cb)
                m.open_port(i)
                self.midi_in, self.name = m, name
                print(f"MIDI connected: {name}")
                return


class Show:
    def __init__(self, show, fullscreen=None, display=None):
        self.show = show
        o = show.output
        self.W, self.H = int(o["width"]), int(o["height"])
        self.aspect = self.W / self.H
        fullscreen = o["fullscreen"] if fullscreen is None else fullscreen
        display = int(o["display"] if display is None else display)

        m = show.midi
        self.state = MidiState(m["visual_channel"], m["drum_channel"], m["kick_notes"],
                               m["snare_notes"], m["hat_notes"])
        self.conn = MidiConnection(self.state, m["port"])
        self.limiter = StrobeLimiter(float(o["max_flash_hz"]))

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        if sys.platform == "darwin":  # macOS only opens a modern OpenGL context with this flag
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if fullscreen:
            size = pygame.display.get_desktop_sizes()[display]
            flags |= pygame.FULLSCREEN
        else:
            pw = int(o["preview_width"])
            size = (pw, round(pw / self.aspect))
            flags |= pygame.RESIZABLE
        pygame.display.set_mode(size, flags, display=display, vsync=1)
        pygame.mouse.set_visible(not fullscreen)
        pygame.display.set_caption("Groovelab26 visuals")

        self.ctx = moderngl.create_context()
        quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.blit = self.ctx.program(vertex_shader=VERT, fragment_shader=BLIT)
        self.vao = self.ctx.vertex_array(self.prog, [(quad, "2f", "in_pos")])
        self.blit_vao = self.ctx.vertex_array(self.blit, [(quad, "2f", "in_pos")])
        self.fractal_prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG_FRACTAL)
        self.fractal_vao = self.ctx.vertex_array(self.fractal_prog, [(quad, "2f", "in_pos")])
        self.out_tex = [self.ctx.texture((self.W, self.H), 4) for _ in range(2)]
        for t in self.out_tex:
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbos = [self.ctx.framebuffer(color_attachments=[t]) for t in self.out_tex]
        for f in self.fbos:
            f.use()
            f.clear(0, 0, 0, 1)
        self.ping = 0

        self.black = Layer(self.ctx, ArraySource.solid())
        self.fallback = self.black
        if o["fallback"] and o["fallback"].exists():
            self.fallback = Layer(self.ctx, ImageSource(o["fallback"]), mode=o["fit"])
        self.cur = self.prev = None
        self.xfade_t0 = 0.0
        self.image_cache = {}
        self.song = None
        self.select_song(0)
        self.last_snare_t = -1e9
        self.flash_t, self.flash_vel = -1e9, 0.0

    # ---- song and scene control ----
    def select_song(self, program):
        song = self.show.songs.get(program)
        if song is None:
            print(f"no song for program {program}", file=sys.stderr)
            return
        self.song = song
        self.image_cache = {}
        for item in song.items.values():  # preload stills so triggers don't hit the disk
            if item.kind == "image" and item.path and item.path.exists():
                try:
                    self.image_cache[item.note] = ImageSource(item.path).frame
                except Exception as e:
                    print(f"cannot load {item.path}: {e}", file=sys.stderr)
        print(f"song {program}: {song.name} ({len(song.items)} items)")

    def make_source(self, item):
        if item.kind == "image":
            if item.note in self.image_cache:
                return ArraySource.from_frame(self.image_cache[item.note])
            return ImageSource(item.path)
        return ClipSource(item.path, item.start, item.end)

    def trigger(self, item, vel, now):
        gain = 1.0 if item is None else item.gain * (0.25 + 0.75 * vel)
        try:
            if item is not None and item.kind in FRACTAL_KINDS:
                layer = FractalLayer(self, item, gain)
            else:
                src = ArraySource.solid() if item is None else self.make_source(item)
                layer = Layer(self.ctx, src, item, gain)
        except Exception as e:
            print(f"trigger failed for note {item.note}: {e}", file=sys.stderr)
            return
        if self.prev is not None:
            self.prev.release()
        self.prev, self.cur = self.cur, layer
        self.xfade_t0 = now

    def handle(self, ev, now):
        kind = ev[0]
        if kind == "note_on" and self.song:
            item = self.song.items.get(ev[1])
            if item:
                self.trigger(item, ev[2], now)
        elif kind == "note_off" and self.cur and self.cur.item:
            if self.cur.item.note == ev[1] and self.cur.item.hold:
                self.trigger(None, 0, now)
        elif kind == "program":
            self.select_song(ev[1])
        elif kind in ("start", "stop", "continue", "locate"):
            print("transport:", *ev)

    # ---- main loop ----
    def cc(self, name):
        return self.state.cc_value(self.show.cc_map[name], self.show.cc_defaults[name])

    def frame(self, now, dt, t_show):
        o = self.show.output
        offset_s = float(o["offset_ms"]) / 1000.0
        kick, snare, hat = self.state.hits()

        if o["pulse_source"] == "beat":
            pulse = beat_pulse(self.state.beats(now, offset_s)) if self.state.running else 0.0
        else:  # kick driven: offset can only delay, so only negative values apply
            pulse = decay(now + min(0.0, offset_s), kick, 0.12)

        flash_amt = self.cc("flash")
        if snare.t != self.last_snare_t:
            self.last_snare_t = snare.t
            if flash_amt > 0 and self.limiter.allow(now):
                self.flash_t, self.flash_vel = now, snare.vel
        flash = flash_amt * self.flash_vel * math.exp(-(now - self.flash_t) / 0.06)

        idle = self.state.idle_for(now)
        show_fallback = self.cur is None or (idle is not None and idle > float(o["idle_seconds"]))
        if show_fallback:
            a, b, xf, drift = self.fallback, self.black, 1.0, 0.6
        else:
            xf = min(1.0, (now - self.xfade_t0) / max(1e-3, float(o["crossfade"])))
            if xf >= 1.0 and self.prev is not None:
                self.prev.release()
                self.prev = None
            a, b, drift = self.cur, self.prev or self.black, self.cc("drift")
            rate = 1.0
            if a.item is not None and a.item.bpm:
                rate = self.state.bpm / float(a.item.bpm)
            a.update(dt, rate)
            if b is not self.black:
                b.update(dt, 1.0)

        k = self.cc("kaleido")
        p = self.prog
        a.tex.use(0)
        b.tex.use(1)
        self.out_tex[1 - self.ping].use(2)
        set_u(p, "tex_cur", 0)
        set_u(p, "tex_prev", 1)
        set_u(p, "tex_fb", 2)
        set_u(p, "fit_cur", a.fit(self.aspect))
        set_u(p, "fit_prev", b.fit(self.aspect))
        set_u(p, "gain_cur", a.gain)
        set_u(p, "gain_prev", b.gain)
        set_u(p, "xfade", xf)
        base = 1.0 + 0.08 * drift if a.mode == "cover" else 1.0  # cover needs a margin so drift never shows an edge
        set_u(p, "zoom", base * (1.0 + pulse * self.cc("zoom") * 0.15))
        set_u(p, "hue", self.cc("hue"))
        set_u(p, "kaleido", 0.0 if k < 0.05 else 2.0 + round(k * 10))
        set_u(p, "feedback", self.cc("feedback"))
        set_u(p, "rgb_split", self.cc("rgb_split") + self.cc("shimmer") * decay(now, hat, 0.05))
        set_u(p, "brightness", self.cc("brightness"))
        set_u(p, "flash", flash)
        set_u(p, "drift", drift)
        set_u(p, "time", t_show)

        fbo = self.fbos[self.ping]
        fbo.use()
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # letterbox the fixed frame into whatever window or screen we have
        ww, wh = pygame.display.get_window_size()
        self.ctx.screen.use()
        self.ctx.screen.viewport = (0, 0, ww, wh)
        self.ctx.screen.clear(0, 0, 0, 1)
        if ww / wh > self.aspect:
            vw, vh = round(wh * self.aspect), wh
        else:
            vw, vh = ww, round(ww / self.aspect)
        self.ctx.screen.viewport = ((ww - vw) // 2, (wh - vh) // 2, vw, vh)
        self.out_tex[self.ping].use(0)
        set_u(self.blit, "tex", 0)
        self.blit_vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()
        self.ping = 1 - self.ping

    def keys(self, e, now):
        if e.key == pygame.K_SPACE:
            self.state.poke("kick")
        elif e.key == pygame.K_s:
            self.state.poke("snare")
        elif pygame.K_1 <= e.key <= pygame.K_9 and self.song:
            notes = sorted(self.song.items)
            i = e.key - pygame.K_1
            if i < len(notes):
                self.trigger(self.song.items[notes[i]], 1.0, now)

    def run(self):
        clock = pygame.time.Clock()
        t0 = prev = time.monotonic()
        next_caption = 0.0
        fps = int(self.show.output["fps"])
        while True:
            now = time.monotonic()
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if e.type == pygame.KEYDOWN:
                    self.keys(e, now)
            self.conn.poll(now)
            for ev in self.state.drain_events():
                self.handle(ev, now)
            dt, prev = now - prev, now
            self.frame(now, dt, now - t0)
            if now >= next_caption:
                next_caption = now + 0.5
                port = self.conn.name or "no MIDI"
                song = self.song.name if self.song else "-"
                pygame.display.set_caption(
                    f"{song} | {self.state.bpm:.1f} bpm | {'RUN' if self.state.running else 'stop'}"
                    f" | {port} | {clock.get_fps():.0f} fps")
            clock.tick(fps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("show", nargs="?", help="show file (TOML)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--display", type=int, help="screen index for fullscreen (projector is usually 1)")
    ap.add_argument("--list-ports", action="store_true", help="list MIDI inputs and exit")
    args = ap.parse_args()
    if args.list_ports:
        for i, name in enumerate(rtmidi.MidiIn().get_ports()):
            print(i, name)
        return
    if not args.show:
        ap.error("show file required")
    show = load_show(args.show)
    for w in show.warnings:
        print("warning:", w, file=sys.stderr)
    app = Show(show, fullscreen=True if args.fullscreen else None, display=args.display)
    try:
        app.run()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
