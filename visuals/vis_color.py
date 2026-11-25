"""Color mirroring after Bram Cohen's color_cycling (public domain), for live visuals.

Each pixel goes to okhsl (Bjorn Ottosson's perceptual hue/saturation/lightness,
which keeps lightness steady as hue changes), its hue h becomes angle - h, and it
comes back to sRGB with the same saturation and lightness. At any moment the two
opposite hues at angle/2 and angle/2 + 180 degrees stay true, the ones 90 degrees
from them are swapped, and as angle moves the true pair travels round the wheel.
okhsl keeps every result inside sRGB, so nothing clips.

The Python here is the readable reference (and what the tests check). The same
math runs on the GPU from OKHSL_GLSL; tests render it and compare.
Hue is in turns (0 to 1) throughout, not degrees.
"""
import math

# ---- sRGB transfer ----
def srgb_to_linear(x):
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def linear_to_srgb(x):
    return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055


# ---- Oklab (Ottosson) ----
def linear_srgb_to_oklab(r, g, b):
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def oklab_to_linear_srgb(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


# ---- okhsl gamut helpers (Ottosson, "Okhsv and Okhsl", 2021) ----
def _max_saturation(a, b):
    if -1.88170328 * a - 0.80936493 * b > 1:
        k = (1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245)
        w = (4.0767416621, -3.3077115913, 0.2309699292)
    elif 1.81444104 * a - 1.19445276 * b > 1:
        k = (0.73956515, -0.45954404, 0.08285427, 0.12541070, 0.14503204)
        w = (-1.2684380046, 2.6097574011, -0.3413193965)
    else:
        k = (1.35733652, -0.00915799, -1.15130210, -0.50559606, 0.00692167)
        w = (-0.0041960863, -0.7034186147, 1.7076147010)
    S = k[0] + k[1] * a + k[2] * b + k[3] * a * a + k[4] * a * b
    kl = 0.3963377774 * a + 0.2158037573 * b
    km = -0.1055613458 * a - 0.0638541728 * b
    ks = -0.0894841775 * a - 1.2914855480 * b
    l_, m_, s_ = 1 + S * kl, 1 + S * km, 1 + S * ks
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    l1, m1, s1 = 3 * kl * l_ * l_, 3 * km * m_ * m_, 3 * ks * s_ * s_
    l2, m2, s2 = 6 * kl * kl * l_, 6 * km * km * m_, 6 * ks * ks * s_
    f = w[0] * l + w[1] * m + w[2] * s
    f1 = w[0] * l1 + w[1] * m1 + w[2] * s1
    f2 = w[0] * l2 + w[1] * m2 + w[2] * s2
    return S - f * f1 / (f1 * f1 - 0.5 * f * f2)


def _find_cusp(a, b):
    S = _max_saturation(a, b)
    r, g, bl = oklab_to_linear_srgb(1, S * a, S * b)
    L = (1 / max(r, g, bl)) ** (1 / 3)
    return L, L * S


def _gamut_intersection(a, b, L1, C1, L0, cusp):
    cL, cC = cusp
    if (L1 - L0) * cC - (cL - L0) * C1 <= 0:
        return cC * L0 / (C1 * cL + cC * (L0 - L1))
    t = cC * (L0 - 1) / (C1 * (cL - 1) + cC * (L0 - L1))
    dL, dC = L1 - L0, C1
    kl = 0.3963377774 * a + 0.2158037573 * b
    km = -0.1055613458 * a - 0.0638541728 * b
    ks = -0.0894841775 * a - 1.2914855480 * b
    ldt, mdt, sdt = dL + dC * kl, dL + dC * km, dL + dC * ks
    L, C = L0 * (1 - t) + t * L1, t * C1
    l_, m_, s_ = L + C * kl, L + C * km, L + C * ks
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    l1, m1, s1 = 3 * ldt * l_ * l_, 3 * mdt * m_ * m_, 3 * sdt * s_ * s_
    l2, m2, s2 = 6 * ldt * ldt * l_, 6 * mdt * mdt * m_, 6 * sdt * sdt * s_
    best = float("inf")
    for w in ((4.0767416621, -3.3077115913, 0.2309699292),
              (-1.2684380046, 2.6097574011, -0.3413193965),
              (-0.0041960863, -0.7034186147, 1.7076147010)):
        f = w[0] * l + w[1] * m + w[2] * s - 1
        f1 = w[0] * l1 + w[1] * m1 + w[2] * s1
        f2 = w[0] * l2 + w[1] * m2 + w[2] * s2
        u = f1 / (f1 * f1 - 0.5 * f * f2)
        if u >= 0:
            best = min(best, -f * u)
    return t + best


def _toe(x):
    k1, k2 = 0.206, 0.03
    k3 = (1 + k1) / (1 + k2)
    return 0.5 * (k3 * x - k1 + math.sqrt((k3 * x - k1) ** 2 + 4 * k2 * k3 * x))


def _toe_inv(x):
    k1, k2 = 0.206, 0.03
    k3 = (1 + k1) / (1 + k2)
    return (x * x + k1 * x) / (k3 * (x + k2))


def _st_mid(a, b):
    S = 0.11516993 + 1 / (7.44778970 + 4.15901240 * b + a * (-2.19557347 + 1.75198401 * b + a * (
        -2.13704948 - 10.02301043 * b + a * (-4.24894561 + 5.38770819 * b + 4.69891013 * a))))
    T = 0.11239642 + 1 / (1.61320320 - 0.68124379 * b + a * (0.40370612 + 0.90148123 * b + a * (
        -0.27087943 + 0.61223990 * b + a * (0.00299215 - 0.45399568 * b - 0.14661872 * a))))
    return S, T


def _cs(L, a, b):
    cusp = _find_cusp(a, b)
    C_max = _gamut_intersection(a, b, L, 1, L, cusp)
    S_max, T_max = cusp[1] / cusp[0], cusp[1] / (1 - cusp[0])
    k = C_max / min(L * S_max, (1 - L) * T_max)
    S_mid, T_mid = _st_mid(a, b)
    Ca, Cb = L * S_mid, (1 - L) * T_mid
    C_mid = 0.9 * k * math.sqrt(math.sqrt(1 / (1 / Ca ** 4 + 1 / Cb ** 4)))
    Ca, Cb = L * 0.4, (1 - L) * 0.8
    C_0 = math.sqrt(1 / (1 / Ca ** 2 + 1 / Cb ** 2))
    return C_0, C_mid, C_max


# ---- okhsl ----
def srgb_to_okhsl(r, g, b):
    """sRGB 0..1 -> (hue in turns, saturation 0..1, lightness 0..1)."""
    L, a, bb = linear_srgb_to_oklab(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))
    l = _toe(L)
    C = math.hypot(a, bb)
    if C < 1e-7 or L <= 1e-7 or L >= 1 - 1e-7:
        return 0.0, 0.0, l
    h = 0.5 + 0.5 * math.atan2(-bb, -a) / math.pi
    a_, b_ = a / C, bb / C
    C_0, C_mid, C_max = _cs(L, a_, b_)
    if C < C_mid:
        k1 = 0.8 * C_0
        k2 = 1 - k1 / C_mid
        s = 0.8 * C / (k1 + k2 * C)
    else:
        k0, k1 = C_mid, 0.2 * C_mid * C_mid * 1.25 * 1.25 / C_0
        k2 = 1 - k1 / (C_max - C_mid)
        s = 0.8 + 0.2 * (C - k0) / (k1 + k2 * (C - k0))
    return h, s, l


def okhsl_to_srgb(h, s, l):
    if l >= 1:
        return 1.0, 1.0, 1.0
    if l <= 0:
        return 0.0, 0.0, 0.0
    a_, b_ = math.cos(2 * math.pi * h), math.sin(2 * math.pi * h)
    L = _toe_inv(l)
    C_0, C_mid, C_max = _cs(L, a_, b_)
    if s < 0.8:
        t = 1.25 * s
        k1 = 0.8 * C_0
        k2 = 1 - k1 / C_mid
        C = t * k1 / (1 - k2 * t)
    else:
        t = (s - 0.8) / 0.2
        k0, k1 = C_mid, 0.2 * C_mid * C_mid * 1.25 * 1.25 / C_0
        k2 = 1 - k1 / (C_max - C_mid)
        C = k0 + t * k1 / (1 - k2 * t)
    rgb = oklab_to_linear_srgb(L, C * a_, C * b_)
    return tuple(min(1.0, max(0.0, linear_to_srgb(max(0.0, v)))) for v in rgb)


def mirror(rgb, angle):
    """Bram's flip_rotate for one color: hue -> angle - hue (turns), same s and l."""
    h, s, l = srgb_to_okhsl(*rgb)
    return okhsl_to_srgb((angle - h) % 1.0, s, l)


def true_hues(angle):
    """The two hues (turns) that mirror() leaves unchanged at this angle."""
    return (angle / 2) % 1.0, (angle / 2 + 0.5) % 1.0


# ---- the same math for the GPU (GLSL 330) ----
OKHSL_GLSL = """
float srgb_to_lin(float x) { return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4); }
float lin_to_srgb(float x) { return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055; }
float cbrt_(float x) { return sign(x) * pow(abs(x), 1.0 / 3.0); }

vec3 lin_to_oklab(vec3 c) {
    float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    l = cbrt_(l); m = cbrt_(m); s = cbrt_(s);
    return vec3(0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s);
}

vec3 oklab_to_lin(vec3 c) {
    float l = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
    float m = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
    float s = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
    l = l * l * l; m = m * m * m; s = s * s * s;
    return vec3(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
               -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
               -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}

float ok_max_saturation(float a, float b) {
    float k0, k1, k2, k3, k4; vec3 w;
    if (-1.88170328 * a - 0.80936493 * b > 1.0) {
        k0 = 1.19086277; k1 = 1.76576728; k2 = 0.59662641; k3 = 0.75515197; k4 = 0.56771245;
        w = vec3(4.0767416621, -3.3077115913, 0.2309699292);
    } else if (1.81444104 * a - 1.19445276 * b > 1.0) {
        k0 = 0.73956515; k1 = -0.45954404; k2 = 0.08285427; k3 = 0.12541070; k4 = 0.14503204;
        w = vec3(-1.2684380046, 2.6097574011, -0.3413193965);
    } else {
        k0 = 1.35733652; k1 = -0.00915799; k2 = -1.15130210; k3 = -0.50559606; k4 = 0.00692167;
        w = vec3(-0.0041960863, -0.7034186147, 1.7076147010);
    }
    float S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;
    vec3 k = vec3(0.3963377774 * a + 0.2158037573 * b,
                 -0.1055613458 * a - 0.0638541728 * b,
                 -0.0894841775 * a - 1.2914855480 * b);
    vec3 q = 1.0 + S * k;
    vec3 v = q * q * q, v1 = 3.0 * k * q * q, v2 = 6.0 * k * k * q;
    float f = dot(w, v), f1 = dot(w, v1), f2 = dot(w, v2);
    return S - f * f1 / (f1 * f1 - 0.5 * f * f2);
}

vec2 ok_find_cusp(float a, float b) {
    float S = ok_max_saturation(a, b);
    vec3 rgb = oklab_to_lin(vec3(1.0, S * a, S * b));
    float L = cbrt_(1.0 / max(max(rgb.r, rgb.g), rgb.b));
    return vec2(L, L * S);
}

float ok_gamut_intersection(float a, float b, float L1, float C1, float L0, vec2 cusp) {
    if ((L1 - L0) * cusp.y - (cusp.x - L0) * C1 <= 0.0)
        return cusp.y * L0 / (C1 * cusp.x + cusp.y * (L0 - L1));
    float t = cusp.y * (L0 - 1.0) / (C1 * (cusp.x - 1.0) + cusp.y * (L0 - L1));
    float dL = L1 - L0, dC = C1;
    vec3 k = vec3(0.3963377774 * a + 0.2158037573 * b,
                 -0.1055613458 * a - 0.0638541728 * b,
                 -0.0894841775 * a - 1.2914855480 * b);
    vec3 dt = dL + dC * k;
    float L = L0 * (1.0 - t) + t * L1, C = t * C1;
    vec3 q = L + C * k;
    vec3 v = q * q * q, v1 = 3.0 * dt * q * q, v2 = 6.0 * dt * dt * q;
    mat3 W = mat3(4.0767416621, -1.2684380046, -0.0041960863,
                  -3.3077115913, 2.6097574011, -0.7034186147,
                  0.2309699292, -0.3413193965, 1.7076147010);
    vec3 f = W * v - 1.0, f1 = W * v1, f2 = W * v2;
    vec3 u = f1 / (f1 * f1 - 0.5 * f * f2);
    vec3 tt = -f * u;
    float best = 1e30;
    if (u.x >= 0.0) best = min(best, tt.x);
    if (u.y >= 0.0) best = min(best, tt.y);
    if (u.z >= 0.0) best = min(best, tt.z);
    return t + best;
}

const float OK_K1 = 0.206, OK_K2 = 0.03, OK_K3 = (1.0 + 0.206) / (1.0 + 0.03);
float ok_toe(float x) { return 0.5 * (OK_K3 * x - OK_K1 + sqrt((OK_K3 * x - OK_K1) * (OK_K3 * x - OK_K1) + 4.0 * OK_K2 * OK_K3 * x)); }
float ok_toe_inv(float x) { return (x * x + OK_K1 * x) / (OK_K3 * (x + OK_K2)); }

vec2 ok_st_mid(float a, float b) {
    float S = 0.11516993 + 1.0 / (7.44778970 + 4.15901240 * b + a * (-2.19557347 + 1.75198401 * b + a * (
        -2.13704948 - 10.02301043 * b + a * (-4.24894561 + 5.38770819 * b + 4.69891013 * a))));
    float T = 0.11239642 + 1.0 / (1.61320320 - 0.68124379 * b + a * (0.40370612 + 0.90148123 * b + a * (
        -0.27087943 + 0.61223990 * b + a * (0.00299215 - 0.45399568 * b - 0.14661872 * a))));
    return vec2(S, T);
}

vec3 ok_cs(float L, float a, float b) {
    vec2 cusp = ok_find_cusp(a, b);
    float C_max = ok_gamut_intersection(a, b, L, 1.0, L, cusp);
    float k = C_max / min(L * cusp.y / cusp.x, (1.0 - L) * cusp.y / (1.0 - cusp.x));
    vec2 st = ok_st_mid(a, b);
    float Ca = L * st.x, Cb = (1.0 - L) * st.y;
    float C_mid = 0.9 * k * sqrt(sqrt(1.0 / (1.0 / (Ca * Ca * Ca * Ca) + 1.0 / (Cb * Cb * Cb * Cb))));
    Ca = L * 0.4; Cb = (1.0 - L) * 0.8;
    float C_0 = sqrt(1.0 / (1.0 / (Ca * Ca) + 1.0 / (Cb * Cb)));
    return vec3(C_0, C_mid, C_max);
}

vec3 srgb_to_okhsl(vec3 rgb) {
    vec3 lab = lin_to_oklab(vec3(srgb_to_lin(rgb.r), srgb_to_lin(rgb.g), srgb_to_lin(rgb.b)));
    float L = lab.x, l = ok_toe(L);
    float C = length(lab.yz);
    if (C < 1e-5 || L <= 1e-5 || L >= 1.0 - 1e-5) return vec3(0.0, 0.0, l);
    float h = 0.5 + 0.5 * atan(-lab.z, -lab.y) / 3.14159265358979;
    vec3 cs = ok_cs(L, lab.y / C, lab.z / C);
    float s;
    if (C < cs.y) {
        float k1 = 0.8 * cs.x, k2 = 1.0 - k1 / cs.y;
        s = 0.8 * C / (k1 + k2 * C);
    } else {
        float k0 = cs.y, k1 = 0.2 * cs.y * cs.y * 1.5625 / cs.x, k2 = 1.0 - k1 / (cs.z - cs.y);
        s = 0.8 + 0.2 * (C - k0) / (k1 + k2 * (C - k0));
    }
    return vec3(h, s, l);
}

vec3 okhsl_to_srgb(vec3 hsl) {
    float h = hsl.x, s = hsl.y, l = hsl.z;
    if (l >= 1.0) return vec3(1.0);
    if (l <= 0.0) return vec3(0.0);
    float a = cos(6.28318530718 * h), b = sin(6.28318530718 * h);
    float L = ok_toe_inv(l);
    vec3 cs = ok_cs(L, a, b);
    float C;
    if (s < 0.8) {
        float t = 1.25 * s, k1 = 0.8 * cs.x, k2 = 1.0 - k1 / cs.y;
        C = t * k1 / (1.0 - k2 * t);
    } else {
        float t = (s - 0.8) / 0.2, k0 = cs.y, k1 = 0.2 * cs.y * cs.y * 1.5625 / cs.x;
        float k2 = 1.0 - k1 / (cs.z - cs.y);
        C = k0 + t * k1 / (1.0 - k2 * t);
    }
    vec3 lin = max(oklab_to_lin(vec3(L, C * a, C * b)), 0.0);
    return clamp(vec3(lin_to_srgb(lin.r), lin_to_srgb(lin.g), lin_to_srgb(lin.b)), 0.0, 1.0);
}

// Bram Cohen's flip_rotate: hue -> angle - hue, same saturation and lightness
vec3 color_mirror(vec3 rgb, float angle) {
    vec3 hsl = srgb_to_okhsl(clamp(rgb, 0.0, 1.0));
    if (hsl.y <= 0.0) return rgb;  // grays have no hue to mirror
    hsl.x = fract(angle - hsl.x);
    return okhsl_to_srgb(hsl);
}
"""
