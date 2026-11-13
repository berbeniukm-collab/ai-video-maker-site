import sys, os, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance

S = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
DUR = 14.70
NF = int(DUR * FPS)

FI = "/usr/share/fonts/opentype/inter/"
def F(name, size): return ImageFont.truetype(FI + name, size)
def SERIF(size): return ImageFont.truetype(S + "/fonts/package/files/playfair-display-cyrillic-400-normal.woff", size)
HEAVY = lambda s: F("Inter-BlackItalic.otf", s)
THIN = lambda s: F("Inter-LightItalic.otf", s)
UI = lambda s: F("Inter-Medium.otf", s)
UIB = lambda s: F("Inter-SemiBold.otf", s)

def clamp(x, a=0., b=1.): return max(a, min(b, x))
def eo(x): x = clamp(x); return 1 - (1 - x) ** 3
def eback(x):
    x = clamp(x); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def prog(t, a, d): return clamp((t - a) / d)

_cache = {}
def src(t):
    i = int(round(t * FPS)) + 1
    i = max(1, min(442, i))
    if i not in _cache:
        if len(_cache) > 40: _cache.clear()
        _cache[i] = Image.open(f"{S}/frames/{i:04d}.jpg").convert("RGB")
    return _cache[i], i

def mask(i):
    p = f"{S}/masks/{i:04d}.png"
    return Image.open(p).convert("L") if os.path.exists(p) else None

def bw(im, contrast=1.15, bright=1.05):
    g = ImageOps.grayscale(im)
    g = ImageEnhance.Contrast(g).enhance(contrast)
    g = ImageEnhance.Brightness(g).enhance(bright)
    return g.convert("RGB")

def muted(im, sat=0.55):
    im = ImageEnhance.Color(im).enhance(sat)
    im = ImageEnhance.Contrast(im).enhance(1.08)
    return ImageEnhance.Brightness(im).enhance(1.04)

def crop_cover(im, cx, cy, zoom, w, h):
    """crop region centred at (cx,cy) of source, aspect w:h, width=1080/zoom"""
    cw = 1080 / zoom; ch = cw * h / w
    if ch > 1920: ch = 1920; cw = ch * w / h
    x0 = clamp(cx - cw / 2, 0, 1080 - cw); y0 = clamp(cy - ch / 2, 0, 1920 - ch)
    return im.resize((w, h), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))

# ---------- background ----------
def make_bg():
    y = np.linspace(0, 1, H)[:, None]
    base = 250 - 14 * y
    arr = np.repeat(base, W, 1)
    return Image.fromarray(np.dstack([arr] * 3).astype(np.uint8))
BG = make_bg()

def sphere(r, dark=165, light=238):
    s = 2 * r
    yy, xx = np.mgrid[0:s, 0:s]
    d = np.hypot(xx - r, yy - r) / r
    # light from top-left
    l = np.hypot(xx - r * 0.55, yy - r * 0.45) / (r * 1.6)
    col = light - (light - dark) * np.clip(l, 0, 1) ** 1.2
    a = np.clip((1 - d) * r / 1.5, 0, 1) * 255
    rgb = np.dstack([col] * 3).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, a.astype(np.uint8)]), "RGBA")
_sph = {}
def put_sphere(img, cx, cy, r, dark=165, light=238):
    r = int(r)
    if r < 4: return
    k = (r // 4 * 4, dark, light)
    if k not in _sph: _sph[k] = sphere(k[0], dark, light)
    sp = _sph[k]
    img.alpha_composite(sp, (int(cx - k[0]), int(cy - k[0])))

def shadow_card(img, card, x, y, radius=46, shadow=0.35, blur=40, offy=28):
    w, h = card.size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius, fill=255)
    pad = blur * 2
    sh = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    sm = Image.new("L", sh.size, 0); sm.paste(m, (pad, pad))
    sm = sm.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: int(v * shadow))
    sh.putalpha(sm)
    img.alpha_composite(sh, (int(x - pad), int(y - pad + offy)))
    c = card.convert("RGBA"); c.putalpha(m)
    img.alpha_composite(c, (int(x), int(y)))

def text_layer(txt, font, fill, alpha=1.0, grad=None):
    bb = font.getbbox(txt)
    w, h = bb[2] - bb[0] + 20, bb[3] - bb[1] + 20
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((10 - bb[0], 10 - bb[1]), txt, font=font, fill=255)
    if grad:
        g = np.linspace(0, 1, h)[:, None]
        c0, c1 = np.array(grad[0]), np.array(grad[1])
        col = (c0[None, None] * (1 - g[..., None]) + c1[None, None] * g[..., None])
        col = np.repeat(col, w, 1).astype(np.uint8)
        L = Image.fromarray(col, "RGB").convert("RGBA")
    else:
        L = Image.new("RGBA", (w, h), fill + (255,))
    L.putalpha(m.point(lambda v: int(v * alpha)))
    return L

def draw_text(img, txt, font, x, y, fill=(40, 40, 40), alpha=1.0, anchor="l", grad=None, sx=1.0):
    if alpha <= 0.01: return 0
    L = text_layer(txt, font, fill, alpha, grad)
    if sx != 1.0: L = L.resize((int(L.width * sx), L.height), Image.BICUBIC)
    xx = x if anchor == "l" else (x - L.width / 2 if anchor == "c" else x - L.width)
    img.alpha_composite(L, (int(xx), int(y)))
    return L.width

def title(img, t, t0, heavy, thin, y, x=70, hs=150, ts=84, thin_first=False, center=False):
    """Pinterest-style: HEAVY italic word + light italic word."""
    p = eo(prog(t, t0, 0.35)); p2 = eo(prog(t, t0 + 0.12, 0.35))
    hf, tf = HEAVY(hs), THIN(ts)
    hw = hf.getbbox(heavy)[2]; tw = tf.getbbox(thin)[2] if thin else 0
    if center: x = (W - hw - tw - 24) / 2
    dark = ((70, 70, 70), (25, 25, 25))
    if thin_first and thin:
        draw_text(img, thin, tf, x - 40 * (1 - p), y + hs * 0.45, (110, 110, 110), p)
        draw_text(img, heavy, hf, x + tw + 24 - 40 * (1 - p2), y, alpha=p2, grad=dark)
    else:
        draw_text(img, heavy, hf, x - 50 * (1 - p), y, alpha=p, grad=dark)
        if thin: draw_text(img, thin, tf, x + hw + 24 - 40 * (1 - p2), y + hs * 0.45, (110, 110, 110), p2)

def big_serif(img, t, t0, word, y, size=360, alpha=1.0, max_w=1000):
    p = eo(prog(t, t0, 0.5))
    f = SERIF(size)
    w = f.getbbox(word)[2]
    sx = min(1.0, max_w / w) * 0.86 if w > max_w else 0.86   # condensed look
    s = 1.0 + 0.04 * (t - t0)
    f2 = SERIF(int(size * s))
    draw_text(img, word, f2, W / 2, y - (s - 1) * size * 0.4 + 30 * (1 - p), alpha=p * alpha,
              anchor="c", grad=((125, 125, 125), (45, 45, 45)), sx=sx)

def prompt_bar(img, t, t0, text, y, btn="Готово"):
    p = eo(prog(t, t0, 0.3))
    if p <= 0: return
    w, h = 900, 120
    x = (W - w) / 2
    bar = Image.new("RGBA", (w, h), (252, 252, 252, 255))
    d = ImageDraw.Draw(bar)
    n = int(len(text) * clamp((t - t0 - 0.15) / (0.04 * len(text) + 0.01)))
    shown = text[:n]
    d.text((36, 38), shown + ("|" if (int(t * 4) % 2 == 0 and n < len(text)) else ""), font=UI(38), fill=(90, 90, 90))
    bw_, bh = 190, 70
    d.rounded_rectangle((w - bw_ - 22, (h - bh) / 2, w - 22, (h + bh) / 2), 35, fill=(205, 205, 205))
    d.text((w - 22 - bw_ / 2, h / 2), btn, font=UIB(32), fill=(60, 60, 60), anchor="mm")
    sh = Image.new("RGBA", (w, h)); sh.paste(bar)
    yy = y + 40 * (1 - p)
    tmp = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_card(tmp, bar.convert("RGB"), x, yy, radius=60, shadow=0.25, blur=26, offy=14)
    tmp.putalpha(tmp.getchannel("A").point(lambda v: int(v * p)))
    img.alpha_composite(tmp)

def chip(img, t, t0, text, x, y):
    p = eback(prog(t, t0, 0.35))
    if p <= 0: return
    f = UIB(40)
    tw = f.getbbox(text)[2]
    w, h = tw + 64, 84
    c = Image.new("RGB", (w, h), (250, 250, 250))
    ImageDraw.Draw(c).text((32, h / 2), text, font=f, fill=(50, 50, 50), anchor="lm")
    sc = max(0.01, p)
    c = c.resize((max(1, int(w * sc)), max(1, int(h * sc))))
    shadow_card(img, c, x + w * (1 - sc) / 2, y + h * (1 - sc) / 2, radius=int(42 * sc), shadow=0.22, blur=18, offy=10)

# ---------- scenes ----------
# (start, end) in output time; audio is untouched so text follows speech
def scene_fullbleed_behind(img, t, ts, src_t, word, wy, heavy, thin, ty, t_title, push=0.03, extra=None):
    fr, i = src(src_t)
    z = 1.0 + push * (t - ts)
    fr = bw(fr)
    if z != 1: fr = crop_cover(fr, 540, 960, z, W, H)
    m = mask(i)
    if m is not None and z != 1: m = crop_cover(m.convert("RGB"), 540, 960, z, W, H).convert("L")
    base = fr.convert("RGBA")
    if m is not None:
        wash = Image.new("RGBA", (W, H), (244, 244, 244, 190))
        base.alpha_composite(wash)
    big_serif(base, t, ts + 0.05, word, wy, size=440, alpha=1.0)
    if m is not None:
        m = m.filter(ImageFilter.GaussianBlur(2))
        fg = fr.convert("RGBA"); fg.putalpha(m)
        base.alpha_composite(fg)
    # white haze at top so titles read like reference
    g = np.clip(1 - np.linspace(0, 1, H) / 0.28, 0, 1)[:, None] * 230
    haze = Image.new("RGBA", (W, H), (248, 248, 248, 0)); haze.putalpha(Image.fromarray(np.repeat(g, W, 1).astype(np.uint8)))
    base.alpha_composite(haze)
    g2 = np.clip((np.linspace(0, 1, H) - 0.70) / 0.22, 0, 1)[:, None] * 255
    haze2 = Image.new("RGBA", (W, H), (245, 245, 245, 0)); haze2.putalpha(Image.fromarray(np.repeat(g2, W, 1).astype(np.uint8)))
    base.alpha_composite(haze2)
    img.alpha_composite(base)
    if extra: extra(img, t)

def scene_card(img, t, ts, te, src_t0, cx, cy, zoom, heavy, thin, serif=None, spheres=(), bar=None, chips=(),
               card_w=860, card_h=1000, card_y=560, thin_first=False, title_y=170, bwcard=False):
    p = eback(prog(t, ts, 0.45))
    for (sx, sy, r) in spheres:
        k = eo(prog(t, ts, 0.6))
        put_sphere(img, sx + 30 * math.sin(t * 1.3 + sx), sy - 40 * k + 20 * math.cos(t + sy), r * (0.6 + 0.4 * k))
    if serif: big_serif(img, t, ts, serif, card_y - 250, size=380, alpha=0.9)
    fr, _ = src(src_t0 + (t - ts))
    z = zoom * (1 + 0.05 * (t - ts))
    cw, ch = int(card_w * (0.88 + 0.12 * p)), int(card_h * (0.88 + 0.12 * p))
    c = crop_cover(fr, cx, cy, z, cw, ch)
    c = bw(c) if bwcard else muted(c)
    yoff = 120 * (1 - eo(prog(t, ts, 0.45)))
    shadow_card(img, c, (W - cw) / 2, card_y + (card_h - ch) / 2 + yoff)
    title(img, t, ts + 0.05, heavy, thin, title_y, thin_first=thin_first, center=True)
    if bar: prompt_bar(img, t, ts + bar[2], bar[0], bar[1])
    for (txt, x, y, at) in chips: chip(img, t, at, txt, x, y)

SCENES = []
def scene(a, b):
    def deco(fn): SCENES.append((a, b, fn)); return fn
    return deco

@scene(0.0, 0.95)
def s1(img, t):
    def ex(img, t):
        title(img, t, 0.0, "СЕГОДНЯ", "в меню", 120, hs=140, ts=80, center=True)
    scene_fullbleed_behind(img, t, 0.0, min(t, 0.45), "МЕНЮ", 330, "", "", 0, 0, extra=ex)

@scene(0.95, 3.25)
def s2(img, t):
    scene_card(img, t, 0.95, 3.25, 1.4, 330, 800, 1.4, "КАША", "гречневая", thin_first=True,
               spheres=[(930, 1500, 210), (150, 520, 90)],
               bar=("Гречневая каша с молоком", 1640, 0.9))

@scene(3.25, 6.70)
def s3(img, t):
    scene_card(img, t, 3.25, 6.70, 3.3, 560, 560, 1.75, "ТВОРОГ", "", serif=None,
               spheres=[(140, 1550, 230), (960, 470, 110)],
               chips=[("с пшёнкой", 120, 1390, 4.94), ("+ семечки", 600, 1480, 5.82)],
               bar=("Творожная масса", 1650, 0.15))

@scene(6.70, 7.80)
def s4(img, t):
    scene_card(img, t, 6.70, 7.80, 5.6, 820, 640, 1.9, "ЗЛАКИ", "овсяные", thin_first=True,
               spheres=[(900, 1560, 180)], card_h=900, card_y=600)

@scene(7.80, 9.15)
def s5(img, t):
    scene_card(img, t, 7.80, 9.15, 8.2, 760, 560, 1.5, "САЛАТ", "крабовый", thin_first=True,
               spheres=[(170, 1500, 200), (920, 500, 80)], card_h=950)

@scene(9.15, 10.20)
def s6(img, t):
    # DRIVING-style: huge serif word with a wide card overlapping it
    big_serif(img, t, 9.15, "ХЛЕБ", 720, size=520, alpha=0.95)
    p = eback(prog(t, 9.3, 0.4))
    fr, _ = src(9.35 + 0.3 * (t - 9.15))
    cw, ch = int(820 * (0.85 + 0.15 * p)), int(560 * (0.85 + 0.15 * p))
    c = muted(crop_cover(fr, 650, 450, 1.3 * (1 + 0.05 * (t - 9.15)), cw, ch))
    if p > 0: shadow_card(img, c, (W - cw) / 2, 1080 + 80 * (1 - p))
    put_sphere(img, 160, 1660, 120 * eo(prog(t, 9.2, 0.5)))

@scene(10.20, 11.40)
def s7(img, t):
    k = eo(prog(t, 10.2, 0.6))
    put_sphere(img, 540, 1000 - 30 * k, 330 * (0.5 + 0.5 * k), dark=150, light=240)
    put_sphere(img, 180, 420, 90 * k); put_sphere(img, 930, 1580, 140 * k)
    title(img, t, 10.20, "И", "это", 860, hs=200, ts=110, center=True)
    p = eo(prog(t, 11.0, 0.3))
    draw_text(img, "ВСЁ", HEAVY(int(260 + 60 * p)), W / 2, 1060 + 30 * (1 - p), alpha=p, anchor="c",
              grad=((70, 70, 70), (20, 20, 20)))

@scene(11.40, 14.70)
def s8(img, t):
    def ex(img, t):
        title(img, t, 11.44, "СЕГОДНЯ", "", 110, hs=110, center=True)
        title(img, t, 12.10, "У МАХИ", "в гараже", 1650, hs=128, ts=72, center=True)
    scene_fullbleed_behind(img, t, 11.40, t, "ГАРАЖ", 300, "", "", 0, 0, push=0.025, extra=ex)

def render_frame(n):
    t = n / FPS
    img = BG.convert("RGBA")
    cur = None
    for a, b, fn in SCENES:
        if a <= t < b: cur = (a, b, fn)
    if cur is None: cur = SCENES[-1]
    a, b, fn = cur
    fn(img, t)
    # cut punch: slight zoom on first frames of each scene + white flash
    d = t - a
    if a > 0 and d < 0.2:
        z = 1 + 0.06 * (1 - eo(d / 0.2))
        img = img.resize((int(W * z), int(H * z)), Image.BICUBIC).crop(
            (int((W * z - W) / 2), int((H * z - H) / 2), int((W * z - W) / 2) + W, int((H * z - H) / 2) + H))
        if d < 0.1:
            fl = Image.new("RGBA", (W, H), (255, 255, 255, int(200 * (1 - d / 0.1))))
            img.alpha_composite(fl)
    img = img.convert("RGB")
    # light grain
    return img

if __name__ == "__main__":
    if sys.argv[1] == "still":
        for tt in sys.argv[2:]:
            render_frame(int(float(tt) * FPS)).save(f"{S}/out/still_{tt}.jpg", quality=88)
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        rng = np.random.default_rng(1)
        for n in range(a, b):
            im = render_frame(n)
            arr = np.asarray(im).astype(np.int16)
            arr = np.clip(arr + rng.normal(0, 3, (H, W, 1)), 0, 255).astype(np.uint8)
            sys.stdout.buffer.write(arr.tobytes())
