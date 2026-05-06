"""
Polish the manually-fixed clearmap.png:
  - identify the track ribbon (the connected dark + yellow region)
  - clean any non-track patches that ended up inside the ribbon
  - redraw the yellow centerline with uniform thickness
  - paint curbs at corners using a curvature analysis of the centerline:
        inner side  -> red    (220, 40, 40)
        outer side  -> orange (255, 120, 0)

Only pixels inside the track ribbon (or in the curb band along its edge) are modified — every
other pixel is copied verbatim from clearmap.png so the user's hand-fixed colours are kept.

Outputs: track_painted.png
"""
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
CLEAR = HERE / "clearmap.png"
OUT = HERE / "track_painted.png"

TRACK_COLOR  = ( 40,  40,  40)
LINE_COLOR   = (255, 230,   0)
INNER_COLOR  = (220,  40,  40)
OUTER_COLOR  = (255, 120,   0)

# ====== TUNABLES ======
CENTERLINE_THICKNESS_PX = 1     # final yellow line width in pixels (1 = thinnest possible)
CURB_THICKNESS_PX       = 4     # how many pixels OUTSIDE the track edge the curb extends
# ======================
CURVATURE_THRESHOLD  = 0.07 # |bend|/SMOOTH_RADIUS — bigger = curbs only on sharper corners
SMOOTH_RADIUS_PX     = 28   # smoothing window for skeleton -> curvature estimation
HOLE_FILL_MAX_PX     = 4000 # don't fill holes bigger than this (preserves the infield)

# ---------- morphology ----------
def shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    ys = slice(max(0, dy), h + min(0, dy))
    xs = slice(max(0, dx), w + min(0, dx))
    yt = slice(max(0, -dy), h + min(0, -dy))
    xt = slice(max(0, -dx), w + min(0, -dx))
    out[ys, xs] = a[yt, xt]
    return out

def dilate(m, iters=1, eight=True):
    for _ in range(iters):
        m2 = m | shift(m, 1, 0) | shift(m, -1, 0) | shift(m, 0, 1) | shift(m, 0, -1)
        if eight:
            m2 |= shift(m, 1, 1) | shift(m, -1, 1) | shift(m, 1, -1) | shift(m, -1, -1)
        m = m2
    return m

def erode(m, iters=1):
    for _ in range(iters):
        m = m & shift(m, 1, 0) & shift(m, -1, 0) & shift(m, 0, 1) & shift(m, 0, -1)
    return m

def closing(m, k=1): return erode(dilate(m, k, eight=False), k)

def chamfer_dt(mask):
    INF = 1e9
    dt = np.where(mask, INF, 0.0).astype(np.float32)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x]: continue
            v = dt[y, x]
            if x > 0:                 v = min(v, dt[y, x-1] + 3)
            if y > 0:                 v = min(v, dt[y-1, x] + 3)
            if x > 0 and y > 0:       v = min(v, dt[y-1, x-1] + 4)
            if x < w-1 and y > 0:     v = min(v, dt[y-1, x+1] + 4)
            dt[y, x] = v
    for y in range(h-1, -1, -1):
        for x in range(w-1, -1, -1):
            if not mask[y, x]: continue
            v = dt[y, x]
            if x < w-1:               v = min(v, dt[y, x+1] + 3)
            if y < h-1:               v = min(v, dt[y+1, x] + 3)
            if x < w-1 and y < h-1:   v = min(v, dt[y+1, x+1] + 4)
            if x > 0 and y < h-1:     v = min(v, dt[y+1, x-1] + 4)
            dt[y, x] = v
    return dt

def skeleton_local_max(dt, mask, min_dt=4):
    is_max = np.ones_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0: continue
            is_max &= dt >= shift(dt, dy, dx)
    return mask & is_max & (dt >= min_dt)

def zhang_suen_thin(mask):
    """Vectorised Zhang-Suen thinning: returns a 1-pixel-wide skeleton of `mask`."""
    img = mask.copy()
    while True:
        removed = False
        for sub in (1, 2):
            p2 = shift(img, -1,  0); p3 = shift(img, -1,  1); p4 = shift(img,  0,  1)
            p5 = shift(img,  1,  1); p6 = shift(img,  1,  0); p7 = shift(img,  1, -1)
            p8 = shift(img,  0, -1); p9 = shift(img, -1, -1)
            B = (p2.astype(np.int8) + p3 + p4 + p5 + p6 + p7 + p8 + p9)
            nbs = [p2, p3, p4, p5, p6, p7, p8, p9]
            A = np.zeros_like(B)
            for i in range(8):
                A += ((~nbs[i]) & nbs[(i + 1) % 8]).astype(np.int8)
            cond = img & (B >= 2) & (B <= 6) & (A == 1)
            if sub == 1:
                cond &= ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                cond &= ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            if cond.any():
                img = img & ~cond
                removed = True
        if not removed:
            return img

def largest_component(mask, scoring=None):
    H, W = mask.shape
    lab = np.zeros(mask.shape, dtype=np.int32); sizes = [0]; scores = [0]; nid = 1
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or lab[y, x]: continue
            q = deque([(y, x)]); lab[y, x] = nid; cnt = 0; sc = 0
            while q:
                cy, cx = q.popleft(); cnt += 1
                if scoring is not None and scoring[cy, cx]: sc += 1
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = nid; q.append((ny, nx))
            sizes.append(cnt); scores.append(sc); nid += 1
    if nid == 1: return np.zeros_like(mask)
    metric = scores if scoring is not None else sizes
    best = int(np.argmax(metric[1:])) + 1
    return lab == best

# ---------- main ----------
def main():
    img = np.array(Image.open(CLEAR).convert("RGB"))
    H, W, _ = img.shape
    R = img[..., 0].astype(np.int16)
    G = img[..., 1].astype(np.int16)
    B = img[..., 2].astype(np.int16)
    lum = 0.299 * R + 0.587 * G + 0.114 * B

    # Yellow centerline: high R, high G, low B.
    yellow = (R > 200) & (G > 180) & (B < 120)
    # Track-dark: very dark grayscale (tarmac).
    dark = (lum < 60) & (np.abs(R - G) < 25) & (np.abs(G - B) < 25)
    # Candidate ribbon = yellow OR dark, then close small gaps.
    candidate = closing(yellow | dark, k=2)

    # Pick the connected component that contains the most yellow centerline (= the actual track).
    track = largest_component(candidate, scoring=yellow)
    print(f"track ribbon: {int(track.sum())} px ({100*track.sum()/(H*W):.1f}% of image)")

    # Fill small holes inside the ribbon (anything fully enclosed and below threshold size).
    inv = ~track
    lab2 = np.zeros_like(track, dtype=np.int32); sizes2 = [0]; tb = [False]; nid2 = 1
    for y in range(H):
        for x in range(W):
            if not inv[y, x] or lab2[y, x]: continue
            q = deque([(y, x)]); lab2[y, x] = nid2; cnt = 0; touch = False
            while q:
                cy, cx = q.popleft(); cnt += 1
                if cy in (0, H-1) or cx in (0, W-1): touch = True
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < H and 0 <= nx < W and inv[ny, nx] and not lab2[ny, nx]:
                        lab2[ny, nx] = nid2; q.append((ny, nx))
            sizes2.append(cnt); tb.append(touch); nid2 += 1
    fill = np.zeros_like(track); n_fill = 0
    for i in range(1, len(sizes2)):
        if not tb[i] and sizes2[i] < HOLE_FILL_MAX_PX:
            fill |= (lab2 == i); n_fill += sizes2[i]
    if n_fill: print(f"filled {n_fill} hole pixel(s) inside the ribbon (small patches)")
    track = track | fill

    # ---- centerline: 1-pixel-wide skeleton via Zhang-Suen thinning ----
    print("thinning track to centerline (Zhang-Suen)...")
    sk = zhang_suen_thin(track)
    skel_yx = np.argwhere(sk)
    if skel_yx.shape[0] == 0:
        raise SystemExit("no skeleton found")
    print(f"skeleton: {skel_yx.shape[0]} points (thinned 1-px)")

    # ---- curvature direction at every skeleton point ----
    # bend(P) = centroid(neighbours within R) - P  →  zero on straights, points to inner side on corners
    N = skel_yx.shape[0]
    bend = np.zeros((N, 2), dtype=np.float32)
    bend_mag = np.zeros(N, dtype=np.float32)
    R2 = SMOOTH_RADIUS_PX * SMOOTH_RADIUS_PX
    chunk = 256
    skel_f = skel_yx.astype(np.float32)
    for i in range(0, N, chunk):
        a = skel_f[i:i+chunk]                              # (c, 2)
        d = skel_f[None, :, :] - a[:, None, :]            # (c, N, 2)
        d2 = (d * d).sum(axis=2)
        m = (d2 < R2).astype(np.float32)
        n = m.sum(axis=1, keepdims=True).clip(1)
        cen = (m[:, :, None] * skel_f[None, :, :]).sum(axis=1) / n
        v = cen - a
        bend[i:i+chunk] = v
        bend_mag[i:i+chunk] = np.linalg.norm(v, axis=1)
    curvature = bend_mag / SMOOTH_RADIUS_PX

    # ---- curb band: pixels OUTSIDE the track within CURB_THICKNESS_PX of its edge ----
    curb_band = dilate(track, iters=CURB_THICKNESS_PX, eight=True) & ~track

    # For every curb-band pixel find its nearest skeleton point, then decide inner/outer
    # using the bend direction at that skeleton point.
    print("classifying curb-band pixels...")
    band_yx = np.argwhere(curb_band)
    if band_yx.shape[0]:
        nearest = np.zeros(band_yx.shape[0], dtype=np.int32)
        chunk = 1024
        band_f = band_yx.astype(np.float32)
        for i in range(0, band_yx.shape[0], chunk):
            e = band_f[i:i+chunk]
            d = e[:, None, :] - skel_f[None, :, :]
            d2 = (d * d).sum(axis=2)
            nearest[i:i+chunk] = d2.argmin(axis=1)
        rel = band_f - skel_f[nearest]
        side = (rel * bend[nearest]).sum(axis=1)
        curve = curvature[nearest] > CURVATURE_THRESHOLD
        inner = curve & (side > 0)
        outer = curve & (side < 0)
    else:
        inner = outer = np.zeros(0, dtype=bool)

    inner_mask = np.zeros_like(track)
    outer_mask = np.zeros_like(track)
    inner_mask[band_yx[:, 0], band_yx[:, 1]] = inner
    outer_mask[band_yx[:, 0], band_yx[:, 1]] = outer

    # ---- compose output: copy clearmap, only retouch the ribbon + curb band ----
    out = img.copy()
    out[track] = TRACK_COLOR
    out[outer_mask] = OUTER_COLOR
    out[inner_mask] = INNER_COLOR        # paint inner last so it wins on overlapping pixels
    if CENTERLINE_THICKNESS_PX <= 1:
        line = sk
    else:
        line = dilate(sk, iters=CENTERLINE_THICKNESS_PX - 1, eight=True)
    out[line] = LINE_COLOR

    Image.fromarray(out).save(OUT)
    print(f"saved {OUT.name}  ({W}x{H})")
    print(f"  inner-curb pixels: {int(inner_mask.sum())}")
    print(f"  outer-curb pixels: {int(outer_mask.sum())}")

if __name__ == "__main__":
    main()
