"""
Clean up the Gemini segmentation PNG:
  - reclassify every pixel to the nearest palette color
  - regularize the track to a uniform width (skeleton dilated by median half-width)
  - drop any tree/building patches that fall inside the cleaned track
  - draw a yellow centerline along the track skeleton

Run:
    python3 clean_segmentation.py

Outputs:
    Gemini_segmentation_cleaned.png   (same size as the input)
"""

from pathlib import Path
import numpy as np
from PIL import Image

SRC = Path(__file__).parent / "Gemini_Generated_Image_96t4n296t4n296t4.png"
OUT = Path(__file__).parent / "Gemini_segmentation_cleaned.png"

# Palette must match vlm_prompt.md / CLASS_TABLE in index.html.
CLASSES = [
    ("track",        ( 40,  40,  40)),
    ("runoff",       (110, 110, 110)),
    ("parking_road", (160, 160, 160)),
    ("curb",         (220,  40,  40)),
    ("barrier",      (255, 120,   0)),
    ("grass",        ( 60, 160,  70)),
    ("tree_canopy",  ( 20,  80,  30)),
    ("building",     (200, 170, 120)),
    ("vehicle",      ( 80,  60, 200)),
    ("rail",         (180, 180, 180)),
    ("background",   (  0,   0,   0)),
]
NAME2COLOR = {n: c for n, c in CLASSES}
PALETTE = np.array([c for _, c in CLASSES], dtype=np.int16)
NAMES = [n for n, _ in CLASSES]

LINE_COLOR = (255, 230, 0)   # centerline color

# ---------- helpers ----------
def shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    ys = slice(max(0, dy), h + min(0, dy))
    xs = slice(max(0, dx), w + min(0, dx))
    yt = slice(max(0, -dy), h + min(0, -dy))
    xt = slice(max(0, -dx), w + min(0, -dx))
    out[ys, xs] = a[yt, xt]
    return out

def dilate(m, iters=1):
    for _ in range(iters):
        m = m | shift(m, 1, 0) | shift(m, -1, 0) | shift(m, 0, 1) | shift(m, 0, -1)
    return m

def erode(m, iters=1):
    for _ in range(iters):
        m = m & shift(m, 1, 0) & shift(m, -1, 0) & shift(m, 0, 1) & shift(m, 0, -1)
    return m

def closing(m, k=1): return erode(dilate(m, k), k)
def opening(m, k=1): return dilate(erode(m, k), k)

def chamfer_dt(mask):
    """Distance-from-zeros transform inside `mask` (chamfer 3-4 units)."""
    INF = 1e9
    dt = np.where(mask, INF, 0.0).astype(np.float32)
    h, w = mask.shape
    # Forward
    for y in range(h):
        for x in range(w):
            if not mask[y, x]: continue
            v = dt[y, x]
            if x > 0:                 v = min(v, dt[y, x-1] + 3)
            if y > 0:                 v = min(v, dt[y-1, x] + 3)
            if x > 0 and y > 0:       v = min(v, dt[y-1, x-1] + 4)
            if x < w-1 and y > 0:     v = min(v, dt[y-1, x+1] + 4)
            dt[y, x] = v
    # Backward
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

def chamfer_dt_from(seeds, region):
    """Distance from seed pixels (==1) outward, restricted to `region` pixels (or full image)."""
    INF = 1e9
    h, w = seeds.shape
    dt = np.where(seeds, 0.0, INF).astype(np.float32)
    if region is not None:
        dt[~region] = INF   # we still write to non-region cells but cap them; fine to leave INF
    for y in range(h):
        for x in range(w):
            v = dt[y, x]
            if x > 0:                 v = min(v, dt[y, x-1] + 3)
            if y > 0:                 v = min(v, dt[y-1, x] + 3)
            if x > 0 and y > 0:       v = min(v, dt[y-1, x-1] + 4)
            if x < w-1 and y > 0:     v = min(v, dt[y-1, x+1] + 4)
            dt[y, x] = v
    for y in range(h-1, -1, -1):
        for x in range(w-1, -1, -1):
            v = dt[y, x]
            if x < w-1:               v = min(v, dt[y, x+1] + 3)
            if y < h-1:               v = min(v, dt[y+1, x] + 3)
            if x < w-1 and y < h-1:   v = min(v, dt[y+1, x+1] + 4)
            if x > 0 and y < h-1:     v = min(v, dt[y+1, x-1] + 4)
            dt[y, x] = v
    return dt

def skeleton_local_maxima(dt, mask, min_dt=6):
    """Skeleton = pixels that are >= all 8 neighbors and DT >= min_dt."""
    sk = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0: continue
    # Vectorized: a pixel is a max if dt[p] >= shift(dt, ...) for all 8 directions.
    is_max = np.ones_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0: continue
            is_max &= dt >= shift(dt, dy, dx)
    sk = mask & is_max & (dt >= min_dt)
    return sk

# ---------- main ----------
def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    h, w, _ = img.shape

    # 1. Reclassify every pixel by nearest palette color.
    flat = img.reshape(-1, 3).astype(np.int16)
    diff = flat[:, None, :] - PALETTE[None, :, :]
    d2 = (diff * diff).sum(axis=2)
    labels = d2.argmin(axis=1).reshape(h, w).astype(np.uint8)

    # 2. Drivable ribbon = track ∪ curb ∪ barrier. Gemini's stylized output paints most of the
    #    racing surface in red/orange hues that nearest-color routes to curb/barrier rather than
    #    track, so we union the three. Then keep only the largest connected component to drop
    #    isolated curb/barrier specks elsewhere on the map.
    drivable = (
        (labels == NAMES.index("track"))
      | (labels == NAMES.index("curb"))
      | (labels == NAMES.index("barrier"))
    )
    drivable = closing(drivable, k=3)
    drivable = opening(drivable, k=1)

    # Largest connected component
    from collections import deque
    lab = np.zeros(drivable.shape, dtype=np.int32)
    sizes = [0]
    nid = 1
    H, W = drivable.shape
    for y in range(H):
        for x in range(W):
            if not drivable[y, x] or lab[y, x]: continue
            q = deque([(y, x)]); lab[y, x] = nid; cnt = 0
            while q:
                cy, cx = q.popleft(); cnt += 1
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < H and 0 <= nx < W and drivable[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = nid; q.append((ny, nx))
            sizes.append(cnt); nid += 1
    best = int(np.argmax(sizes[1:])) + 1 if len(sizes) > 1 else 0
    track = (lab == best)
    print(f"largest drivable component: {int(track.sum())} px (kept), {len(sizes)-1} blobs total")

    # Fill SMALL holes inside the track ribbon (tree canopies, vehicles, building specks that
    # ended up fully surrounded by track). Skip the big infield blob — it's enclosed by the
    # track loop and we don't want to fill that.
    inv = ~track
    lab2 = np.zeros_like(track, dtype=np.int32)
    sizes2 = [0]; touches_border = [False]; nid2 = 1
    for y in range(H):
        for x in range(W):
            if not inv[y, x] or lab2[y, x]: continue
            q2 = deque([(y, x)]); lab2[y, x] = nid2; cnt = 0; tb = False
            while q2:
                cy, cx = q2.popleft(); cnt += 1
                if cy == 0 or cy == H-1 or cx == 0 or cx == W-1: tb = True
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < H and 0 <= nx < W and inv[ny, nx] and not lab2[ny, nx]:
                        lab2[ny, nx] = nid2; q2.append((ny, nx))
            sizes2.append(cnt); touches_border.append(tb); nid2 += 1
    SMALL_HOLE_PX = 4000   # ~ a 60x60 patch — enough for trees/vehicles, not the infield
    fill = np.zeros_like(track)
    n_filled = 0
    for i in range(1, len(sizes2)):
        if not touches_border[i] and sizes2[i] < SMALL_HOLE_PX:
            fill |= (lab2 == i); n_filled += sizes2[i]
    if n_filled:
        print(f"filled {n_filled} hole pixel(s) inside the track ribbon (small patches)")
    track = track | fill

    # 3. Distance transform inside track + skeleton.
    print("computing distance transform inside track...")
    dt = chamfer_dt(track)
    print("computing skeleton...")
    sk = skeleton_local_maxima(dt, track, min_dt=6)
    skel_dts = dt[sk]
    if skel_dts.size == 0:
        raise SystemExit("no skeleton points found — track mask is empty?")
    median_chamfer = float(np.median(skel_dts))
    median_hw_px = median_chamfer / 3.0
    print(f"skeleton points: {sk.sum()}  median half-width: {median_hw_px:.1f} px")

    # 4. Build a uniform-width track by dilating the skeleton to the median half-width.
    print("computing distance from skeleton...")
    dt_from_skel = chamfer_dt_from(sk, region=None)
    clean_track = dt_from_skel <= median_chamfer

    # 5. Render the cleaned image:
    #    - start from the reclassified palette image
    #    - paint clean_track region with the track color (this also wipes any tree/building/grass
    #      patches that fell inside the track)
    #    - keep everything else as-is from the reclassified image
    out = PALETTE[labels].astype(np.uint8)             # (h, w, 3)
    out[clean_track] = NAME2COLOR["track"]

    # Pixels that were part of the *original* track but lie outside the clean track get painted as
    # grass — otherwise we'd see jagged holes around the trimmed-down sections.
    trimmed = track & ~clean_track
    out[trimmed] = NAME2COLOR["grass"]

    # 6. Draw the yellow centerline on top, thickened proportional to track width.
    line_thickness = max(2, int(round(median_hw_px * 0.18)))
    line = dilate(sk, iters=line_thickness)
    out[line] = LINE_COLOR

    Image.fromarray(out).save(OUT)
    print(f"saved {OUT.name}  ({w}x{h})")

    # Quick stats
    total = h * w
    print(f"  original track:  {int(track.sum()):>8d} px ({100*track.sum()/total:.1f}%)")
    print(f"  clean    track:  {int(clean_track.sum()):>8d} px ({100*clean_track.sum()/total:.1f}%)")
    print(f"  trimmed -> grass:{int(trimmed.sum()):>8d} px")

if __name__ == "__main__":
    main()
