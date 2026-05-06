"""
Generate per-class binary masks from the aerial track photo, plus a colored
segmentation overlay and a dedicated track-border mask.

Run:
    python3 segment.py

Outputs (PNG, same resolution as the source):
    mask_asphalt.png  mask_grass.png  mask_curb.png  mask_tree.png
    mask_paint.png    mask_border.png segmentation.png
"""

from pathlib import Path
import numpy as np
from PIL import Image

SRC = Path(__file__).parent / "Karting_Wohlen_5.jpg"
OUT = Path(__file__).parent

# ---------- classification ----------
def classify(rgb: np.ndarray) -> dict[str, np.ndarray]:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)

    greenish = (g > r + 8) & (g > b + 4)
    isgray   = (mx - mn) < 22
    isred    = (r > g + 30) & (r > b + 30)
    isbright = lum > 215
    isdark   = lum < 55

    tree    = greenish & (lum < 90)
    grass   = greenish & ~tree
    curb    = isred
    asphalt = isgray & (lum < 175) & ~grass & ~tree
    paint   = isbright & ~grass & ~tree
    shadow  = isdark & ~tree

    # Treat shadow as asphalt for downstream use (most shadows in the photo fall on tarmac).
    asphalt = asphalt | shadow

    other = ~(tree | grass | curb | asphalt | paint)
    return dict(tree=tree, grass=grass, curb=curb, asphalt=asphalt, paint=paint, other=other)

# ---------- binary morphology (no scipy) ----------
def shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
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

def opening(m, k=1): return dilate(erode(m, k), k)
def closing(m, k=1): return erode(dilate(m, k), k)

def connected_components(m: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Label 4-connected components. Returns (labels, sizes_sorted_desc)."""
    from collections import deque
    h, w = m.shape
    labels = np.zeros((h, w), dtype=np.int32)
    sizes: list[int] = [0]  # label 0 = background
    next_id = 1
    for y in range(h):
        for x in range(w):
            if not m[y, x] or labels[y, x]:
                continue
            q = deque([(y, x)])
            labels[y, x] = next_id
            count = 0
            while q:
                cy, cx = q.popleft()
                count += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = next_id
                        q.append((ny, nx))
            sizes.append(count)
            next_id += 1
    return labels, sizes

# ---------- main ----------
def main():
    img = Image.open(SRC).convert("RGB")
    rgb = np.array(img)
    masks = classify(rgb)

    # Cleanup: close small gaps in asphalt, open to drop speckle. Same idea for grass/tree.
    masks["asphalt"] = opening(closing(masks["asphalt"], k=2), k=1)
    masks["grass"]   = opening(closing(masks["grass"],   k=1), k=1)
    masks["tree"]    = opening(closing(masks["tree"],    k=1), k=1)
    masks["curb"]    = closing(masks["curb"], k=1)

    # Track = connected asphalt component(s). Aggressive closing first to bridge gaps caused by
    # trees overhanging the tarmac (which would otherwise split the track into many small blobs).
    asphalt = masks["asphalt"]
    asphalt_bridged = closing(asphalt, k=4)
    labels, sizes = connected_components(asphalt_bridged)

    import sys
    seed = None
    if len(sys.argv) >= 3:
        seed = (int(sys.argv[2]), int(sys.argv[1]))   # (y, x) from "x y" args (image coords)
    if seed is not None and 0 <= seed[0] < labels.shape[0] and 0 <= seed[1] < labels.shape[1]:
        best_label = int(labels[seed])
        if best_label == 0:
            print(f"warning: seed pixel {seed[::-1]} is not asphalt; falling back to largest blob")
            best_label = int(np.argmax(sizes[1:])) + 1 if len(sizes) > 1 else 0
    else:
        best_label = int(np.argmax(sizes[1:])) + 1 if len(sizes) > 1 else 0

    track = (labels == best_label) & asphalt    # intersect with raw mask so closing-filled gaps are not "drivable"
    track = closing(track, k=1)
    masks["track"] = track

    # Track border = boundary of the track region (the actual usable border).
    track_border = track & ~erode(track, iters=1)
    track_border = dilate(track_border, iters=1)
    masks["track_border"] = track_border

    # Generic asphalt border (kept for reference: borders of every asphalt blob).
    border = asphalt & ~erode(asphalt, iters=1)
    border = dilate(border, iters=1)
    masks["border"] = border

    # Save individual masks
    for name, m in masks.items():
        if name == "other":
            continue
        Image.fromarray((m.astype(np.uint8) * 255), mode="L").save(OUT / f"mask_{name}.png")

    # Color-coded segmentation for visual inspection.
    palette = {
        "asphalt":      ( 80,  80,  80),
        "track":        ( 40,  40,  40),
        "grass":        ( 60, 160,  70),
        "tree":         ( 20,  80,  30),
        "curb":         (220,  40,  40),
        "paint":        (240, 240, 240),
        "border":       (200, 160,   0),
        "track_border": (255, 230,   0),
        "other":        (140, 110,  90),
    }
    seg = np.zeros_like(rgb)
    # Paint in priority order so higher-priority classes win on overlap.
    order = ["other", "grass", "tree", "asphalt", "track", "paint", "curb", "border", "track_border"]
    for name in order:
        if name not in masks: continue
        seg[masks[name]] = palette[name]
    Image.fromarray(seg).save(OUT / "segmentation.png")

    # Print pixel counts so you can sanity-check the classifier.
    total = rgb.shape[0] * rgb.shape[1]
    print(f"image: {rgb.shape[1]}x{rgb.shape[0]} ({total} px)")
    for name, m in masks.items():
        pct = 100.0 * m.sum() / total
        print(f"  {name:8s} {int(m.sum()):>8d} px  {pct:5.1f}%")

if __name__ == "__main__":
    main()
