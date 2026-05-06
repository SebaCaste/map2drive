from PIL import Image
import math

INPUT_IMAGE = "track_painted.png"
OUTPUT_IMAGE = "track_clean_color_clean.png"

CLASS_TABLE = [
    ("track",        (40,  40,  40)),
    ("runoff",       (110, 110, 110)),
    ("parking_road", (160, 160, 160)),
    ("curb",         (220, 40,  40)),
    ("barrier",      (255, 120, 0)),
    ("grass",        (60,  160, 70)),
    ("tree_canopy",  (20,  80,  30)),
    ("building",     (200, 170, 120)),
    ("vehicle",      (80,  60,  200)),
    ("rail",         (180, 180, 180)),
    ("background",   (0,   0,   0)),
]
def is_yellow(r, g, b):
    # Detect bright yellow (tolerant range)
    return (
        r > 180 and
        g > 180 and
        b < 120 and
        abs(r - g) < 80   # ensures it's not greenish
    )
def closest_color(r, g, b):

    # --- HARD OVERRIDE ---
    if is_yellow(r, g, b):
        return (40, 40, 40)  # force to track

    best_rgb = None
    best_dist = float("inf")

    for name, (cr, cg, cb) in CLASS_TABLE:
        dr = r - cr
        dg = g - cg
        db = b - cb

        dist = dr * dr + dg * dg + db * db

        # Slight bias toward track
        if name == "track":
            dist *= 0.8

        if dist < best_dist:
            best_dist = dist
            best_rgb = (cr, cg, cb)

    return best_rgb

def process():
    img = Image.open(INPUT_IMAGE).convert("RGB")
    pixels = img.load()

    width, height = img.size

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            pixels[x, y] = closest_color(r, g, b)

        if y % 50 == 0:
            print(f"Processing row {y}/{height}")

    img.save(OUTPUT_IMAGE)
    print("Saved:", OUTPUT_IMAGE)

if __name__ == "__main__":
    process()