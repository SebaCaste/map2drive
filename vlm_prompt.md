# VLM segmentation prompt — Karting Wohlen aerial

Paste everything below the `---` line into the VLM together with `Karting_Wohlen_5.jpg`.
The image resolution is **800 × 449 pixels**. All coordinates the model returns must be in
that pixel space, with origin (0,0) at the **top-left** corner, +x right, +y down.

---

You are an expert aerial-imagery analyst. You will receive ONE top-down photograph of a
go-kart racing facility ("Karting Wohlen"). Your job is to produce a precise semantic
segmentation of the scene plus a structured inventory of discrete objects.

The image is **800 pixels wide × 449 pixels tall**. All coordinates you output must be
integers in this pixel space. Origin (0,0) is the **top-left** corner; +x goes right, +y
goes down.

## Classes

Segment the image into these classes. Each has a fixed RGB color that the renderer will
use to paint the mask. Do **not** invent new classes; if something doesn't fit, use
`background`.

| id | name           | color (RGB)       | description                                                                |
|----|----------------|-------------------|----------------------------------------------------------------------------|
| 1  | track          | (40, 40, 40)      | The drivable karting circuit asphalt only — NOT the parking lot or roads.  |
| 2  | runoff         | (110, 110, 110)   | Paved areas adjacent to the track that aren't the racing line (apron, pit lane). |
| 3  | parking_road   | (160, 160, 160)   | Asphalt that is parking lot, access road, or public road — not the circuit. |
| 4  | curb           | (220, 40, 40)     | Red/white painted kerbs lining the racing line.                            |
| 5  | barrier        | (255, 120, 0)     | Tire stacks, plastic barriers, or red/white striped crash barriers.        |
| 6  | grass          | (60, 160, 70)     | Open grass / lawn / infield grass.                                         |
| 7  | tree_canopy    | (20, 80, 30)      | Tree foliage (canopy area as seen from above).                             |
| 8  | building       | (200, 170, 120)   | Buildings, sheds, container structures, roofed pit boxes.                  |
| 9  | vehicle        | (80, 60, 200)     | Cars, trucks, karts visible parked or on track.                            |
| 10 | rail           | (180, 180, 180)   | Train tracks / railway visible at the edge of the image.                   |
| 0  | background     | (0, 0, 0)         | Anything not covered above (paths, dirt, water, unknown).                  |

## Output

Return **only** valid JSON, no prose, no Markdown fences. The top-level object MUST match
this schema exactly:

```json
{
  "image": { "width": 800, "height": 449 },
  "regions": [
    {
      "class": "track",
      "polygon": [[x1,y1], [x2,y2], ...],
      "holes": [ [[hx1,hy1], ...], ... ]
    }
  ],
  "objects": [
    {
      "class": "tree_canopy",
      "id": "tree_001",
      "center": [x, y],
      "radius_px": r,
      "confidence": 0.0
    },
    {
      "class": "building",
      "id": "bld_001",
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": 0.0
    },
    {
      "class": "vehicle",
      "id": "veh_001",
      "bbox": [x_min, y_min, x_max, y_max],
      "heading_deg": 0,
      "confidence": 0.0
    }
  ],
  "notes": "free-form short string, may be empty"
}
```

### Field rules

- `regions` — area classes (`track`, `runoff`, `parking_road`, `grass`, `rail`,
  `background`, plus the area form of `curb` if it's a continuous strip). Each region is
  one closed polygon with optional inner `holes` (e.g., the grass infield inside the
  track). Polygons must have **≥ 4** vertices, listed counter-clockwise, with vertices
  no more than ~15 px apart along curves.
- `objects` — discrete instances:
  - `tree_canopy`: one entry per visible tree, `center` + `radius_px` (radius in pixels of
    the canopy as seen from above).
  - `building`: axis-aligned `bbox`.
  - `vehicle`: tight `bbox` around the vehicle plus `heading_deg` (0 = vehicle nose
    pointing image-right / +x, 90 = down / +y, measured clockwise; null if unsure).
  - `barrier`: short polyline as `polygon` with 2+ points.
- `confidence` is a float 0..1 estimating how sure you are about the instance / region.
- All numbers are integers except `confidence` and `heading_deg`.
- Use stable `id`s prefixed by class (`tree_001`, `tree_002`, …).

### Quality rules

1. **Be conservative on `track`**: only include the karting circuit itself. The parking
   lot at the top of the image and the road on the right side are `parking_road`, not
   `track`, even though they're paved.
2. The track infield contains grass islands — represent these as `holes` inside the
   `track` region polygon, not as separate grass regions if they are fully enclosed.
3. Trees: include each individual canopy you can distinguish, even if they overlap
   slightly. If two canopies have merged into one blob you can't separate, output one
   `tree_canopy` with the larger radius.
4. Shadows belong to whatever surface they fall on (don't classify shadow as tree).
5. Do not emit empty regions or zero-area objects.
6. Do not output anything outside the JSON object — no commentary, no Markdown, no
   trailing text. The first character of your reply must be `{` and the last `}`.
