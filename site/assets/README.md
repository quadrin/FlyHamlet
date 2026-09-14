# Fly artwork

The live view (`site/flight-view.js`) draws three sprites. The body is always
needed. The shadow and the wings are optional: if a file is missing or does not
load, the renderer draws the old procedural shape instead.

## `fruit-fly.png` — body

Transparent PNG, square, head up. The replay and the live view both use it.
Give the fly tucked wings: the live view animates the wings from
`fly-wings.png` and draws them behind the body.

The legs come from this image. The live view does not draw legs.

Art direction: amber striped abdomen, red compound eyes, six fine legs,
head pointing upward, isolated background.

The current file is an original AI-generated image made with OpenAI image
generation for this interface on 2026-09-13. It is a 3D-style macro
illustration of Drosophila melanogaster, not a specimen photograph.

## `fly-shadow.png` — ground shadow

Transparent PNG, square, a soft dark blob that fades to nothing at the edge.
The renderer scales it to the shadow radius, turns it with the heading and
flattens it. Altitude sets the opacity, so keep the artwork itself neutral.
Tune the strength with `SHADOW_SPRITE_GAIN` in `site/flight-view.js`.

## `fly-wings.png` — wingbeat sheet

Transparent PNG. A 4 x 3 grid of equal cells, so 12 cells in total. Every cell
holds **both** wings, head up, with the hinges at the centre of the cell. The
12 cells are one complete wingbeat, read left to right along row 1, then row 2,
then row 3. Cell 0, at the top left, is the top of the upstroke.

The renderer clips each cell to one half, so the left and right wing can hold
different stroke phases while the fly turns. Keep the hinges on the cell centre
line, or the two halves will not meet.

Geometry lives in `site/flight-view.js`: `WING_SHEET_COLS`, `WING_SHEET_ROWS`,
`WING_SHEET_SPAN` and `WING_SHEET_ANCHOR_Y`.
