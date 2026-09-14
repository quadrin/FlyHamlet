# Fly artwork

The live view (`site/flight-view.js`) draws three sprites. The body is always
needed.

The body art owns the legs and the wings. The live view draws no legs. It
animates wings only when `fly-wings.png` loads. Thus the fly never shows two
sets of wings. But you must keep the two files in agreement: if you add
`fly-wings.png`, give the body image tucked wings, or small wings, or no wings.

If `fly-shadow.png` does not load, the view draws a gradient instead.

## `fruit-fly.png` — body

Transparent PNG, square, head up. The replay and the live view both use it.
Give the fly tucked wings: the live view animates the wings from
`fly-wings.png` and draws them behind the body.

The legs come from this image. The live view does not draw legs.

The wings also come from this image while `fly-wings.png` is absent. Add both
files together: an old body image with spread wings, plus an animated pair,
gives the fly four wings.

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

Transparent PNG. 12 wing pairs in 4 columns and 3 rows. Every cell holds
**both** wings, head up. The 12 cells are one complete wingbeat, read left to
right along row 1, then row 2, then row 3. Cell 0, at the top left, is the top
of the upstroke.

The cells are **not** on a uniform grid. In the current artwork the hinges sit
at x = 283, 818, 1354 and 1878, which a 4-column cut would put at 271.5, 814.5,
1357.5 and 1900.5. One pair also crosses a uniform row boundary. So the renderer
does not divide the sheet. It holds a table of 12 source rectangles, each with
its own hinge, in the `WING_CELLS` constant in `site/flight-view.js`.

**If you change this image, regenerate the table:**

    python3 scripts/measure_wing_sheet.py > /tmp/cells.js

Paste the result over `WING_CELLS`. The script finds each pair by its alpha
bounding box, then finds the hinge as the row where the two wings come closest
together. Check the `COLUMN_HINGES`, `COLUMN_BANDS` and `ROW_BANDS` constants at
the top of the script if the layout moves.

The renderer puts the hinge on the body's wing root and clips there, so the left
and right wing can hold different stroke phases while the fly turns.

Two other constants in `site/flight-view.js` set the fit: `WING_SHEET_SCALE`
(sheet pixel to body size; 0.0030 makes a wing about as long as the body) and
`WING_ANCHOR_Y` (the wing root on the thorax, -0.06 of the body size above
centre).
