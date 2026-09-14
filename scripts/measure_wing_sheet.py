#!/usr/bin/env python3
"""Measure site/assets/fly-wings.png and print the WING_CELLS table.

The wing sheet holds 12 wing pairs in 4 columns and 3 rows, but the cells are
not on a uniform grid: the hinges sit at x = 283, 818, 1354 and 1878, and one
pair crosses a uniform row boundary. So the renderer does not divide the sheet.
It uses the table that this script prints.

For each cell the script finds the alpha bounding box, then the hinge: the row
where the two wings come closest together. Paste the output into the
WING_CELLS constant in site/flight-view.js.

Usage: python3 scripts/measure_wing_sheet.py [path/to/fly-wings.png]
"""
import struct, sys, zlib

THRESHOLD = 16
PAD = 3
# Column bands and row bands come from the gaps in the alpha profile. Re-check
# them with --profile if the artwork is re-exported.
COLUMN_HINGES = [283, 818, 1354, 1878]
COLUMN_BANDS = [(0, 540), (541, 1085), (1086, 1620), (1621, 2171)]
ROW_BANDS = [(0, 255), (256, 466), (467, 723)]


def load_rgba(path):
    data = open(path, 'rb').read()
    pos, idat, width, height, depth, colour = 8, b'', 0, 0, 0, 0
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if kind == b'IHDR':
            width, height, depth, colour = struct.unpack('>IIBB', body[:10])
        elif kind == b'IDAT':
            idat += body
        elif kind == b'IEND':
            break
        pos += 12 + length
    if colour != 6 or depth != 8:
        raise SystemExit(f'{path}: need 8-bit RGBA, got colour type {colour} depth {depth}')
    raw = zlib.decompress(idat)
    bpp, stride = 4, width * 4
    out, prev, pos = bytearray(height * stride), bytearray(stride), 0
    for y in range(height):
        filt = raw[pos]; pos += 1
        line = bytearray(raw[pos:pos + stride]); pos += stride
        if filt == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif filt == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif filt == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 255
        elif filt == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                c = prev[i - bpp] if i >= bpp else 0
                b = prev[i]
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return width, height, bytes(out[i * 4 + 3] for i in range(width * height))


def bounding_box(alpha, width, x0, x1, y0, y1):
    bx0, by0, bx1, by1 = width, y1, -1, -1
    for y in range(y0, y1 + 1):
        row = y * width
        for x in range(x0, x1 + 1):
            if alpha[row + x] > THRESHOLD:
                bx0, bx1 = min(bx0, x), max(bx1, x)
                by0, by1 = min(by0, y), max(by1, y)
    return None if bx1 < 0 else (bx0, by0, bx1, by1)


def hinge_row(alpha, width, box, hinge_x):
    """The wings meet at the root, so the hinge is the row where the gap between
    the left wing's right edge and the right wing's left edge is smallest."""
    x0, y0, x1, y1 = box
    best_gap, rows = 1 << 30, []
    for y in range(y0, y1 + 1):
        row = y * width
        left = next((x for x in range(hinge_x, x0 - 1, -1) if alpha[row + x] > THRESHOLD), None)
        right = next((x for x in range(hinge_x, x1 + 1) if alpha[row + x] > THRESHOLD), None)
        if left is None or right is None:
            continue
        gap = right - left
        if gap < best_gap:
            best_gap, rows = gap, [y]
        elif gap == best_gap:
            rows.append(y)
    return sum(rows) / len(rows), best_gap


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'site/assets/fly-wings.png'
    width, height, alpha = load_rgba(path)
    print(f'// {path}  {width} x {height}', file=sys.stderr)
    print('  const WING_CELLS = [')
    for row, (ry0, ry1) in enumerate(ROW_BANDS):
        for col, (cx0, cx1) in enumerate(COLUMN_BANDS):
            box = bounding_box(alpha, width, cx0, min(cx1, width - 1), ry0, min(ry1, height - 1))
            if box is None:
                raise SystemExit(f'cell {row * 4 + col} is empty; check the band constants')
            hinge_x = COLUMN_HINGES[col]
            hinge_y, gap = hinge_row(alpha, width, box, hinge_x)
            sx = max(0, box[0] - PAD)
            sy = max(0, box[1] - PAD)
            sw = min(width, box[2] + 1 + PAD) - sx
            sh = min(height, box[3] + 1 + PAD) - sy
            print(f'    [{sx}, {sy}, {sw}, {sh}, {hinge_x - sx}, {round(hinge_y - sy)}],'
                  f'  // {row * 4 + col:2}  gap {gap}px')
    print('  ];')


if __name__ == '__main__':
    main()
