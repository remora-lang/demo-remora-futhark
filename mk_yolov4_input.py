#!/usr/bin/env python3
"""Build the Futhark benchmark data files for the YOLOv4 pipeline.

Writes yolov4_input.bin (the argument tuple for entry point "main") and
yolov4_expected.bin (yolov4_out.bin with a Futhark binary-format header).

The parameter *order* and rank come from the entry point declaration in
yolov4.fut_soacs; the concrete shapes come from the entry point in
yolov4.remora. This mirrors what run_yolo.rkt does.

"""

import re
import struct
import sys

IR = "yolov4.fut_soacs"
REMORA = "yolov4.remora"
IMAGE = "input.bin"
WEIGHTS = "yolov4.weights"
WEIGHTS_HEADER = 20  # darknet's major/minor/revision/seen preamble
OUT = "yolov4_input.bin"
REFERENCE = "yolov4_out.bin"
EXPECTED = "yolov4_expected.bin"


def entry_params():
    """[(name, rank)] in the order the entry point takes them."""
    ir = open(IR).read()
    m = re.search(r'entry\("main",\s*\{(.*?)\},\n', ir, re.S)
    if not m:
        sys.exit("no entry point 'main' found in " + IR)
    params = []
    for item in m.group(1).split(","):
        name, ty = (s.strip() for s in item.strip().split(":"))
        assert ty.endswith("f32"), ty
        params.append((name, ty.count("[]")))
    return params


def entry_shapes():
    """{name: shape} for every [Float d ...] binding in the Remora entry point."""
    rem = open(REMORA).read()
    entry = rem[rem.index("(entry (main"):]
    shapes = {}
    for name, dims in re.findall(r"\(([A-Za-z0-9-]+)\s+\[Float((?:\s+\d+)+)\]\)", entry):
        shapes[name] = tuple(int(d) for d in dims.split())
    return shapes


def header(shape):
    return b"b" + bytes([2, len(shape)]) + b" f32" + b"".join(
        struct.pack("<Q", d) for d in shape
    )


def nbytes(shape):
    n = 4
    for d in shape:
        n *= d
    return n


def main():
    params = entry_params()
    shapes = entry_shapes()
    for name, rank in params:
        if name not in shapes:
            sys.exit("no shape for parameter " + name)
        if len(shapes[name]) != rank:
            sys.exit("rank mismatch for parameter " + name)

    image = open(IMAGE, "rb").read()
    if len(image) != nbytes(shapes["image"]):
        sys.exit("%s is %d bytes, expected %d"
                 % (IMAGE, len(image), nbytes(shapes["image"])))

    weights = open(WEIGHTS, "rb").read()[WEIGHTS_HEADER:]

    with open(OUT, "wb") as out:
        offset = 0
        for name, _ in params:
            shape = shapes[name]
            n = nbytes(shape)
            if name == "image":
                chunk = image
            else:
                chunk = weights[offset:offset + n]
                offset += n
            if len(chunk) != n:
                sys.exit("ran out of weights at parameter " + name)
            out.write(header(shape))
            out.write(chunk)
    if offset != len(weights):
        sys.exit("%d unused bytes left in %s" % (len(weights) - offset, WEIGHTS))

    dets = open(REFERENCE, "rb").read()
    shape = (len(dets) // 4 // 85, 85)
    with open(EXPECTED, "wb") as out:
        out.write(header(shape))
        out.write(dets)

    print("wrote %s (%d parameters) and %s (%dx%d)"
          % (OUT, len(params), EXPECTED, shape[0], shape[1]))


if __name__ == "__main__":
    main()
