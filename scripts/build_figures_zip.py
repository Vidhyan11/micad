"""Assemble the camera-ready 'separate figure files' zip.

Expects, in D:/micad/submission/:
  - 'Fig 1.jpg'                 (architecture; produced by make_fig1.py)
  - faithfulness_bars.png       (paper Fig 2; download from Kaggle and place here)
  - qualitative_melanoma.png    (paper Fig 3; download from Kaggle and place here)

Converts the two PNGs to 'Fig 2.jpg' / 'Fig 3.jpg' (high quality, flattened on white)
and zips Fig 1/2/3 into 'figures.zip'.
"""
from __future__ import annotations

import os
import zipfile

from PIL import Image

SUB = r"D:\micad\submission"
MAP = {  # source in submission/  ->  output name
    "faithfulness_bars.png": "Fig 2.jpg",
    "qualitative_melanoma.png": "Fig 3.jpg",
}


def png_to_jpg(src, dst):
    im = Image.open(src).convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[-1])
    bg.save(dst, "JPEG", quality=95, dpi=(300, 300))


def main():
    fig1 = os.path.join(SUB, "Fig 1.jpg")
    if not os.path.exists(fig1):
        raise SystemExit("Missing 'Fig 1.jpg' — run scripts/make_fig1.py first.")

    missing = [s for s in MAP if not os.path.exists(os.path.join(SUB, s))]
    if missing:
        raise SystemExit(
            "Place these files in D:/micad/submission/ first, then re-run:\n  "
            + "\n  ".join(missing)
            + "\n(Download them from Kaggle: /kaggle/working/artifacts/figures/)")

    outputs = ["Fig 1.jpg"]
    for src, dst in MAP.items():
        png_to_jpg(os.path.join(SUB, src), os.path.join(SUB, dst))
        outputs.append(dst)

    zpath = os.path.join(SUB, "figures.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in outputs:
            z.write(os.path.join(SUB, name), arcname=name)

    print("figures.zip contents:")
    for name in outputs:
        sz = os.path.getsize(os.path.join(SUB, name))
        w, h = Image.open(os.path.join(SUB, name)).size
        print(f"  {name}  ({w}x{h}, {sz} bytes)")
    print("->", zpath)


if __name__ == "__main__":
    main()
