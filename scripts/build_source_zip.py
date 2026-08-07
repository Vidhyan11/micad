"""Assemble the camera-ready bundle: final PDF + compilable LaTeX source, zipped.

Creates D:/micad/submission/camera_ready/ with:
  Faithful-by-Concept.pdf            (the final manuscript PDF)
  main.tex                           (paper source)
  svproc.cls, aliascnt.sty, remreset.sty  (Springer proceedings class + deps)
  faithfulness_bars.png              (Fig 2, name matches \includegraphics)
  qualitative_melanoma.png           (Fig 3, name matches \includegraphics)
then zips it to D:/micad/submission/Faithful-by-Concept_camera_ready.zip.
(Fig 1 is TikZ inside main.tex, so no image file is needed to compile.)
"""
from __future__ import annotations

import os
import re
import shutil
import zipfile

ROOT = r"D:\micad"
SUB = os.path.join(ROOT, "submission")
DEST = os.path.join(SUB, "camera_ready")
STYLES = os.path.join(ROOT, "latex template", "styles")

PDF_SRC = os.path.join(ROOT,
    "Faithful_by_Concept__Verifiable_and_Equitable_Concept_Reasoning_for_Skin_Lesion_Diagnosis.pdf")

# (source path, destination name)
COPIES = [
    (PDF_SRC, "Faithful-by-Concept.pdf"),
    (os.path.join(ROOT, "paper", "main.tex"), "main.tex"),
    (os.path.join(STYLES, "svproc.cls"), "svproc.cls"),
    (os.path.join(STYLES, "aliascnt.sty"), "aliascnt.sty"),
    (os.path.join(STYLES, "remreset.sty"), "remreset.sty"),
    (os.path.join(SUB, "Fig 2.png"), "faithfulness_bars.png"),
    (os.path.join(SUB, "Fig 3.png"), "qualitative_melanoma.png"),
]


def main():
    if os.path.isdir(DEST):
        shutil.rmtree(DEST)
    os.makedirs(DEST)

    for src, name in COPIES:
        if not os.path.exists(src):
            raise SystemExit(f"MISSING: {src}")
        shutil.copy2(src, os.path.join(DEST, name))

    # sanity: every \includegraphics target exists in the bundle
    tex = open(os.path.join(DEST, "main.tex"), encoding="utf-8", errors="ignore").read()
    imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    for img in imgs:
        if not os.path.exists(os.path.join(DEST, img)):
            print(f"  WARNING: main.tex references '{img}' but it's not in the bundle")

    zpath = os.path.join(SUB, "Faithful-by-Concept_camera_ready.zip")
    if os.path.exists(zpath):
        os.remove(zpath)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(DEST)):
            z.write(os.path.join(DEST, name), arcname=os.path.join("camera_ready", name))

    print("Bundle folder:", DEST)
    print("includegraphics targets in main.tex:", imgs or "(none)")
    print("\nzip contents:")
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            print(f"  {info.filename}  ({info.file_size} bytes)")
    print("\n->", zpath)


if __name__ == "__main__":
    main()
