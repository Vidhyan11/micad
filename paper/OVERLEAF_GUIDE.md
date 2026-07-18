# Overleaf — step-by-step guide to compile the paper

Goal: get our `main.tex` (Springer `svproc` class) compiling on Overleaf into the paper PDF.

---

## ⚠️ IMPORTANT: upload `svproc.cls`
The common Springer Overleaf template is the **LNCS** one (main file `samplepaper.tex`,
ships `llncs.cls`) — it does **NOT** contain `svproc.cls`, which our paper (and MICAD's
LaTeX package) uses. If you see **`File 'svproc.cls' not found`**, upload these three
files from your local **`latex template\styles\`** folder into the Overleaf project root:
**`svproc.cls`**, **`aliascnt.sty`**, **`remreset.sty`**. Then Recompile.

## Option A — Use MICAD's Overleaf template (recommended)
The template may already contain `svproc.cls`; if not, upload it as noted above.

### 1. Account
- Go to **overleaf.com** → **Register** (free) or **Log in**.

### 2. Open the MICAD template
- On the MICAD website, click **Download the Template@Overleaf**.
- It opens Overleaf showing the Springer Proceedings template.
- Click the green **Open as Template** button (top). Overleaf creates an editable copy in your
  account and opens it in the editor.

### 3. Find the files (left "Files" panel)
- You'll see a main `.tex` file (e.g. `author.tex` / `main.tex` / `samplepaper.tex`), plus
  `svproc.cls` and maybe style/`.bst`/sample-figure files. **Leave `svproc.cls` alone.**

### 4. Put OUR paper in
- Open our paper on GitHub: **github.com/Vidhyan11/micad → `paper/main.tex` → click "Raw"**.
- Select all (**Ctrl+A**) and copy (**Ctrl+C**).
- In Overleaf, click the template's main `.tex` file to open it, select all its text
  (**Ctrl+A**), delete, and **paste ours** (**Ctrl+V**).
- (Optional) rename that file to `main.tex`: right-click it → Rename.

### 5. Tell Overleaf which file is "main"
- **Menu** (top-left) → **Main document** → choose the file you pasted into.
  (Overleaf usually auto-detects the one with `\documentclass`.)

### 6. Add the figure
- Download `qualitative_melanoma.png` from Kaggle (see "Getting the figure" below).
- In Overleaf, click the **Upload** icon (up-arrow, top-left) → upload the PNG.
- It must sit in the **project root** (same level as `main.tex`) — our file references it as
  `qualitative_melanoma.png` with no folder.

### 7. Compile
- Click the green **Recompile** button. The PDF appears on the right.

### 8. Check
- Scroll the PDF; confirm tables, the figure, and the bibliography render.
- Check the **page count** — MICAD limit is **≤10 pages**. If over, we trim.

---

## Option B — Blank project + upload (if you don't use the template link)
1. Overleaf → **New Project → Blank Project**.
2. **Upload** these files (Upload icon):
   - our `paper/main.tex`
   - `qualitative_melanoma.png`
   - from your local `latex template/styles/` folder: **`svproc.cls`**, **`aliascnt.sty`**,
     **`remreset.sty`** (svproc needs these two `.sty` files).
3. Menu → Main document → `main.tex`. **Recompile.**

*(You do NOT need the `.bst` files — we use a manual bibliography, so there's no BibTeX step.)*

---

## Getting the figure from Kaggle
In your Kaggle notebook (after running `make_report.py`):
```python
from IPython.display import FileLink
FileLink('/kaggle/working/artifacts/figures/qualitative_melanoma.png')
```
Click the printed link to download. (Or use the notebook's **Output** tab → `artifacts/figures/`.)

---

## Common errors & fixes
| Message | Fix |
|---|---|
| `File 'qualitative_melanoma.png' not found` | PNG not uploaded, in a subfolder, or name mismatch. Put it in the project root, exact filename. **Or** temporarily comment the figure: put `%` before each line of the `\begin{figure}…\end{figure}` block to compile without it. |
| `File 'svproc.cls' not found` | You're in a blank project without the class — use Option A, or upload `svproc.cls` (+ `aliascnt.sty`, `remreset.sty`). |
| `Undefined control sequence \keywords` | You're not on the `svproc` class. Confirm the first line is `\documentclass{svproc}` and the class file is present. |
| Compile timeout / stuck | Menu → Recompile from scratch; or switch compiler to pdfLaTeX (Menu → Settings). |

---

## Keeping in sync with updates
When I push a new `main.tex` (e.g. after the citation-verification pass), just re-open
`paper/main.tex` on GitHub → **Raw** → copy → paste over the Overleaf file → Recompile.
Small edits you can also make directly in Overleaf.
