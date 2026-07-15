# Notebooks

Thin Kaggle drivers. Each imports `fbc`, does one job, writes one artifact to
`/kaggle/working/artifacts`, and is chained forward via "Add data → Notebook output".

| Notebook | Milestone | Produces |
|---|---|---|
| `01_extract_embeddings.ipynb` | ME | cached frozen-encoder embeddings per dataset×encoder |
| `02_pseudolabels.ipynb` | MP | foundation zero-shot concept pseudo-labels + GT-agreement report |
| `03_train_cbm.ipynb` | MM | trained concept/diagnosis heads + Experiment 1 table |
| `04_faithfulness.ipynb` | MF1 | faithfulness scores + Experiment 2 table + per-case records |
| `05_fairness.ipynb` | MF3 | per-Fitzpatrick audit + mitigation + Experiment 3 table |
| `06_tables_figures.ipynb` | MR | all 5 tables (CSV/LaTeX) + qualitative figure |

Notebook cells are authored once the corresponding package modules exist.
Setup preamble for each notebook:

```python
import sys; sys.path.insert(0, "/kaggle/working/micad/src")   # or pip install -e
import fbc, fbc.config as C
C.ensure_dirs()
```
