# Dev Guide for ML_Intro

## Scope & Repo Hygiene
- Versioned areas: `README.md`, `01_notebook/`, `02_daten/`, `03_skript/`, `04_model/`.
- Ignore per .gitignore: `_misc/`, `_db/`, `.ipynb_checkpoints/`, `.jupyter/`, `.virtual_documents/`, `03_skript/Transformer`, `*.pptx`, `*.png`, `*.jpeg`, `X*.ipynb`, `_*.ipynb`, `03_skript/Vita.pdf` (except `02_daten/02_bild/*.png|*.jpg`).
- Legacy/reference: `90_repo/*` not part of normal edits.

## Workflow
- Sandbox/approvals: verify current sandbox mode before writes; avoid destructive git commands; never revert user changes.
- Naming: lowercase + underscores; avoid non-ASCII unless file already contains it.
- Use helpers in dry-run first: `python rename_files.py`, `python update_notebook_paths.py`, `python fix_subdirectories.py`.
- Notebooks: prefer consistent paths to datasets under `02_daten/` with correct subfolders; keep outputs minimal when committing.

## Environments
- Target Python 3.11+; common libs: scikit-learn, pandas, numpy, matplotlib/plotly, tensorflow/keras, xgboost, langchain, gradio.
- Set seeds for reproducibility where practical.

## Contribution tips
- Add brief comments only for non-obvious code.
- Respect data licensing; note external sources in markdown.
- When unsure, ask before touching non-versioned areas.
