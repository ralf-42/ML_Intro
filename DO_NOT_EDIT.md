# Do Not Edit List

The following paths/files should be considered read-only unless explicitly requested:
- `90_repo/*` (archival/reference notebooks and materials)
- `_misc/*`, `_db/*`, `.ipynb_checkpoints/*`, `.jupyter/*`, `.virtual_documents/*`
- `/03_skript/Transformer/`
- Presentation/image binaries ignored by .gitignore (e.g., `*.pptx`, `*.png`, `*.jpeg`, `03_skript/Vita.pdf`)

Rationale: avoid clobbering archives, checkpoints, or ignored assets; keep focus on versioned course materials (`01_notebook`, `02_daten`, `03_skript`, `04_model`).
