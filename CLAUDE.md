# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a German-language Machine Learning course repository (`ML_Intro`) containing educational materials from basic ML concepts to advanced deep learning applications. The course covers supervised/unsupervised learning, neural networks, ensemble methods, and specialized ML applications.

## Directory Structure

```
ML_Intro/
├── 01_notebook/     # Course notebooks (10 modules: 00_general through 09_diverse)
├── 02_daten/        # Datasets (organized by type: text, images, audio, video, tables)
├── 03_skript/       # Course presentations and documentation (PDFs)
├── 04_model/        # Trained models (e.g., diamonds_model.pmml)
├── 90_repo/         # Archived reference materials (read-only, not versioned)
└── *_Simulator/     # Interactive educational tools (Perzeptron, Entscheidungsbaum)
```

### Course Modules (`01_notebook/`)

The course is structured in 10 numbered modules, each in its own subdirectory:

- **00_general/** - ML basics, pandas fundamentals, dataset exploration
- **01_supervised/** - Decision trees (Titanic), linear regression (MPG), random forests (Diamonds)
- **02_unsupervised/** - K-means, DBSCAN, PCA, Apriori, isolation forest
- **03_network/** - MLPs and Keras neural networks (Cancer, Diamonds datasets)
- **04_ensemble/** - XGBoost, stacking methods
- **05_tuning/** - Cross-validation, grid/random search, ROC-AUC, AutoML with PyCaret
- **06_workflow/** - Scikit-learn pipelines
- **07_special/** - Computer vision (MNIST, YOLO), NLP (spam detection), time series, autoencoders
- **08_genai/** - LangChain integration, LLM applications (currently empty in main course)
- **09_diverse/** - XAI (explainable AI), Gradio apps, model persistence, Gemini AI

**Notebook naming**: `b###_topic.ipynb` (e.g., `b110_sl_dt_titanic.ipynb`)

### Dataset Organization (`02_daten/`)

```
02_daten/
├── 01_text/          # Text data (e.g., smsspamcollection for NLP)
├── 02_bild/          # Images for CV tasks (*.png, *.jpg)
├── 03_audio/         # Audio files
├── 04_video/         # Video files (e.g., pexels_pixabay_people.mp4)
└── 05_tabellen/      # Tabular datasets (CSV/XLSX)
```

**Key datasets** in `05_tabellen/`:
- `titanic.xlsx` - Survival prediction (classification)
- `diamonds.csv` - Price prediction (regression)
- `breast_cancer_wisconsin.csv` - Medical diagnosis
- `auto_mpg.csv` - Fuel efficiency prediction
- `ccpp.csv` - Combined Cycle Power Plant
- `wa_fn_usec__telco_customer_churn.csv` - Customer churn

## Notebook Structure

All course notebooks follow a consistent 5-phase structure:

```python
# 0 | Install & Import
# 1 | Understand      # Task, EDA, visualization
# 2 | Prepare         # Cleaning, encoding, train-test split
# 3 | Modeling        # Model selection, training, hyperparameter tuning
# 4 | Evaluate        # Predictions, metrics, confusion matrix, feature importance
# 5 | Deploy          # Model export, documentation
```

Each phase has a checklist (📋 Checkliste) in markdown cells.

## Common Patterns

### Data Loading

Notebooks typically load data from GitHub URLs:

```python
from pandas import read_excel, read_csv

df = read_excel(
    "https://raw.githubusercontent.com/ralf-42/ML_Intro/main/02_daten/05_tabellen/titanic.xlsx",
    usecols=["pclass", "survived", "sex", "age", "sibsp", "parch"]
)
```

### Standard Workflow

```python
# Train-test split with stratification
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.20, random_state=42, stratify=target
)

# Model training
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(data_train, target_train)

# Evaluation
target_test_pred = model.predict(data_test)
accuracy_score(target_test, target_test_pred)
```

### Visualization

Uses `plotly.express` and `matplotlib`:

```python
import plotly.express as px

# Feature importance visualization
px.bar(x=model.feature_importances_, y=data.columns, title="Feature Importance")
```

## Naming Conventions

**CRITICAL**: All files and directories must follow strict naming rules:

- **Lowercase only** (`titanic.csv`, not `Titanic.csv`)
- **Underscores instead of hyphens** (`auto_mpg.csv`, not `auto-mpg.csv`)
- **No spaces** (`breast_cancer_wisconsin.csv`, not `breast cancer wisconsin.csv`)

### Maintenance Scripts

Three Python scripts enforce naming consistency:

1. **`rename_files.py`** - Converts filenames to lowercase and replaces `-` with `_`
   ```bash
   python rename_files.py          # Preview
   python rename_files.py --execute  # Apply changes
   ```

2. **`update_notebook_paths.py`** - Updates file references in notebooks
   ```bash
   python update_notebook_paths.py --analyze   # Show file references
   python update_notebook_paths.py --execute   # Update paths
   ```

3. **`fix_subdirectories.py`** - Fixes directory structure inconsistencies

**Always run these in dry-run mode first before executing.**

## Versioning & Git Ignore

### Versioned Areas
- `01_notebook/`, `02_daten/`, `03_skript/`, `04_model/`
- `README.md`, `DEV_GUIDE.md`, `DO_NOT_EDIT.md`

### Ignored Paths (per `.gitignore`)
- **Subdirectories**: `_misc/`, `_db/`, `.ipynb_checkpoints/`, `.jupyter/`, `.virtual_documents/`
- **Files**: `*.pptx`, `*.png`, `*.jpeg` (except `02_daten/02_bild/*.png|*.jpg`)
- **Notebooks**: `X*.ipynb`, `_*.ipynb` (experimental/temp)
- **Specific**: `03_skript/Transformer/`, `03_skript/Vita.pdf`
- **Archives**: `90_repo/*` (read-only reference materials)

**IMPORTANT**: Never edit files in `90_repo/`, `_misc/`, or checkpoint directories unless explicitly requested.

## Technology Stack

### Core Libraries
- **ML Framework**: scikit-learn (pipelines, models, metrics)
- **Data Handling**: pandas, numpy
- **Deep Learning**: Keras, TensorFlow
- **Visualization**: plotly (express, subplots), matplotlib
- **Advanced ML**: XGBoost, PyCaret (AutoML)
- **Tree Visualization**: dtreeviz, graphviz
- **Web Apps**: Gradio
- **Gen AI**: LangChain, OpenAI API (module 08)

### Environment
- **Python**: 3.11+
- **IDEs**: Google Colab, Jupyter Notebook/Lab
- **Package manager**: pip (some notebooks use `uv pip`)

### Common Installations

Notebooks often include inline installations:

```python
!uv pip install git+https://github.com/parrt/dtreeviz.git
```

## Development Workflow

### Working with Notebooks

1. **Read before editing**: Always use the Read tool on notebooks before suggesting modifications
2. **Large files**: Use `NotebookEdit` for cell-level changes to avoid context limits
3. **Language**: All educational content, comments, and variables are in **German**
4. **Outputs**: Keep cell outputs minimal when committing
5. **Emojis**: Headers use emojis (🎯, 📚, ⚠️, 🔢, ✂️, 🏃, 🔭, etc.) - maintain existing conventions

### Reproducibility

Set random seeds consistently:

```python
random_state=42  # Standard seed across course
```

### Error Handling

Notebooks suppress warnings for cleaner output:

```python
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
```

## Common Commands

### Jupyter
```bash
jupyter lab
jupyter notebook
```

### Environment Setup (Windows)
```bash
set OPENAI_API_KEY=your-key-here
```

### Package Installation
```bash
pip install scikit-learn pandas numpy matplotlib plotly tensorflow keras xgboost
pip install gradio langchain langchain-openai  # For GenAI module
pip install dtreeviz graphviz pycaret  # Optional tools
```

## Educational Context

- **Audience**: German-speaking ML students/practitioners
- **Focus**: Hands-on practical examples with real datasets
- **Philosophy**: Clear structure (5-phase workflow), reproducible results, visual explanations
- **Assessment**: Each module includes checklists to track learning objectives

## Special Considerations

1. **Module 08 (GenAI)**: Currently empty in main course notebooks but mentioned in README. If creating GenAI content, follow LangChain 1.0 patterns from parent `CLAUDE.md`.

2. **Simulators**: Interactive tools in `30_Simulator_Perzeptron/` and `30_enter_Entscheidungsbaum_Ersteller/` are standalone Python GUI applications, not Jupyter notebooks.

3. **Modified Files**: Current git status shows modifications to:
   - `01_notebook/01_supervised/b110_sl_dt_titanic.ipynb`
   - `01_notebook/09_diverse/b900_xai_titanic_fehlerhaft.ipynb` (Note: filename includes "fehlerhaft" = faulty/erroneous)

4. **External Datasets**: Many notebooks reference datasets via GitHub URLs. Local paths should use relative references: `../../02_daten/05_tabellen/dataset.csv`

## Quick Reference

| Task | Tool/Pattern |
|------|--------------|
| Load CSV | `pandas.read_csv('https://raw.githubusercontent.com/ralf-42/ML_Intro/main/02_daten/...')` |
| Train-test split | `train_test_split(..., test_size=0.20, random_state=42, stratify=target)` |
| Visualization | `plotly.express` (preferred) or `matplotlib` |
| Decision trees | `dtreeviz` for visual explanations |
| Model saving | PMML format (e.g., `diamonds_model.pmml` in `04_model/`) |
| Web apps | Gradio for interactive demos |
