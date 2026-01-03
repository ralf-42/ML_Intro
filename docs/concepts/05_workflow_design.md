---
layout: default
title: Workflow Design
parent: Konzepte
nav_order: 5
description: "ML Workflow Design und Best Practices"
---

# Workflow Design

Praktische Patterns und Best Practices für ML-Workflows.

## Scikit-learn Pipelines

Automatisierung von Preprocessing und Modeling:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
```

## Cross-Validation

- K-Fold Cross-Validation
- Stratified K-Fold
- Time Series Split

## Hyperparameter Tuning

- Grid Search
- Random Search
- Bayesian Optimization

## Model Evaluation

- Confusion Matrix
- ROC-AUC
- Precision-Recall Curves
- Feature Importance

## AutoML

- PyCaret
- Auto-sklearn
- H2O AutoML

_Detaillierte Inhalte folgen_


---

**Version:** 1.0
**Stand:** Januar 2026
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.
