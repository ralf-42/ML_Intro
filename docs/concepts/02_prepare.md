---
layout: default
title: Prepare
parent: Konzepte
nav_order: 2
has_children: true
description: "Datenaufbereitung für Machine Learning"
---

# Prepare

Systematische Datenaufbereitung und Preprocessing für Machine Learning Modelle.

Die wichtigsten Aspekte der Datenaufbereitung:

- **Data Cleaning** - Bereinigung und Qualitätssicherung
- **Feature Engineering** - Merkmalserstellung und -transformation
- **Data Transformation** - Skalierung, Normalisierung, Encoding
- **Train-Test Split** - Aufteilung in Trainings- und Testdaten

---

## Data Cleaning

Bereinigung und Qualitätssicherung der Rohdaten.

| Thema | Beschreibung |
|-------|--------------|
| [Missing Values](prepare/missing_values.html) | Fehlende Werte erkennen und behandeln (SimpleImputer, KNNImputer) |
| [Outlier](prepare/outlier.html) | Ausreißer erkennen und behandeln (Z-Score, IQR, Isolation Forest) |

---

## Feature Engineering

Merkmalserstellung und -transformation zur Verbesserung der Modellperformance.

| Thema | Beschreibung |
|-------|--------------|
| [Feature Engineering](prepare/feature-engineering.html) | Feature Creation, Selection, Extraction und Domain Knowledge Integration |

---

## Data Transformation

Skalierung, Normalisierung und Encoding für ML-Algorithmen.

| Thema | Beschreibung |
|-------|--------------|
| [Skalierung](prepare/skalierung.html) | Normalisierung und Standardisierung (StandardScaler, MinMaxScaler) |
| [Kodierung](prepare/kodierung_kategorialer_daten.html) | Kategoriale Daten kodieren (OrdinalEncoder, OneHotEncoder, TargetEncoder) |

---

## Train-Test Split

Aufteilung in Trainings- und Testdaten für zuverlässige Modellbewertung.

| Thema | Beschreibung |
|-------|--------------|
| [Train-Test-Split](prepare/train_test_split.html) | Datenaufteilung, Stratifizierung, Data Leakage vermeiden |

## Abgrenzung zu verwandten Dokumenten

| Dokument | Frage |
|---|---|
| [Grundlagen](./01_grundlagen.html) | Welche ML-Logik und welchen Workflow setzt Datenvorbereitung voraus? |
| [Modeling](./03_modeling.html) | Welche Modelle profitieren wovon in der Vorverarbeitung? |
| [Evaluate](./04_evaluate.html) | Wie wird geprüft, ob die vorbereiteten Daten zu belastbaren Ergebnissen führen? |
| [XAI](./08_xai.html) | Wie wirkt sich Vorverarbeitung auf spätere Interpretierbarkeit aus? |

