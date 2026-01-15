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
| [Missing Values](prepare/missing_values.md) | Fehlende Werte erkennen und behandeln (SimpleImputer, KNNImputer) |
| [Outlier](prepare/outlier.md) | Ausreißer erkennen und behandeln (Z-Score, IQR, Isolation Forest) |

---

## Feature Engineering

Merkmalserstellung und -transformation zur Verbesserung der Modellperformance.

| Thema | Beschreibung |
|-------|--------------|
| [Feature Engineering](prepare/feature-engineering.md) | Feature Creation, Selection, Extraction und Domain Knowledge Integration |

---

## Data Transformation

Skalierung, Normalisierung und Encoding für ML-Algorithmen.

| Thema | Beschreibung |
|-------|--------------|
| [Skalierung](prepare/skalierung.md) | Normalisierung und Standardisierung (StandardScaler, MinMaxScaler) |
| [Kodierung](prepare/kodierung_kategorialer_daten.md) | Kategoriale Daten kodieren (OrdinalEncoder, OneHotEncoder, TargetEncoder) |

---

## Train-Test Split

Aufteilung in Trainings- und Testdaten für zuverlässige Modellbewertung.

| Thema | Beschreibung |
|-------|--------------|
| [Train-Test-Split](prepare/train_test_split.md) | Datenaufteilung, Stratifizierung, Data Leakage vermeiden |


---

**Version:** 1.0    
**Stand:** Januar 2026    
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.    
