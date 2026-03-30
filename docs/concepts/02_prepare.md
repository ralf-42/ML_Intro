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

**[Missing Values](https://ralf-42.github.io/ML_Intro/concepts/prepare/missing_values.html)** – *Wie werden fehlende Werte erkannt und sinnvoll behandelt?* SimpleImputer, KNNImputer und typische Strategien im Vergleich.

**[Outlier](https://ralf-42.github.io/ML_Intro/concepts/prepare/outlier.html)** – *Wie werden Ausreißer identifiziert und bereinigt?* Z-Score, IQR und Isolation Forest als komplementäre Methoden.

## Feature Engineering

Merkmalserstellung und -transformation zur Verbesserung der Modellperformance.

**[Feature Engineering](https://ralf-42.github.io/ML_Intro/concepts/prepare/feature-engineering.html)** – *Wie werden aus Rohdaten informative Merkmale gewonnen?* Feature Creation, Selection, Extraction und Domain Knowledge Integration.

## Data Transformation

Skalierung, Normalisierung und Encoding für ML-Algorithmen.

**[Skalierung](https://ralf-42.github.io/ML_Intro/concepts/prepare/skalierung.html)** – *Wann und wie werden Merkmale skaliert oder normalisiert?* StandardScaler, MinMaxScaler und die Auswirkungen auf distanzbasierte Algorithmen.

**[Kodierung](https://ralf-42.github.io/ML_Intro/concepts/prepare/kodierung_kategorialer_daten.html)** – *Wie werden kategoriale Daten modellierbar gemacht?* OrdinalEncoder, OneHotEncoder und TargetEncoder im Vergleich.

## Train-Test Split

Aufteilung in Trainings- und Testdaten für zuverlässige Modellbewertung.

**[Train-Test-Split](https://ralf-42.github.io/ML_Intro/concepts/prepare/train_test_split.html)** – *Wie wird die Datentrennung zuverlässig und leckagefrei umgesetzt?* Stratifizierung, Datenaufteilung und Vermeidung von Data Leakage.

