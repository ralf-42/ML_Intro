---
layout: default
title: Prepare
parent: Konzepte
nav_order: 2
has_children: true
description: "Datenaufbereitung für Machine Learning"
---

# Prepare

## Zuerst lesen

Beginne mit **Train-Test-Split** und danach mit **Prepare nach Modell**. Erst wenn klar ist, welches Modell welche Vorverarbeitung braucht, werden Imputation, Kodierung, Skalierung oder Ausreißerbehandlung gezielt eingesetzt.

## Lesepfad

1. [Train-Test-Split](https://ralf-42.github.io/ML_Intro/concepts/prepare/train_test_split.html)
2. [Prepare nach Modell](https://ralf-42.github.io/ML_Intro/concepts/prepare/prepare_nach_modell.html)
3. [Missing Values](https://ralf-42.github.io/ML_Intro/concepts/prepare/missing_values.html)
4. [Kodierung](https://ralf-42.github.io/ML_Intro/concepts/prepare/kodierung_kategorialer_daten.html)
5. [Skalierung](https://ralf-42.github.io/ML_Intro/concepts/prepare/skalierung.html)
6. [Outlier](https://ralf-42.github.io/ML_Intro/concepts/prepare/outlier.html)
7. [Feature Engineering](https://ralf-42.github.io/ML_Intro/concepts/prepare/feature-engineering.html)

## Train-Test Split

**[Train-Test-Split](https://ralf-42.github.io/ML_Intro/concepts/prepare/train_test_split.html)** – *Wie wird die Datentrennung zuverlässig und leckagefrei umgesetzt?* Stratifizierung, Datenaufteilung und Vermeidung von Data Leakage.

## Modellbezogene Entscheidung

**[Prepare nach Modell](https://ralf-42.github.io/ML_Intro/concepts/prepare/prepare_nach_modell.html)** – *Welche Vorverarbeitung braucht welcher Kursalgorithmus?* Fehlende Werte, Skalierung, Kodierung, Ausreißer und Besonderheiten für die im Kurs eingesetzten Modelle.

## Data Cleaning

**[Missing Values](https://ralf-42.github.io/ML_Intro/concepts/prepare/missing_values.html)** – *Wie werden fehlende Werte erkannt und sinnvoll behandelt?* SimpleImputer, KNNImputer und typische Strategien im Vergleich.

**[Outlier](https://ralf-42.github.io/ML_Intro/concepts/prepare/outlier.html)** – *Wie werden Ausreißer identifiziert und bereinigt?* Z-Score, IQR und Isolation Forest als komplementäre Methoden.

## Data Transformation

**[Kodierung](https://ralf-42.github.io/ML_Intro/concepts/prepare/kodierung_kategorialer_daten.html)** – *Wie werden kategoriale Daten modellierbar gemacht?* OrdinalEncoder, OneHotEncoder und TargetEncoder im Vergleich.

**[Skalierung](https://ralf-42.github.io/ML_Intro/concepts/prepare/skalierung.html)** – *Wann und wie werden Merkmale skaliert oder normalisiert?* StandardScaler, MinMaxScaler und die Auswirkungen auf distanzbasierte Algorithmen.

## Feature Engineering

**[Feature Engineering](https://ralf-42.github.io/ML_Intro/concepts/prepare/feature-engineering.html)** – *Wie werden aus Rohdaten informative Merkmale gewonnen?* Feature Creation, Selection, Extraction und Domain Knowledge Integration.
