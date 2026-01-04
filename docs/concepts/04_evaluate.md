---
layout: default
title: Evaluate
parent: Konzepte
nav_order: 4
has_children: true
description: "Bewertung und Evaluation von ML-Modellen"
---

# Evaluate

Methoden und Metriken zur Bewertung und Evaluation von Machine Learning Modellen.

Die wichtigsten Aspekte der Modellbewertung:

- **Klassifikationsmetriken** - Accuracy, Precision, Recall, F1-Score
- **Regressionsmetriken** - MSE, RMSE, MAE, R²
- **ROC & AUC** - Receiver Operating Characteristic und Area Under Curve
- **Confusion Matrix** - Visualisierung von Klassifikationsergebnissen
- **Cross-Validation** - K-Fold, Stratified K-Fold, Leave-One-Out
- **Model Selection** - Vergleich und Auswahl von Modellen

## Klassifikationsmetriken

- **Accuracy** - Anteil korrekter Vorhersagen
- **Precision** - Anteil korrekt positiver unter allen als positiv klassifizierten
- **Recall (Sensitivity)** - Anteil gefundener positiver Fälle
- **F1-Score** - Harmonisches Mittel aus Precision und Recall
- **Specificity** - Anteil korrekt negativer Fälle

## Regressionsmetriken

- **Mean Squared Error (MSE)** - Mittlerer quadratischer Fehler
- **Root Mean Squared Error (RMSE)** - Wurzel des MSE
- **Mean Absolute Error (MAE)** - Mittlerer absoluter Fehler
- **R² (Bestimmtheitsmaß)** - Anteil erklärter Varianz

## Confusion Matrix

Visualisierung der Klassifikationsergebnisse:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

## ROC & AUC

- **ROC-Kurve** - True Positive Rate vs. False Positive Rate
- **AUC** - Fläche unter der ROC-Kurve (0.5 - 1.0)
- **Threshold-Optimierung** - Auswahl des optimalen Schwellenwerts

## Cross-Validation

- **K-Fold** - Aufteilung in k gleich große Teile
- **Stratified K-Fold** - Erhalt der Klassenverteilung
- **Leave-One-Out** - Spezialfall mit k = Anzahl Samples
- **Time Series Split** - Für zeitabhängige Daten

## Overfitting & Underfitting

- **Overfitting Detection** - Training vs. Validation Error
- **Regularisierung** - L1, L2, Elastic Net
- **Learning Curves** - Visualisierung des Lernverhaltens
- **Bias-Variance Tradeoff** - Balance zwischen Modellkomplexität und Generalisierung

_Detaillierte Inhalte folgen_


---

**Version:** 1.0
**Stand:** Januar 2026
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.
