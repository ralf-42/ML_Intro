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

- **Metriken** - Bewertungsmaße für Klassifikation, Regression, Clustering
- **Cross-Validation** - Robuste Modellbewertung durch Kreuzvalidierung
- **Hyperparameter-Tuning** - Optimierung der Modellparameter
- **Overfitting vermeiden** - Regularisierung und Generalisierung

---

## Metriken (Klassifikation / Regression)

Bewertungsmetriken für verschiedene ML-Aufgaben.

**[Allgemein](https://ralf-42.github.io/ML_Intro/concepts/evaluate/bewertung_allgemein.html)** – *Wie wird die Grundlogik der Modellbewertung verstanden?* Grundlagen der Modellbewertung: Trainings- vs. Testfehler, Bias-Variance-Tradeoff.

**[Klassifizierung](https://ralf-42.github.io/ML_Intro/concepts/evaluate/bewertung_klassifizierung.html)** – *Welche Metriken zeigen, ob ein Klassifikationsmodell wirklich taugt?* Confusion Matrix, Precision, Recall, F1-Score und ROC-AUC.

**[Regression](https://ralf-42.github.io/ML_Intro/concepts/evaluate/bewertung_regression.html)** – *Woran wird die Qualität eines Regressionsmodells gemessen?* R², MSE, RMSE, MAE und Residuenanalyse.

**[Clustering](https://ralf-42.github.io/ML_Intro/concepts/evaluate/bewertung_clustering.html)** – *Wie wird bewertet, ob eine Clusterstruktur sinnvoll ist?* Silhouette-Koeffizient und weitere Cluster-Validierungsmetriken.

**[Anomalie](https://ralf-42.github.io/ML_Intro/concepts/evaluate/bewertung_anomalie.html)** – *Wie wird die Güte eines Anomalie-Erkennungsmodells eingeschätzt?* Anomalie-Scores und Methoden zur Ausreißer-Erkennung.

## Cross-Validation

Robuste Modellbewertung durch wiederholte Aufteilung der Daten.

**[Cross-Validation](https://ralf-42.github.io/ML_Intro/concepts/evaluate/cross_validation.html)** – *Wie liefert Kreuzvalidierung eine robuste Schätzung der Modellgüte?* K-Fold, Stratified K-Fold, Leave-One-Out und Time Series Split.

**[Bootstrapping](https://ralf-42.github.io/ML_Intro/concepts/evaluate/bootstrapping.html)** – *Wie quantifiziert Resampling die Unsicherheit einer Schätzung?* Resampling-Verfahren zur Berechnung von Konfidenzintervallen und Modellunsicherheit.

## Hyperparameter-Tuning

Systematische Optimierung der Modellparameter.

**[Hyperparameter-Tuning](https://ralf-42.github.io/ML_Intro/concepts/evaluate/hyperparameter_tuning.html)** – *Wie werden Modellparameter systematisch optimiert?* Grid Search, Random Search und Bayesian Optimization im Vergleich.

## Overfitting vermeiden

Strategien zur Verbesserung der Generalisierung.

**[Overfitting](https://ralf-42.github.io/ML_Intro/concepts/evaluate/overfitting.html)** – *Woran wird Overfitting erkannt und wie wird es verhindert?* Learning Curves, Bias-Variance Tradeoff und praktische Gegenmaßnahmen.

**[Regularisierung](https://ralf-42.github.io/ML_Intro/concepts/evaluate/regularisierung.html)** – *Wie reduziert Regularisierung Overfitting ohne die Modellkomplexität zu opfern?* L1 (Lasso), L2 (Ridge) und Elastic Net.

