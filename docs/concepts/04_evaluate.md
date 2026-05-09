---
layout: default
title: Evaluate
parent: Alle Konzepte
nav_order: 4
has_children: true
description: Bewertung und Evaluation von ML-Modellen
grand_parent: Konzepte
---

# Evaluate

**Version:** 1.0<br>
**Stand:** Mai 2026<br>
**Kurs:** Machine Learning Einführung

## Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

## Metriken (Klassifikation / Regression)

**[Allgemein](evaluate/bewertung_allgemein.html)** – *Wie wird die Grundlogik der Modellbewertung verstanden?* Grundlagen der Modellbewertung: Trainings- vs. Testfehler, Bias-Variance-Tradeoff.

**[Klassifizierung](evaluate/bewertung_klassifizierung.html)** – *Welche Metriken zeigen, ob ein Klassifikationsmodell wirklich taugt?* Confusion Matrix, Precision, Recall, F1-Score und ROC-AUC.

**[Regression](evaluate/bewertung_regression.html)** – *Woran wird die Qualität eines Regressionsmodells gemessen?* R², MSE, RMSE, MAE und Residuenanalyse.

**[Clustering](evaluate/bewertung_clustering.html)** – *Wie wird bewertet, ob eine Clusterstruktur sinnvoll ist?* Silhouette-Koeffizient und weitere Cluster-Validierungsmetriken.

**[Anomalieerkennung](evaluate/bewertung_anomalie.html)** – *Wie werden Auffälligkeiten und Anomaly Scores interpretiert?* `decision_function`, Thresholds, Top-k-Review und Metriken für seltene Anomalien.

## Cross-Validation

**[Cross-Validation](evaluate/cross_validation.html)** – *Wie liefert Kreuzvalidierung eine robuste Schätzung der Modellgüte?* K-Fold, Stratified K-Fold, Leave-One-Out und Time Series Split.

**[Bootstrapping](evaluate/bootstrapping.html)** – *Wie quantifiziert Resampling die Unsicherheit einer Schätzung?* Resampling-Verfahren zur Berechnung von Konfidenzintervallen und Modellunsicherheit.

## Hyperparameter-Tuning

**[Hyperparameter-Tuning](evaluate/hyperparameter_tuning.html)** – *Wie werden Modellparameter systematisch optimiert?* Grid Search, Random Search und Bayesian Optimization im Vergleich.

## Overfitting vermeiden

**[Overfitting](evaluate/overfitting.html)** – *Woran wird Overfitting erkannt und wie wird es verhindert?* Learning Curves, Bias-Variance Tradeoff und praktische Gegenmaßnahmen.

**[Regularisierung](evaluate/regularisierung.html)** – *Wie reduziert Regularisierung Overfitting ohne die Modellkomplexität zu opfern?* L1 (Lasso), L2 (Ridge) und Elastic Net.

## Abgrenzung zu verwandten Dokumenten

| Dokument | Frage |
|---|---|
