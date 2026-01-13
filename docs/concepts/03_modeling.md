---
layout: default
title: Modeling
parent: Konzepte
nav_order: 3
has_children: true
description: "Machine Learning Modelle und Algorithmen"
---

# Modeling

Überblick über klassische Machine Learning Modelle und ihre Anwendungsfälle.

## Modellauswahl

- **[Modellauswahl](modeling/modellauswahl)** - Systematische Kriterien und Strategien zur Auswahl des optimalen Machine-Learning-Modells für verschiedene Problemstellungen
- **[Modell-Steckbriefe](modeling/modell-steckbriefe)** - Kompakte Übersicht aller wichtigen ML-Algorithmen mit Einsatzbereichen, Eigenschaften und Bewertungsmetriken

## Supervised Learning

### Klassifikation

- **[Entscheidungsbaum](modeling/decision_tree)** - Hierarchische Regelstruktur für Klassifikation und Regression
- Support Vector Machines - Maximal-Margin-Klassifikatoren
- Naive Bayes - Probabilistische Klassifikation

### Regression

- **[Regression](modeling/regression)** - Lineare und logistische Regression für stetige Vorhersagen
- Ridge & Lasso - Regularisierte Regression
- Polynomial Regression - Nicht-lineare Beziehungen

## Unsupervised Learning

### Clustering

- **[K-Means & DBSCAN](modeling/kmeans-dbscan)** - Partitions- und dichtebasiertes Clustering
- Hierarchisches Clustering - Agglomerative und divisive Methoden
- Mean Shift - Dichtebasiertes nicht-parametrisches Clustering

### Anomalie-Erkennung

- **[Isolation Forest](modeling/isolation_forest)** - Ensemble-basierte Anomalie-Erkennung
- One-Class SVM - Support Vector Maschinen für Ausreißer
- Local Outlier Factor (LOF) - Lokale Dichteanomalien

### Assoziationsanalyse

- **[Apriori](modeling/apriori)** - Entdeckung von Zusammenhängen in Transaktionsdaten
- FP-Growth - Effiziente Alternative zum Apriori-Algorithmus
- Eclat - Vertikales Mining von häufigen Itemsets

### Dimensionsreduktion

- **[PCA und LDA](modeling/pca-lda)** - Principal Component Analysis und Linear Discriminant Analysis für Dimensionsreduktion
- t-SNE - Visualisierung hochdimensionaler Daten
- UMAP - Manifold-basierte Dimensionsreduktion

## Ensemble Methods

- **[Ensemble-Methoden](modeling/ensemble)** - Übersicht: Bagging, Boosting und Stacking-Konzepte
- **[Random Forest](modeling/random-forest)** - Bagging-basiertes Ensemble: Kombination multipler Entscheidungsbäume für robuste Vorhersagen
- **[XGBoost](modeling/xgboost)** - Extreme Gradient Boosting: Optimierte Boosting-Implementierung für höchste Performance
- **[Stacking](modeling/stacking)** - Kombination heterogener Modelle durch Voting oder Meta-Learning

## Deep Learning

- **[Neuronale Netze](modeling/neuronale-netze)** - Grundlagen künstlicher neuronaler Netze: Architektur, Aktivierungsfunktionen und Training
- **[Spezielle Neuronale Netze](modeling/spezielle-neuronale-netze)** - Computer Vision mit CNNs, Sequenzmodellierung mit RNNs/LSTMs und AutoEncoder

## Automatisierung

- **[AutoML](modeling/automl)** - Automatisiertes Machine Learning: Workflow-Automatisierung von Datenvorbereitung bis Modellauswahl mit PyCaret

---


**Version:** 1.0    
**Stand:** Januar 2026    
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.     
