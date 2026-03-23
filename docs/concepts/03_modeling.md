---
layout: default
title: Modeling
parent: Konzepte
nav_order: 3
has_children: true
has_toc: true
description: "Machine Learning Modelle und Algorithmen"
---

# Modeling
{: .no_toc }

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---


Überblick über Machine Learning Modelle und ihre Anwendungsfälle.

---

## Modellauswahl

Einstiegspunkt für die systematische Auswahl des passenden Algorithmus.

| Thema | Beschreibung |
|-------|--------------|
| [Modellauswahl](modeling/modellauswahl) | Systematische Kriterien und Strategien zur Auswahl des optimalen Modells |
| [Modell-Steckbriefe](modeling/modell-steckbriefe) | Kompakte Übersicht aller wichtigen ML-Algorithmen |

---

## Supervised Learning

Modelle, die aus gelabelten Daten lernen, um Vorhersagen zu treffen.

| Thema                                       | Beschreibung                                                  |
| ------------------------------------------- | ------------------------------------------------------------- |
| [Entscheidungsbaum](modeling/decision_tree) | Hierarchische Regelstruktur für Klassifikation und Regression |
| [Random Forest](modeling/random-forest)     | Bagging-Ensemble aus multiplen Entscheidungsbäumen            |
| [Regression](modeling/regression)           | Lineare und logistische Regression für stetige Vorhersagen    |
| [XGBoost](modeling/xgboost)                 | Extreme Gradient Boosting für höchste Performance             |

---

## Unsupervised Learning

Modelle, die Strukturen in ungelabelten Daten entdecken.

| Thema | Beschreibung |
|-------|--------------|
| [Clustering (K-Means & DBSCAN)](modeling/kmeans-dbscan) | Partitions- und dichtebasiertes Clustering |
| [PCA und LDA](modeling/pca-lda) | Dimensionsreduktion und Visualisierung |
| [Apriori](modeling/apriori) | Association Rules für Warenkorbanalyse |

---

## Deep Learning

Neuronale Netze für komplexe Muster in Bildern, Text und Sequenzen.

| Thema | Beschreibung |
|-------|--------------|
| [Neuronale Netze](modeling/neuronale-netze) | Grundlagen: Architektur, Aktivierungsfunktionen, Training |
| [Spezielle Neuronale Netze](modeling/spezielle-neuronale-netze) | CNN (Computer Vision), RNN/LSTM (Zeitreihen), AutoEncoder |

---

## Ensemble-Methoden

Kombination mehrerer Modelle für bessere Vorhersagen.

| Thema                                   | Beschreibung                                        |
| --------------------------------------- | --------------------------------------------------- |
| [Ensemble-Methoden](modeling/ensemble)  | Übersicht: Bagging, Boosting und Stacking-Konzepte  |
| [Random Forest](modeling/random-forest) | Bagging-Ensemble aus multiplen Entscheidungsbäumen  |
| [XGBoost](modeling/xgboost)             | Extreme Gradient Boosting für höchste Performance   |
| [Stacking](modeling/stacking)           | Kombination heterogener Modelle durch Meta-Learning |

---

## Automatisierung

| Thema | Beschreibung |
|-------|--------------|
| [AutoML](modeling/automl) | Workflow-Automatisierung mit PyCaret |

## Abgrenzung zu verwandten Dokumenten

| Dokument | Frage |
|---|---|
| [Grundlagen](./01_grundlagen.html) | Welche Problemtypen und Lernparadigmen liegen der Modellwahl zugrunde? |
| [Prepare](./02_prepare.html) | Welche Vorverarbeitung ist nötig, bevor ein Modell sinnvoll trainiert werden kann? |
| [Evaluate](./04_evaluate.html) | Wie wird entschieden, ob ein gewähltes Modell tatsächlich überzeugt? |
| [XAI](./08_xai.html) | Wie lassen sich komplexere Modelle später nachvollziehbar erläutern? |

---

**Version:** 1.0
**Stand:** März 2026
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.
