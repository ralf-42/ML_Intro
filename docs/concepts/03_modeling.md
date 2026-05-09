---
layout: default
title: Modeling
parent: Alle Konzepte
nav_order: 3
has_children: true
description: Machine Learning Modelle und Algorithmen
grand_parent: Konzepte
---

# Modeling

**Version:** 1.0<br>
**Stand:** Mai 2026<br>
**Kurs:** Machine Learning Einführung

Überblick über Machine Learning Modelle und ihre Anwendungsfälle.

---

## Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

## Modellauswahl

Einstiegspunkt für die systematische Auswahl des passenden Algorithmus.

**[Modellauswahl](modeling/modellauswahl.html)** – *Welche Kriterien entscheiden über die Modellwahl?* Systematische Strategien zur Auswahl des optimalen Algorithmus für verschiedene Problemstellungen.

**[Modell-Steckbriefe](modeling/modell-steckbriefe.html)** – *Welches Modell eignet sich für welchen Anwendungsfall?* Kompakte Übersicht aller wichtigen ML-Algorithmen mit Vor- und Nachteilen.

**[Prepare nach Modell](prepare/prepare_nach_modell.html)** – *Welche Vorverarbeitung braucht das gewählte Modell?* Modellbezogene Entscheidungshilfe zu Missing Values, Skalierung, Kodierung und Ausreißern.

## Supervised Learning

Modelle, die aus gelabelten Daten lernen, um Vorhersagen zu treffen.

**[Entscheidungsbaum](modeling/decision_tree.html)** – *Wie trifft ein Entscheidungsbaum Vorhersagen?* Hierarchische Regelstruktur für Klassifikation und Regression.

**[Random Forest](modeling/random-forest.html)** – *Wie verbessert Bagging die Vorhersagequalität?* Ensemble aus multiplen Entscheidungsbäumen mit Feature-Randomisierung.

**[Regression](modeling/regression.html)** – *Wann ist lineare oder logistische Regression die richtige Wahl?* Lineare und logistische Regression für stetige und binäre Vorhersagen.

**[XGBoost](modeling/xgboost.html)** – *Was macht Gradient Boosting schneller und präziser als andere Ensembles?* Extreme Gradient Boosting für höchste Performance bei tabellarischen Daten.

## Unsupervised Learning

Modelle, die Strukturen in ungelabelten Daten entdecken.

**[Clustering (K-Means & DBSCAN)](modeling/kmeans-dbscan.html)** – *Wie werden Datenpunkte ohne Labels sinnvoll gruppiert?* Partitions- und dichtebasiertes Clustering im Vergleich.

**[Isolation Forest](modeling/isolation-forest.html)** – *Wie werden Anomalien ohne Labels erkannt?* Baum-Ensemble zur unüberwachten Ausreißer- und Anomalieerkennung.

**[PCA und LDA](modeling/pca-lda.html)** – *Wie werden hochdimensionale Daten reduziert ohne wesentliche Information zu verlieren?* Dimensionsreduktion und Visualisierung mit PCA und LDA.

**[Apriori](modeling/apriori.html)** – *Wie werden häufige Muster in Transaktionsdaten gefunden?* Association Rules und Warenkorbanalyse mit dem Apriori-Algorithmus.

## Deep Learning

Neuronale Netze für komplexe Muster in Bildern, Text und Sequenzen.

**[Neuronale Netze](modeling/neuronale-netze.html)** – *Wie lernen neuronale Netze aus Daten?* Grundlagen: Architektur, Aktivierungsfunktionen und Training.

**[Spezielle Neuronale Netze](modeling/spezielle-neuronale-netze.html)** – *Welche Netzarchitektur eignet sich für Bilder, Zeitreihen oder Anomalien?* CNN (Computer Vision), RNN/LSTM (Zeitreihen) und AutoEncoder.

## Ensemble-Methoden

Kombination mehrerer Modelle für bessere Vorhersagen.

**[Ensemble-Methoden](modeling/ensemble.html)** – *Wie werden mehrere Modelle zu einem besseren kombiniert?* Übersicht über Bagging, Boosting und Stacking-Konzepte.

**[Random Forest](modeling/random-forest.html)** – *Wie verbessert Bagging die Vorhersagequalität?* Bagging-Ensemble aus multiplen Entscheidungsbäumen.

**[XGBoost](modeling/xgboost.html)** – *Was macht Gradient Boosting schneller und präziser als andere Ensembles?* Extreme Gradient Boosting für höchste Performance.

**[Stacking](modeling/stacking.html)** – *Wie lernt ein Meta-Modell aus den Vorhersagen anderer Modelle?* Kombination heterogener Modelle durch Meta-Learning.

## Automatisierung

**[AutoML](modeling/automl.html)** – *Wie automatisiert PyCaret den ML-Workflow?* Workflow-Automatisierung von Datenvorbereitung bis Modellauswahl.

## Abgrenzung zu verwandten Dokumenten

| Dokument | Frage |
|---|---|
