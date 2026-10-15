---
layout: default
title: Interaktive Visualisierung
parent: Ressourcen
nav_order: 1
description: "Interaktive Web-Demos für ML-Konzepte: Statistik, Preprocessing, Modelle und Evaluierung"
has_toc: true
---

# Interaktive Visualisierung
{: .no_toc }

> **Interaktive Web-Demos für Machine Learning Konzepte - organisiert nach dem ML-Workflow**

---

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 📊 Understand

Explorative Datenanalyse und statistische Grundlagen

| Thema                       | Beschreibung                            | Link                                                                           |
| --------------------------- | --------------------------------------- | ------------------------------------------------------------------------------ |
| **Lage- und Streuungsmaße** | Mittelwert, Varianz, Standardabweichung | [Statistik_01]({{ '/visualisierungen/statistik_01/' | relative_url }}){:target="_blank" rel="noopener"}          |
| **Verteilungsmaße**         | Quantile, Boxplots, Verteilungen        | [Verteilungsmasse]({{ '/visualisierungen/verteilungsmasse/' | relative_url }}){:target="_blank" rel="noopener"}      |
| **Korrelation & Kovarianz** | Beziehungen zwischen Variablen          | [Korrelation_Kovarianz]({{ '/visualisierungen/korrelation_kovarianz/' | relative_url }}){:target="_blank" rel="noopener"} |
| **Häufigkeit & Entropie**   | Häufigkeiten und Kategorien             | [Haeufigkeit_Entropie]({{ '/visualisierungen/haeufigkeit_entropie/' | relative_url }}){:target="_blank" rel="noopener"}  |

---

## ✂️ Prepare

Datenaufbereitung und Feature Engineering

| Thema | Beschreibung | Link |
|-------|--------------|------|
| **Missing Values** | Umgang mit fehlenden Werten | [MissingValues]({{ '/visualisierungen/missing_values/' | relative_url }}){:target="_blank" rel="noopener"} |
| **Kodierung** | One-Hot, Label Encoding | [Kodierung]({{ '/visualisierungen/encoding/' | relative_url }}){:target="_blank" rel="noopener"} |
| **Skalierung** | Standardisierung & Normalisierung | [StandardisierungNormalisierung]({{ '/visualisierungen/standardisierung_normalisierung/' | relative_url }}){:target="_blank" rel="noopener"} |
| **Outlier** | Ausreißererkennung und -behandlung | [Outlier]({{ '/visualisierungen/outlier/' | relative_url }}){:target="_blank" rel="noopener"} |

---

## 🏃 Modeling

Machine Learning Algorithmen und Modellierung

### Supervised Learning

| Thema                  | Beschreibung                            | Link                                                                       |
| ---------------------- | --------------------------------------- | -------------------------------------------------------------------------- |
| **Entscheidungsbaum**  | Interaktive Tree-Visualisierung         | [Entscheidungsbaum]({{ '/visualisierungen/entscheidungsbaum/' | relative_url }}){:target="_blank" rel="noopener"} |
| **Entropy**            | Information Gain und Splits             | [Entropy]({{ '/visualisierungen/entropy/' | relative_url }}){:target="_blank" rel="noopener"}           |
| **Gini/Entropy**       | Split-Kriterien im Entscheidungsbaum    | [Gini/Entropy]({{ '/visualisierungen/dt_split/' | relative_url }}){:target="_blank" rel="noopener"}      |
| **Lineare Regression** | Regressionslinie und Residuen           | [LineareRegression]({{ '/visualisierungen/lineare_regression/' | relative_url }}){:target="_blank" rel="noopener"} |
| **Random Forest**      | Visualisierung der Entscheidungsfindung | [Random Forest]({{ '/visualisierungen/random_forest/' | relative_url }}){:target="_blank" rel="noopener"}     |
| **XGBoost**            | Abfolge lernender Entscheidungsbäume    | [xgboost]({{ '/visualisierungen/xgboost/' | relative_url }}){:target="_blank" rel="noopener"}           |
| **Ensemble Stacking**  | Kombination mehrerer Algorithmen        | [Stacking]({{ '/visualisierungen/stacking/' | relative_url }}){:target="_blank" rel="noopener"}          |

### Unsupervised Learning

| Thema       | Beschreibung             | Link                                                                                   |
| ----------- | ------------------------ | -------------------------------------------------------------------------------------- |
| **K-Means** | Clustering-Algorithmus   | [KMeans]({{ '/visualisierungen/kmeans/' | relative_url }}){:target="_blank" rel="noopener"}                        |
| **DBSCAN**  | Density-based Clustering | [DBScan]({{ '/visualisierungen/dbscan/' | relative_url }}){:target="_blank" rel="noopener"}                        |
| **Apriori** | Assoziationsanalyse      | [Associationanalysis]({{ '/visualisierungen/association_analysis/' | relative_url }}){:target="_blank" rel="noopener"}           |
| **PCA**     | Hauptkomponentenanalyse  | [PCA]({{ '/visualisierungen/pca/' | relative_url }}){:target="_blank" rel="noopener"}                           |
| **PCA**     | Hauptkomponentenanalyse  | [PCA - Lamarr-Institut](https://lamarr-institute.org/de/blog/dimensionsreduktion-pca/) |

### Neural Networks

| Thema           | Beschreibung                | Link                                                                                                  |
| --------------- | --------------------------- | ----------------------------------------------------------------------------------------------------- |
| **NN-Training** | Neuronales Netz - Training  | [NN - Training - neu]({{ '/visualisierungen/nn_training_neu/' | relative_url }}){:target="_blank" rel="noopener"}                          |
| **Gewichte**    | Einfache Netze visualisiert | [Einfache Netze](https://raw.githack.com/IMAGINARY/simple-networks/feature/pasch-workshop/index.html) |
| **Architektur** | NN-Architektur Designer     | [Neural Numbers](https://imaginary.github.io/neural-numbers/index.html#design-network)                |

### Convolutional Neural Networks (CNN)

| Thema | Beschreibung | Link |
|-------|--------------|------|
| **CNN Filter Demo** | Convolutional Filter in Aktion | [CNN Demo]({{ '/visualisierungen/cnn_demo/' | relative_url }}){:target="_blank" rel="noopener"} |
| **CNN Filter Matrix** | Filter-Matrizen verstehen | [CNN Filter]({{ '/visualisierungen/cnn_filter/' | relative_url }}){:target="_blank" rel="noopener"} |

### Regularisierung & Weitere

| Thema | Beschreibung | Link |
|-------|--------------|------|
| **Regularisierung** | Ridge, Lasso, Elastic Net | [Regularization]({{ '/visualisierungen/regularization/' | relative_url }}){:target="_blank" rel="noopener"} |
| **Reinforcement Learning** | RL interaktiv erleben | [Reinforcement Learning](https://imaginary.github.io/reinforcement-learning-2/exhibit.html?lang=de) |

---

## 🔭 Evaluate

Modellbewertung und Optimierung

| Thema | Beschreibung | Link |
|-------|--------------|------|
| **Klassifikationsmetriken** | Precision, Recall, F1-Score | [Threshold]({{ '/visualisierungen/threshold/' | relative_url }}){:target="_blank" rel="noopener"} |
| **Threshold Optimization** | Schwellenwert-Optimierung | [Threshold]({{ '/visualisierungen/threshold/' | relative_url }}){:target="_blank" rel="noopener"} |
| **Cross Validation** | k-Fold Cross-Validation | [Cross-Validation]({{ '/visualisierungen/cross_validation/' | relative_url }}){:target="_blank" rel="noopener"} |
| **Bootstrapping** | Resampling-Methode | [Bootstrapping]({{ '/visualisierungen/bootstrapping/' | relative_url }}){:target="_blank" rel="noopener"} |

## Abgrenzung zu verwandten Dokumenten

| Dokument | Frage |
|---|---|

---

**Version:** 1.0<br>
**Stand:** Januar 2026<br>
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.
