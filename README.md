# Machine Learning Kurs

<table>
  <tr>
    <td><a href="./README.md"><img src="https://img.shields.io/badge/Last%20Updated-2026--10--24-blue" alt="Last Updated 2026-10-24"></a></td>
    <td><a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-brightgreen" alt="Python 3.11+"></a></td>
    <td><a href="./README.md"><img src="https://img.shields.io/badge/Status-Active-success" alt="Status Active"></a></td>
    <td><a href="./license"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License MIT"></a></td>
    <td><a href="./01_notebook/"><img src="https://img.shields.io/badge/Notebooks-45-orange" alt="Notebooks 45"></a></td>
    <td><a href="./01_notebook/"><img src="https://img.shields.io/badge/Modules-10-blueviolet" alt="Modules 10"></a></td>
  </tr>
</table>

Dieses Repository enthält Lehrmaterialien für den Machine Learning-Kurs.

Dieser Kurs bietet einen strukturierten Lernpfad von grundlegenden Machine Learning-Konzepten bis hin zu fortgeschrittenen Deep Learning-Anwendungen und umfasst sowohl überwachte als auch unüberwachte Lerntechniken, neuronale Netze, Ensemble-Methoden und moderne generative KI-Anwendungen.

## 🌐 Kurs-Website

**[→ ML Kurs-Website (GitHub Pages)](https://ralf-42.github.io/ML_Intro)**

Die Web-Dokumentation bietet:
- Interaktive Navigation durch alle Konzepte und Frameworks
- Mermaid-Diagramme zur Visualisierung von ML-Workflows
- Strukturierte Übersicht über Deployment, Regulatorisches und Ressourcen

## 🎯 Zielgruppe

Deutschsprachige Studierende und Praktiker:innen mit Python-Grundkenntnissen, die Machine Learning praxisnah an realen Datensätzen erlernen möchten.

## 📁 Projektstruktur

```
ML_Intro/
├── 01_notebook/    # Jupyter Notebooks mit Kursinhalten (10 Module)
├── 02_daten/       # Datensätze für praktische Übungen
├── 03_skript/      # Präsentationsmaterialien und Skripte
├── 04_model/       # Trainierte Modelle (z. B. diamonds_model.pmml, diamonds_pipeline.joblib)
├── 07_image/       # Bilder und Visualisierungen für die Kursnotebooks
└── docs/           # Kursdokumentation und rechtliche Hinweise
```

## 🛠️ Technologie-Stack

### Kernbibliotheken
- **Machine Learning**: scikit-learn, pandas, numpy
- **Deep Learning**: Keras, TensorFlow
- **Datenvisualisierung**: matplotlib, plotly
- **Spezialisierte ML**: XGBoost, PyCaret
- **Zeitreihen**: Amazon Chronos
- **Modell-Export**: pypmml, nyoka (PMML-Format), joblib
- **Generative KI**: Google Gemini API
- **Webanwendungen**: Gradio

## 📚 Kursmodule

Jedes Modul liegt in einem eigenen Unterordner von `01_notebook/`.

### 🟢 Modul 00: Allgemeine Konzepte
**Pfad**: `01_notebook/00_general/`
- Grundlegende ML-Konzepte und pandas-Grundlagen
- Datensatzbehandlung und -exploration
- Beispiele: `a000_snippets_ml.ipynb`, `b020_pandas_basics.ipynb`, `b040_datasets.ipynb`

### 🔵 Modul 01: Überwachtes Lernen
**Pfad**: `01_notebook/01_supervised/`
- Entscheidungsbäume (Titanic-Datensatz)
- Lineare Regression (MPG-Vorhersage)
- Beispiele: `b110_sl_dt_titanic.ipynb`, `b120_sl_lr_mpg.ipynb`

### 🟣 Modul 02: Unüberwachtes Lernen
**Pfad**: `01_notebook/02_unsupervised/`
- K-means und DBSCAN-Clustering (Standortdaten, NID-Datensatz)
- Isolation Forest für Anomalieerkennung
- PCA für Dimensionsreduktion
- Assoziationsregeln (Apriori-Algorithmus, Lebensmittel)
- Beispiele: `b200_ul_kmeans_dbscan_location.ipynb`, `b221_ul_isoforest_nid.ipynb`, `b230_ul_apriori_food.ipynb`, `b240_ul_pca_special.ipynb`

### 🟡 Modul 03: Neuronale Netze
**Pfad**: `01_notebook/03_network/`
- Multi-Layer Perceptron (MLP) Implementierungen
- Keras/TensorFlow neuronale Netze
- Anwendungen auf Krebs- und Diamant-Datensätzen
- Beispiele: `b310_nn_mlp_cancer.ipynb`, `b320_nn_keras_cancer.ipynb`, `b330_nn_mlp_diamonds.ipynb`, `b340_nn_keras_diamonds.ipynb`

### 🟠 Modul 04: Ensemble-Methoden
**Pfad**: `01_notebook/04_ensemble/`
- Random Forests (Diamantpreisvorhersage)
- XGBoost-Implementierung (Krebs, Diamanten)
- Stacking-Ensemble-Techniken
- Beispiele: `b400_sl_rf_diamonds_inverse.ipynb`, `b410_xg_cancer.ipynb`, `b420_xg_diamonds.ipynb`, `b430_stacking_titanic.ipynb`

### ⚪ Modul 05: Modell-Tuning & Validierung
**Pfad**: `01_notebook/05_tuning/`
- Kreuzvalidierungstechniken (Titanic, Diamonds)
- Bootstrapping
- Hyperparameter-Optimierung (Grid Search, Random Search)
- ROC-AUC-Analyse und Schwellenwertoptimierung
- Validierungskurven
- AutoML mit PyCaret
- Beispiele: `b510_cv_dt_titanic.ipynb`, `b515_cv_dt_diamonds.ipynb`, `b516_cv_rf_diamonds.ipynb`, `b520_bootstrapping_dt_titanic.ipynb`, `b530_gridsearch_nn_mlp_cancer.ipynb`, `b550_randomizedsearch_rf_diamonds.ipynb`, `b560_randomsearch_keras_ccpp.ipynb`, `b570_roc_auc_threshold.ipynb`, `b580_validation_curve_titanic.ipynb`, `b590_automl_pycaret_regression.ipynb`

### 🔴 Modul 06: ML-Workflows
**Pfad**: `01_notebook/06_workflow/`
- Scikit-learn Pipelines
- End-to-End ML-Workflow-Automatisierung
- Beispiel: `b600_pipeline_dt_diamonds.ipynb`

### 🟤 Modul 07: Spezialisierte NN-Anwendungen
**Pfad**: `01_notebook/07_special_nn/`
- Computer Vision (MNIST, YOLO)
- Natural Language Processing (Spam-Erkennung)
- Zeitreihenanalyse (Wettervorhersage mit Keras und Amazon Chronos)
- Autoencoder für Dimensionsreduktion (NID-Datensatz)
- Beispiele: `b710_vision_keras_mnist.ipynb`, `b715_vision_yolo.ipynb`, `b720_nlp_keras_spam.ipynb`, `b730_ts_keras_wetter.ipynb`, `b740_ts_chronos_wetter.ipynb`, `b750_autoencoder_nid.ipynb`

### ⚫ Modul 08: Save, Load & Deployment
**Pfad**: `01_notebook/08_save_load_deploy/`
- Modellpersistierung und Modell-Export (PMML, Pipeline, Keras)
- Gradio-Webanwendungen für Diamonds-Modelle
- Beispiele: `b810_save_load_pmml_dt_diamonds.ipynb`, `b820_data_app_gradio_diamonds.ipynb`, `b830_save_load_pipeline_rf_diamonds.ipynb`, `b840_data_app_gradio_pipeline_diamonds.ipynb`, `b850_save_load_keras_sinus_multi.ipynb`

### ⚫ Modul 09: XAI & KI-Integration
**Pfad**: `01_notebook/09_xai_ai/`
- Erklärbare KI (XAI) mit Titanic-Beispielen und Framework-Vergleich
- Vertiefung zu umfangreicheren XAI-Analysen
- KI-Integration und Business Intelligence mit Google Gemini AI (Kreditdaten, Verkehr)
- Beispiele: `b900_xai_titanic.ipynb`, `b910_xai_xxl_titanic.ipynb`, `b940_ai_integration.ipynb`, `b950_analyse_credit_data_with_gemini_chat.ipynb`, `b960_analyse_traffic_mit_gemini_chat.ipynb`

## 🗂️ Datensatz-Sammlung

Der Kurs verwendet verschiedene reale Datensätze für praktisches Lernen (unter `02_daten/`):

### Tabellarische Daten (`02_daten/05_tabellen/`)
- **`titanic.xlsx`** - Überlebensvorhersage (Klassifikation)
- **`diamonds.csv`** - Preisvorhersage (Regression)
- **`breast_cancer_wisconsin.csv`** - Medizinische Diagnose (Klassifikation)
- **`auto_mpg.csv`** - Kraftstoffeffizienzvorhersage (Regression)
- **`ccpp.csv`** - Combined Cycle Power Plant (Regression)
- **`wa_fn_usec__telco_customer_churn.csv`** - Kundenabwanderung
- **`smsspamcollection.csv`** - SMS Spam-Erkennung (NLP)
- **`credit.csv`** - Kreditdatenanalyse
- **`weather_hobart.csv`** - Wetterdaten für Zeitreihenanalyse
- Und weitere spezialisierte Datensätze

### Text-Daten (`02_daten/01_text/`)
- **`maintenance_data.txt`** und **`maintenance_features.txt`** - Textdaten zur Maintenance-Analyse

### Bild-Daten (`02_daten/02_bild/`)
- Bilddateien für Computer Vision-Aufgaben

### Video-Daten (`02_daten/04_video/`)
- **`pexels_pixabay_people.mp4`** - Videoanalyse

## 🔑 Voraussetzungen

- **Laufzeit**: Python 3.11+, Java 8+ (für PMML)
- **IDE**: Google Colab, Jupyter Notebook/Lab

## 💡 Nutzung

Alle Notebooks sind eigenständig lauffähig und für Google Colab optimiert. Sie folgen demselben 5-Phasen-Workflow (Understand, Prepare, Modeling, Evaluate, Deploy); jede Phase endet mit einer Checkliste zur Lernkontrolle.

## 🎓 Lernziele

Nach Abschluss dieses Kurses können Studierende:

- ✅ Grundlegende ML-Konzepte und Workflows verstehen
- ✅ Überwachte und unüberwachte Lernalgorithmen implementieren
- ✅ Neuronale Netze mit Keras/TensorFlow erstellen und optimieren
- ✅ Ensemble-Methoden und fortgeschrittene ML-Techniken anwenden
- ✅ Hyperparameter-Tuning und Modellvalidierung durchführen
- ✅ Mit spezialisierten Anwendungen arbeiten (CV, NLP, Zeitreihen)
- ✅ Einsetzbare ML-Anwendungen mit modernen Tools erstellen
- ✅ Machine Learning-Modelle interpretieren und erklären
- ✅ Modelle im PMML-Format exportieren und portabel deployen

## 📖 Dokumentation

- **Kurs-Website:** [https://ralf-42.github.io/ML_Intro](https://ralf-42.github.io/ML_Intro)

## ⚖️ Lizenzen

Der **Quellcode** steht unter der [MIT License](./license).
Die **Kursmaterialien** (z. B. Folien, Texte, Grafiken) sind unter der [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) veröffentlicht, soweit nicht anders angegeben.      

Die ausführliche Lizenzzuordnung für die Kurs-Website steht unter [Rechtliches → Lizenzen](./docs/legal/lizenzen.md).

© 2025-2026 Ralf-42      

> [!NOTE]
> **Hinweis zur Entstehung:** Bei der Erstellung dieser Unterlagen wurden KI-Werkzeuge unterstützend eingesetzt, unter anderem für Recherche, Strukturierung, Formulierung und einzelne Medien. Die Inhalte wurden anschließend fachlich geprüft, eingeordnet und redaktionell überarbeitet.

---

**Stand:** Oktober 2026     
**Version:** 1.2     
