# Machine Learning Kurs

[![Last Updated](https://img.shields.io/badge/Last%20Updated-2026--04--29-blue)](./README.md)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-brightgreen)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./license)
[![Notebooks](https://img.shields.io/badge/Notebooks-45-orange)](./01_notebook/)
[![Modules](https://img.shields.io/badge/Modules-10-blueviolet)](./01_notebook/)

Dieses Repository enthält Lehrmaterialien für den Machine Learning-Kurs.      

## 📖 Dokumentation

Die vollständige Kursdokumentation ist verfügbar unter:  **[https://ralf-42.github.io/ML_Intro](https://ralf-42.github.io/ML_Intro)**

Die Web-Dokumentation bietet:
- Interaktive Navigation durch alle Konzepte und Frameworks
- Mermaid-Diagramme zur Visualisierung von ML-Workflows
- Strukturierte Übersicht über Deployment, Regulatorisches und Ressourcen

# 1 📚 Kursübersicht
Dieser Kurs bietet einen strukturierten Lernpfad von grundlegenden Machine Learning-Konzepten bis hin zu fortgeschrittenen Deep Learning-Anwendungen und umfasst sowohl überwachte als auch unüberwachte Lerntechniken, neuronale Netze, Ensemble-Methoden und moderne generative KI-Anwendungen.

# 2 🎓 Lernziele

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

# 3 📚 Repository-Struktur

## 3.1 Hauptverzeichnisse

- **`01_notebook/`** - Jupyter Notebooks mit Kursinhalten (10 Module)
- **`02_daten/`** - Datensätze für praktische Übungen
- **`03_skript/`** - Präsentationsmaterialien und Skripte
- **`04_model/`** - Trainierte Modelle (z. B. `diamonds_model.pmml`, `diamonds_pipeline.joblib`)

## 3.2 Kursmodule in `01_notebook/`

### 3.2.1 🟢 Modul 00: Allgemeine Konzepte
**Pfad**: `01_notebook/00_general/`
- Grundlegende ML-Konzepte und pandas-Grundlagen
- Datensatzbehandlung und -exploration
- Beispiele: `a000_snippets_ml.ipynb`, `b020_pandas_basics.ipynb`, `b040_datasets.ipynb`

### 3.2.2 🔵 Modul 01: Überwachtes Lernen
**Pfad**: `01_notebook/01_supervised/`
- Entscheidungsbäume (Titanic-Datensatz)
- Lineare Regression (MPG-Vorhersage)
- Beispiele: `b110_sl_dt_titanic.ipynb`, `b120_sl_lr_mpg.ipynb`

### 3.2.3 🟣 Modul 02: Unüberwachtes Lernen
**Pfad**: `01_notebook/02_unsupervised/`
- K-means und DBSCAN-Clustering (Standortdaten, NID-Datensatz)
- Isolation Forest für Anomalieerkennung
- PCA für Dimensionsreduktion
- Assoziationsregeln (Apriori-Algorithmus, Lebensmittel)
- Beispiele: `b200_ul_kmeans_dbscan_location.ipynb`, `b220_ul_dbscan_nid.ipynb`, `b230_ul_apriori_food.ipynb`, `b240_ul_pca_special.ipynb`

### 3.2.4 🟡 Modul 03: Neuronale Netze
**Pfad**: `01_notebook/03_network/`
- Multi-Layer Perceptron (MLP) Implementierungen
- Keras/TensorFlow neuronale Netze
- Anwendungen auf Krebs- und Diamant-Datensätzen
- Beispiele: `b310_nn_mlp_cancer.ipynb`, `b320_nn_keras_cancer.ipynb`, `b330_nn_mlp_diamonds.ipynb`, `b340_nn_keras_diamonds.ipynb`

### 3.2.5 🟠 Modul 04: Ensemble-Methoden
**Pfad**: `01_notebook/04_ensemble/`
- Random Forests (Diamantpreisvorhersage)
- XGBoost-Implementierung (Krebs, Diamanten)
- Stacking-Ensemble-Techniken
- Beispiele: `b400_sl_rf_diamonds_inverse.ipynb`, `b410_xg_cancer.ipynb`, `b420_xg_diamonds.ipynb`, `b430_stacking_titanic.ipynb`

### 3.2.6 ⚪ Modul 05: Modell-Tuning & Validierung
**Pfad**: `01_notebook/05_tuning/`
- Kreuzvalidierungstechniken (Titanic, Diamonds)
- Bootstrapping
- Hyperparameter-Optimierung (Grid Search, Random Search)
- ROC-AUC-Analyse und Schwellenwertoptimierung
- Validierungskurven
- AutoML mit PyCaret
- Beispiele: `b510_cv_dt_titanic.ipynb`, `b515_cv_dt_diamonds.ipynb`, `b516_cv_rf_diamonds.ipynb`, `b520_bootstrapping_dt_titanic.ipynb`, `b530_gridsearch_nn_mlp_cancer.ipynb`, `b550_randomizedsearch_rf_diamonds.ipynb`, `b560_randomsearch_keras_ccpp.ipynb`, `b570_roc_auc_threshold.ipynb`, `b580_validation_curve_titanic.ipynb`, `b590_automl_pycaret_regression.ipynb`

### 3.2.7 🔴 Modul 06: ML-Workflows
**Pfad**: `01_notebook/06_workflow/`
- Scikit-learn Pipelines
- End-to-End ML-Workflow-Automatisierung
- Beispiel: `b600_pipeline_dt_diamonds.ipynb`

### 3.2.8 🟤 Modul 07: Spezialisierte NN-Anwendungen
**Pfad**: `01_notebook/07_special_nn/`
- Computer Vision (MNIST, YOLO)
- Natural Language Processing (Spam-Erkennung)
- Zeitreihenanalyse (Wettervorhersage mit Keras und Amazon Chronos)
- Autoencoder für Dimensionsreduktion (NID-Datensatz)
- Beispiele: `b710_vision_keras_mnist.ipynb`, `b715_vision_yolo.ipynb`, `b720_nlp_keras_spam.ipynb`, `b730_ts_keras_wetter.ipynb`, `b740_ts_chronos_wetter.ipynb`, `b750_autoencoder_nid.ipynb`

### 3.2.9 ⚫ Modul 08: Save, Load & Deployment
**Pfad**: `01_notebook/08_save_load_deploy/`
- Modellpersistierung und Modell-Export (PMML, Pipeline, Keras)
- Gradio-Webanwendungen für Diamonds-Modelle
- Beispiele: `b810_save_load_pmml_dt_diamonds.ipynb`, `b820_data_app_gradio_diamonds.ipynb`, `b830_save_load_pipeline_rf_diamonds.ipynb`, `b840_data_app_gradio_pipeline_diamonds.ipynb`, `b850_save_load_keras_sinus_multi.ipynb`

### 3.2.10 ⚫ Modul 09: XAI & KI-Integration
**Pfad**: `01_notebook/09_xai_ai/`
- Erklärbare KI (XAI) mit Titanic-Beispielen und Framework-Vergleich
- Vertiefung zu umfangreicheren XAI-Analysen
- KI-Integration und Business Intelligence mit Google Gemini AI (Kreditdaten, Verkehr)
- Beispiele: `b900_xai_titanic.ipynb`, `b910_xai_xxl_titanic.ipynb`, `b940_ai_integration.ipynb`, `b950_analyse_credit_data_with_gemini_chat.ipynb`, `b960_analyse_traffic_mit_gemini_chat.ipynb`



# 4 🛠️ Technologie-Stack

## 4.1 Kernbibliotheken
- **Machine Learning**: scikit-learn, pandas, numpy
- **Deep Learning**: Keras, TensorFlow
- **Datenvisualisierung**: matplotlib, plotly
- **Spezialisierte ML**: XGBoost, PyCaret
- **Zeitreihen**: Amazon Chronos
- **Modell-Export**: pypmml, nyoka (PMML-Format), joblib
- **Generative KI**: Google Gemini API
- **Webanwendungen**: Gradio

## 4.2 Entwicklungsumgebung
- **Laufzeit**: Python 3.11+, Java 8+ (für PMML)
- **IDE**: Google Colab, Jupyter Notebook/Lab



# 5 🗂️ Datensatz-Sammlung

Der Kurs verwendet verschiedene reale Datensätze für praktisches Lernen (unter `02_daten/`):

## 5.1 Tabellarische Daten (`02_daten/05_tabellen/`)
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

## 5.2 Text-Daten (`02_daten/01_text/`)
- **`maintenance_data.txt`** und **`maintenance_features.txt`** - Textdaten zur Maintenance-Analyse

## 5.3 Bild-Daten (`02_daten/02_bild/`)
- Bilddateien für Computer Vision-Aufgaben

## 5.4 Video-Daten (`02_daten/04_video/`)
- **`pexels_pixabay_people.mp4`** - Videoanalyse



# 6 ⚖️ Lizenz

Dieses Projekt steht unter der **MIT-Lizenz** (siehe `license`-Datei).      

**MIT License - Copyright (c) 2025 Ralf**      

Die Kursmaterialien können frei verwendet, modifiziert und weiterverbreitet werden.     

---

**Letzte Aktualisierung:** 29. April 2026     
**Version:** 1.1
