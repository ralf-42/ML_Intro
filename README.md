# Machine Learning Kurs

[![Last Updated](https://img.shields.io/badge/Last%20Updated-2026--01--10-blue)](./README.md)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-brightgreen)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

Dieses Repository enthält Lehrmaterialien für den Machine Learning-Kurs.

## 📖 Dokumentation

Die vollständige Kursdokumentation ist verfügbar unter:  **[https://ralf-42.github.io/ML_Intro](https://ralf-42.github.io/ML_Intro)**

Die Web-Dokumentation bietet:
- Interaktive Navigation durch alle Konzepte und Frameworks
- Mermaid-Diagramme zur Visualisierung von ML-Workflows
- Strukturierte Übersicht über Deployment, Regulatorisches und Ressourcen

# 1 📚 Kursübersicht
Dieser Kurs bietet einen strukturierten Lernpfad von grundlegenden Machine Learning-Konzepten bis hin zu fortgeschrittenen Deep Learning-Anwendungen und umfasst sowohl überwachte als auch unüberwachte Lerntechniken, neuronale Netze, Ensemble-Methoden und moderne generative KI-Anwendungen.

# 2🎓 Lernziele

Nach Abschluss dieses Kurses können Studierende:

- ✅ Grundlegende ML-Konzepte und Workflows verstehen
- ✅ Überwachte und unüberwachte Lernalgorithmen implementieren
- ✅ Neuronale Netze mit Keras/TensorFlow erstellen und optimieren
- ✅ Ensemble-Methoden und fortgeschrittene ML-Techniken anwenden
- ✅ Hyperparameter-Tuning und Modellvalidierung durchführen
- ✅ Mit spezialisierten Anwendungen arbeiten (CV, NLP, Zeitreihen)
- ✅ Einsetzbare ML-Anwendungen mit modernen Tools erstellen
- ✅ Machine Learning-Modelle interpretieren und erklären

# 3 📚 Repository-Struktur

 

## 3.1 Hauptverzeichnisse

- **`01_notebook/`** - Jupyter Notebooks mit Kursinhalten (10 Module)
- **`02_daten/`** - Datensätze für praktische Übungen
- **`03_skript/`** - Präsentationsmaterialien und Skripte
- **`04_model/`** - Trainierte Modelle

## 3.2 Kursmodule in `01_notebook/`

### 3.2.1 🟢 Modul 00: Allgemeine Konzepte
**Pfad**: `01_notebook/00_general/`
- Grundlegende ML-Konzepte und pandas-Grundlagen
- Datensatzbehandlung und -exploration
- Beispiele: `b000_launch.ipynb`, `b020_pandas_basics.ipynb`, `b040_datasets.ipynb`

### 3.2.2 🔵 Modul 01: Überwachtes Lernen
**Pfad**: `01_notebook/01_supervised/`
- Entscheidungsbäume (Titanic-Datensatz)
- Lineare Regression (MPG-Vorhersage)
- Random Forests (Diamantpreisvorhersage)
- Beispiele: `b110_sl_dt_titanic.ipynb`, `b120_sl_lr_mpg.ipynb`, `b130_sl_rf_diamonds_inverse.ipynb`

### 3.2.3 🟣 Modul 02: Unüberwachtes Lernen
**Pfad**: `01_notebook/02_unsupervised/`
- K-means und DBSCAN-Clustering
- Isolation Forest für Anomalieerkennung
- PCA für Dimensionsreduktion
- Assoziationsregeln (Apriori-Algorithmus)
- Beispiele: `b200_ul_kmeans_dbscan_location.ipynb`, `b240_ul_pca_special.ipynb`

### 3.2.4 🟡 Modul 03: Neuronale Netze
**Pfad**: `01_notebook/03_network/`
- Multi-Layer Perceptron (MLP) Implementierungen
- Keras/TensorFlow neuronale Netze
- Anwendungen auf Krebs- und Diamant-Datensätzen
- Beispiele: `b310_nn_mlp_cancer.ipynb`, `b320_nn_keras_cancer.ipynb`

### 3.2.5 🟠 Modul 04: Ensemble-Methoden
**Pfad**: `01_notebook/04_ensemble/`
- XGBoost-Implementierung
- Stacking-Ensemble-Techniken
- Beispiele: `b410_xg_cancer.ipynb`, `b430_stacking_titanic.ipynb`

### 3.2.6 ⚪ Modul 05: Modell-Tuning & Validierung
**Pfad**: `01_notebook/05_tuning/`
- Kreuzvalidierungstechniken
- Hyperparameter-Optimierung (Grid Search, Random Search)
- ROC-AUC-Analyse und Schwellenwertoptimierung
- AutoML mit PyCaret
- Lernkurven und Validierungsstrategien
- Beispiele: `b510_cv_dt_titanic.ipynb`, `b530_gridsearch_nn_mlp_cancer.ipynb`

### 3.2.7 🔴 Modul 06: ML-Workflows
**Pfad**: `01_notebook/06_workflow/`
- Scikit-learn Pipelines
- End-to-End ML-Workflow-Automatisierung
- Beispiel: `b610_pipeline_dt_diamonds.ipynb`

### 3.2.8 🟤 Modul 07: Spezialisierte Anwendungen
**Pfad**: `01_notebook/07_special/`
- Computer Vision (MNIST, YOLO)
- Natural Language Processing (Spam-Erkennung)
- Zeitreihenanalyse (Wettervorhersage)
- Autoencoder für Dimensionsreduktion
- Beispiele: `b710_vision_keras_mnist.ipynb`, `b720_nlp_keras_spam.ipynb`

### 3.2.9 ⚫ Modul 08: Generative KI
**Pfad**: `01_notebook/08_genai/`
- LangChain-Integration mit OpenAI
- PDF-Zusammenfassung mit LLMs
- Interaktive Chat-Anwendungen mit Gradio
- Beispiele: `b800_simple_chat_langchain_openai_gradio.ipynb`, `b810_pdf_llm_summary.ipynb`

### 3.2.10 🔵 Modul 09: Vielfältige Anwendungen
**Pfad**: `01_notebook/09_diverse/`
- Erklärbare KI (XAI) Techniken
- Gradio-Webanwendungen
- Modellpersistierung (Speichern/Laden)
- Business Intelligence mit Gemini AI
- Beispiele: `b900_xai_titanic.ipynb`, `b910_data_app_gradio_diamonds.ipynb`



# 4 🛠️ Technologie-Stack

## 4.1 Kernbibliotheken
- **Machine Learning**: scikit-learn, pandas, numpy
- **Deep Learning**: Keras, TensorFlow
- **Datenvisualisierung**: matplotlib, plotly
- **Spezialisierte ML**: XGBoost, PyCaret
- **Generative KI**: google.colab ai
- **Webanwendungen**: Gradio
- **Datenverarbeitung**: pandas, numpy

## 4.2 Entwicklungsumgebung
- **Laufzeit**: Python 3.11+
- **IDE**: Google Colab, Jupyter Notebook



# 5 🗂️ Datensatz-Sammlung

Der Kurs verwendet verschiedene reale Datensätze für praktisches Lernen (unter `02_daten/`):

## 5.1 Tabellarische Daten (`02_daten/05_tabellen/`)
- **`titanic.csv`** - Überlebensvorhersage (Klassifikation)
- **`diamonds.csv`** - Preisvorhersage (Regression)
- **`breast_cancer_wisconsin.csv`** - Medizinische Diagnose (Klassifikation)
- **`auto_mpg.csv`** - Kraftstoffeffizienzvorhersage (Regression)
- **`ccpp.csv`** - Combined Cycle Power Plant (Regression)
- **`wa_fn_usec__telco_customer_churn.csv`** - Kundenabwanderung
- Und viele weitere spezialisierte Datensätze

## 5.2 Text-Daten (`02_daten/01_text/`)
- **`smsspamcollection`** - SMS Spam-Erkennung (NLP)

## 5.3 Bild-Daten (`02_daten/02_bild/`)
- Bilddateien für Computer Vision-Aufgaben

## 5.4 Video-Daten (`02_daten/04_video/`)
- **`pexels_pixabay_people.mp4`** - Videoanalyse



# 6 ⚖️ Lizenz

Dieses Projekt steht unter der **MIT-Lizenz** (siehe `license`-Datei).      

**MIT License - Copyright (c) 2025 Ralf**      

Die Kursmaterialien können frei verwendet, modifiziert und weiterverbreitet werden.     

---

**Letzte Aktualisierung:** 10. Januar 2026     
**Version:** 1.0         