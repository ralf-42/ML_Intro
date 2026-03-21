---
layout: default
title: Deployment
nav_order: 4
has_children: true
description: "Von der Entwicklung zum produktiven Einsatz von ML-Anwendungen"
---

# Deployment

Vom Experiment zur Produktion — ML-Modelle produktiv einsetzen.

## Inhalte

- **[Gradio Deployment](https://ralf-42.github.io/ML_Intro/deployment/Gradio_Deployment.html)** - Web-Interfaces, Hugging Face Spaces, Docker
- **[Produktionsreife Anwendung](https://ralf-42.github.io/ML_Intro/deployment/aus-entwicklung-ins-deployment.html)** - Schritt-für-Schritt vom Notebook ins Deployment

---

## Model Persistence

Modelle speichern und in Produktionsumgebungen laden.

| Thema | Beschreibung |
|-------|--------------|
| **joblib** | Serialisierung von scikit-learn Modellen (empfohlen) |
| **pickle** | Python-Standard für Objektspeicherung |
| **ONNX** | Framework-übergreifendes Modellformat |
| **TensorFlow SavedModel** | TensorFlow/Keras Modellformat |

---

## Pipelines

Reproduzierbare und wartbare ML-Workflows.

| Konzept | Beschreibung |
|---------|--------------|
| **scikit-learn Pipeline** | Verkettung von Preprocessing und Modell |
| **ColumnTransformer** | Unterschiedliche Transformationen für verschiedene Spaltentypen |
| **MLflow** | Experiment Tracking und Model Registry |
| **DVC** | Versionierung von Daten und Modellen |

---

## Web Apps (Gradio)

Interaktive Web-Interfaces für ML-Modelle.

| Thema | Beschreibung |
|-------|--------------|
| **[Gradio Deployment](https://ralf-42.github.io/ML_Intro/deployment/Gradio_Deployment.html)** | Web-Interfaces, Hugging Face Spaces, Docker |

**Weitere Web-Frameworks:**
- **Streamlit** - Datenbasierte Web-Apps
- **Flask/FastAPI** - REST APIs für ML-Modelle

---

## Cloud & MLOps

| Plattform | Beschreibung |
|-----------|--------------|
| **Hugging Face Spaces** | Kostenloses Hosting für ML-Demos |
| **AWS SageMaker** | Vollständige ML-Plattform |
| **Google Cloud AI** | Managed ML Services |
| **Azure ML** | Microsoft Cloud ML-Lösung |
