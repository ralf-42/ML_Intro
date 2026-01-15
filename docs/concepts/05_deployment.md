---
layout: default
title: Deployment
parent: Konzepte
nav_order: 5
has_children: true
description: "ML-Modelle in Produktion bringen"
---

# Deployment

Vom Experiment zur Production - ML-Modelle produktiv einsetzen.

Die wichtigsten Aspekte des Deployments:

- **Model Persistence** - Modelle speichern und laden
- **Pipelines** - Reproduzierbare ML-Workflows
- **Web Apps** - Interaktive Interfaces mit Gradio
- **XAI** - Erklärbarkeit und Interpretierbarkeit

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
| [Gradio Deployment](deploy) | Web-Interfaces, Hugging Face Spaces, Docker |

**Weitere Web-Frameworks:**
- **Streamlit** - Datenbasierte Web-Apps
- **Flask/FastAPI** - REST APIs für ML-Modelle

---

## XAI (Explainability)

Erklärbarkeit und Interpretierbarkeit von Modellentscheidungen.

| Thema | Beschreibung |
|-------|--------------|
| [XAI - Explainable AI](08_xai) | SHAP, LIME, Feature Importance, Counterfactuals |

---

## Cloud & MLOps

| Plattform | Beschreibung |
|-----------|--------------|
| **Hugging Face Spaces** | Kostenloses Hosting für ML-Demos |
| **AWS SageMaker** | Vollständige ML-Plattform |
| **Google Cloud AI** | Managed ML Services |
| **Azure ML** | Microsoft Cloud ML-Lösung |

---

**Version:** 1.0    
**Stand:** Januar 2026    
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.     
