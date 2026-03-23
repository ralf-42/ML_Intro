---
layout: default
title: Deployment
nav_order: 4
has_children: true
description: "Von der Entwicklung zum produktiven Einsatz von ML-Anwendungen"
has_toc: true
---

# Deployment

Deployment beginnt im ML-Kontext nicht erst bei Docker oder Cloud-Hosting. Relevant wird es in dem Moment, in dem ein Modell wiederholbar laufen, mit echten Eingaben umgehen und außerhalb des Trainingsnotebooks nutzbar bleiben soll. Genau an dieser Schwelle zeigt sich, ob ein Projekt nur eine Auswertung ist oder bereits eine Anwendung.

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Einstieg

Die beiden Deployment-Dokumente decken im Kurs zwei typische Übergänge ab: den Weg von einem Notebook zu einer strukturierten Anwendung und den Sonderfall einer einfachen Web-Oberfläche mit Gradio.

- **[Gradio Deployment](https://ralf-42.github.io/ML_Intro/deployment/Gradio_Deployment.html)** – *Wie wird aus einem Modell eine kleine Web-App?* Gradio, Hugging Face Spaces und einfache Bereitstellungspfade.
- **[Produktionsreife Anwendung](https://ralf-42.github.io/ML_Intro/deployment/aus-entwicklung-ins-deployment.html)** – *Was passiert zwischen Notebook und produktiver Anwendung?* Strukturierung, Konfiguration, Tests und Go-Live-Vorbereitung.

---

## Modellbereitstellung

Nicht jedes ML-Projekt braucht einen vollwertigen Service. Häufig reicht zunächst ein gespeichertes Modell, das in einem Skript, einer internen Anwendung oder einer Demo wieder geladen werden kann. Der relevante Unterschied liegt darin, ob das Ergebnis reproduzierbar und wartbar bleibt.

| Thema | Beschreibung |
|-------|--------------|
| **joblib** | Serialisierung von scikit-learn Modellen (empfohlen) |
| **pickle** | Python-Standard für Objektspeicherung |
| **ONNX** | Framework-übergreifendes Modellformat |
| **TensorFlow SavedModel** | TensorFlow/Keras Modellformat |

---

## Pipelines

Pipelines sind im Deployment nicht nur Komfort, sondern Absicherung gegen inkonsistente Vorverarbeitung. Sobald Preprocessing und Modell voneinander getrennt laufen, steigen Fehler und Reproduktionsprobleme schnell.

| Konzept | Beschreibung |
|---------|--------------|
| **scikit-learn Pipeline** | Verkettung von Preprocessing und Modell |
| **ColumnTransformer** | Unterschiedliche Transformationen für verschiedene Spaltentypen |
| **MLflow** | Experiment Tracking und Model Registry |
| **DVC** | Versionierung von Daten und Modellen |

---

## Web Apps (Gradio)

Gradio eignet sich im Kurs vor allem für den Übergang von der Modelllogik zur vorzeigbaren Oberfläche. Das ist kein Ersatz für größere Produktarchitekturen, aber ein guter erster Deployment-Schritt.

| Thema | Beschreibung |
|-------|--------------|
| **[Gradio Deployment](https://ralf-42.github.io/ML_Intro/deployment/Gradio_Deployment.html)** | Web-Interfaces, Hugging Face Spaces, Docker |

**Weitere Web-Frameworks:**
- **Streamlit** - Datenbasierte Web-Apps
- **Flask/FastAPI** - REST APIs für ML-Modelle

---

## Cloud & MLOps

Mit wachsender Komplexität treten weitere Fragen hinzu: Modellversionierung, Experiment Tracking, Monitoring und Managed Services. Nicht jedes Projekt braucht diesen Stack sofort, aber die Begriffe sollten eingeordnet sein.

| Plattform | Beschreibung |
|-----------|--------------|
| **Hugging Face Spaces** | Kostenloses Hosting für ML-Demos |
| **AWS SageMaker** | Vollständige ML-Plattform |
| **Google Cloud AI** | Managed ML Services |
| **Azure ML** | Microsoft Cloud ML-Lösung |

---

**Version:** 1.0
**Stand:** März 2026
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.
