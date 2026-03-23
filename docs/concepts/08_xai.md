---
layout: default
title: XAI
parent: Konzepte
nav_order: 8
has_children: true
has_toc: true
description: "Explainable AI - Interpretierbarkeit von ML-Modellen"
---

# XAI - Explainable AI
{: .no_toc }

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---


Methoden zur Erklärbarkeit und Interpretierbarkeit von Machine Learning Modellen.

## XAI Methoden & Frameworks

**[Methoden & Frameworks](xai_erklaerbare_ki)** - Umfassender Guide zu XAI-Techniken und -Tools:

- **SHAP (SHapley Additive exPlanations)** - Theoretisch fundierte Feature-Attribution
- **LIME (Local Interpretable Model-agnostic Explanations)** - Lokale Erklärungen durch Surrogate-Modelle
- **ELI5** - Einfache Permutation Importance
- **InterpretML** - Microsoft Framework mit interaktiven Dashboards
- **Feature Importance** - Tree-basierte Modelle
- **Ceteris Paribus Analysen** - "What-if"-Szenarien
- **Framework-Vergleich** und Best Practices

## Weitere XAI-Themen

### Model-agnostic Methods

- SHAP für verschiedene Modelltypen
- LIME für Tabular, Text und Image Data
- Partial Dependence Plots
- Accumulated Local Effects

### Model-specific Methods

- Decision Tree Visualization (dtreeviz)
- Linear Model Koeffizienten
- Neural Network Activation Maps

### Fairness & Bias

- Bias Detection
- Fairness Metrics
- Debiasing Techniques

### Debugging ML Models

- Error Analysis
- Feature Attribution
- Adversarial Examples

## Abgrenzung zu verwandten Dokumenten

| Dokument | Frage |
|---|---|
| [Grundlagen](./01_grundlagen.html) | Welche ML-Grundbegriffe und Lernparadigmen sollte XAI voraussetzen? |
| [Prepare](./02_prepare.html) | Wie beeinflusst Vorverarbeitung die spätere Interpretierbarkeit? |
| [Modeling](./03_modeling.html) | Welche Modellarten stellen unterschiedliche Anforderungen an Erklärbarkeit? |
| [Evaluate](./04_evaluate.html) | Wie ergänzt XAI klassische Bewertungsmetriken durch inhaltliche Nachvollziehbarkeit? |

---

**Version:** 1.0
**Stand:** März 2026
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.
