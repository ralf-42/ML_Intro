---
layout: default
title: AutoML
parent: Modeling
grand_parent: Konzepte
nav_order: 15
description: "Automatisiertes Machine Learning (AutoML) - Workflow-Automatisierung von Datenvorbereitung bis Modellauswahl"
has_toc: true
---

# AutoML
{: .no_toc }

> **Automatisiertes Machine Learning (AutoML) automatisiert den gesamten ML-Workflow – von der Datenvorbereitung über Feature Engineering bis zur Modellauswahl und Hyperparameter-Optimierung.**

---

## Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Was ist AutoML?

AutoML (Automated Machine Learning) ist ein Bereich der künstlichen Intelligenz, der darauf abzielt, den Prozess des maschinellen Lernens auf reale Probleme zu automatisieren.

```mermaid
flowchart LR
    subgraph "Traditioneller ML-Workflow"
        A[Daten] --> B[Manuelle<br/>Vorbereitung]
        B --> C[Feature<br/>Engineering]
        C --> D[Modell-<br/>auswahl]
        D --> E[Hyper-<br/>parameter]
        E --> F[Evaluation]
    end
    
    subgraph "AutoML"
        G[Daten] --> H[🤖 Automatisiert]
        H --> I[Bestes<br/>Modell]
    end
    
    style H fill:#4CAF50,color:#fff
```

---

## Kernfunktionen

AutoML-Systeme übernehmen automatisch die zeitaufwändigsten Schritte des ML-Prozesses:

| Funktion | Beschreibung |
|----------|-------------|
| **Automatische Datenvorbereitung** | Behandlung fehlender Daten, Kategorien kodieren, Transformationen auswählen |
| **Feature Engineering** | Automatische Identifikation und Erstellung wichtiger Merkmale |
| **Algorithmen-Auswahl** | Auswahl der am besten geeigneten ML-Algorithmen für das Problem |
| **Hyperparameter-Tuning** | Automatische Optimierung der Modelleinstellungen |
| **Kreuzvalidierung** | Gründliche Validierung zur Vermeidung von Overfitting |

```mermaid
flowchart TD
    subgraph "AutoML Pipeline"
        A[📊 Rohdaten] --> B[Datenaufbereitung]
        B --> C[Feature Engineering]
        C --> D[Modellvergleich]
        D --> E[Hyperparameter-Tuning]
        E --> F[Ensemble & Stacking]
        F --> G[🎯 Bestes Modell]
    end
    
    B -.-> B1[Missing Values<br/>Encoding<br/>Scaling]
    D -.-> D1[Random Forest<br/>XGBoost<br/>LightGBM<br/>...]
    
    style G fill:#2196F3,color:#fff
```

---

## AutoML mit PyCaret

[PyCaret](https://pycaret.org/) ist eine Open-Source Python-Bibliothek für Low-Code Machine Learning. Sie automatisiert ML-Workflows und ermöglicht schnelles Experimentieren.


---



## AutoML-Plattformen im Vergleich

| Plattform | Open Source | Stärken | Einsatz |
|-----------|-------------|---------|---------|
| **PyCaret** | ✅ | Low-Code, schnell, umfangreich | Prototyping, Experimente |
| **Auto-sklearn** | ✅ | Scikit-learn basiert, robust | Forschung, Produktion |
| **H2O AutoML** | ✅ | Skalierbar, Enterprise-ready | Big Data, Unternehmen |
| **Google AutoML** | ❌ | Cloud-basiert, einfach | Cloud-native Projekte |
| **Azure AutoML** | ❌ | Microsoft-Integration | Enterprise, Azure-Nutzer |

---

## Vorteile und Grenzen

### Vorteile

- **Zeitersparnis:** Automatisierung repetitiver Aufgaben
- **Demokratisierung:** ML auch ohne tiefes Expertenwissen nutzbar
- **Konsistenz:** Standardisierte, reproduzierbare Pipelines
- **Exploration:** Schneller Überblick über geeignete Modelle

### Grenzen

- **Black-Box-Charakter:** Weniger Kontrolle über Entscheidungen
- **Domänenwissen:** Ersetzt nicht das Verständnis des Problems
- **Spezialfälle:** Komplexe, individuelle Anforderungen oft schwer abbildbar
- **Rechenaufwand:** Kann ressourcenintensiv sein

```mermaid
flowchart TD
    subgraph "Wann AutoML nutzen?"
        A{Projektsituation} --> B[Schneller Prototyp<br/>→ ✅ AutoML]
        A --> C[Baseline-Modell<br/>→ ✅ AutoML]
        A --> D[Komplexe Pipeline<br/>→ ⚠️ Hybrid]
        A --> E[Maximale Kontrolle<br/>→ ❌ Manuell]
    end
    
    style B fill:#4CAF50,color:#fff
    style C fill:#4CAF50,color:#fff
    style D fill:#FFC107
    style E fill:#f44336,color:#fff
```

---

## Best Practices

1. **Datenqualität prüfen:** AutoML ersetzt keine Datenexploration
2. **Baseline etablieren:** Einfaches Modell zum Vergleich erstellen
3. **Ergebnisse verstehen:** Nicht blind dem besten Modell vertrauen
4. **Reproduzierbarkeit:** Immer `session_id` setzen
5. **Iteration:** AutoML als Startpunkt, dann manuell optimieren


---

**Version:** 1.0       
**Stand:** Januar 2026     
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.     
