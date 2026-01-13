---
layout: default
title: Ensemble-Methoden
parent: Modeling
grand_parent: Konzepte
nav_order: 10
description: "Ensemble-Learning kombiniert mehrere Modelle zu leistungsfähigeren Vorhersagesystemen durch Bagging, Boosting und Stacking"
has_toc: true
---

# Ensemble-Methoden
{: .no_toc }

> **Ensemble-Learning kombiniert mehrere Machine-Learning-Modelle, um bessere Vorhersagen zu erzielen als einzelne Modelle.**
> Die wichtigsten Strategien sind Bagging (parallele, homogene Modelle), Boosting (sequentielle, homogene Modelle) und Stacking (parallele, heterogene Modelle).

---

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Übersicht der Ensemble-Strategien

Ensemble-Methoden nutzen die "Weisheit der Vielen" – die Kombination mehrerer Modelle führt oft zu besseren und robusteren Ergebnissen als jedes einzelne Modell.

```mermaid
flowchart TD
    subgraph Ensemble["Ensemble-Methoden"]
        direction TB
        
        subgraph Bagging["Bagging"]
            B1["Homogene Modelle"]
            B2["Paralleles Training"]
        end
        
        subgraph Boosting["Boosting"]
            BO1["Homogene Modelle"]
            BO2["Sequentielles Training"]
        end
        
        subgraph Stacking["Stacking"]
            S1["Heterogene Modelle"]
            S2["Paralleles Training"]
            S3["Voting"]
            S4["Meta-Learning"]
        end
    end
    
    style Bagging fill:#e8f5e9,stroke:#4caf50
    style Boosting fill:#e3f2fd,stroke:#2196f3
    style Stacking fill:#fff3e0,stroke:#ff9800
```

| Strategie | Modelltyp | Training | Beispiele |
|-----------|-----------|----------|-----------|
| **Bagging** | Homogen | Parallel | Random Forest |
| **Boosting** | Homogen | Sequentiell | XGBoost, AdaBoost |
| **Stacking** | Heterogen | Parallel | Voting, Meta-Learning |

---

## Bagging (Bootstrap Aggregating)

Beim **Bagging** werden mehrere gleichartige Modelle parallel trainiert und deren Vorhersagen kombiniert. Der Name steht für "Bootstrap Aggregating".

### Funktionsweise

```mermaid
flowchart LR
    D[("Originaldaten")] --> S1["Stichprobe 1"]
    D --> S2["Stichprobe 2"]
    D --> S3["Stichprobe 3"]
    D --> SN["..."]
    
    S1 --> M1["Modell 1"]
    S2 --> M2["Modell 2"]
    S3 --> M3["Modell 3"]
    SN --> MN["Modell N"]
    
    M1 --> A{{"Aggregation"}}
    M2 --> A
    M3 --> A
    MN --> A
    
    A --> P["Finale Vorhersage"]
    
    style D fill:#e3f2fd,stroke:#1976d2
    style A fill:#fff9c4,stroke:#fbc02d
    style P fill:#c8e6c9,stroke:#388e3c
```

**Die drei Schritte des Bagging:**

1. **Bootstrap-Sampling:** Es werden zufällige Stichproben aus den Daten gezogen (mit Zurücklegen)
2. **Paralleles Training:** Jedes Modell wird unabhängig auf seiner Stichprobe trainiert
3. **Aggregation:** Die Vorhersagen werden kombiniert:
   - **Klassifikation:** Mehrheitsentscheidung (Voting)
   - **Regression:** Mittelwert oder Median

> **Vorteile**
>
> - Reduziert Overfitting durch Varianzreduktion
> - Modelle können parallel trainiert werden
> - Robust gegenüber Ausreißern in einzelnen Stichproben

---

## Random Forest

**Random Forest** ist die bekannteste Bagging-Methode und besteht aus einem Ensemble von Entscheidungsbäumen.

### Besonderheiten

Random Forest erweitert das klassische Bagging um eine zusätzliche Zufallskomponente:

```mermaid
flowchart TD
    subgraph RF["Random Forest Prinzip"]
        D[("Datensatz")] --> B1["Bootstrap<br>Stichprobe 1"]
        D --> B2["Bootstrap<br>Stichprobe 2"]
        D --> BN["Bootstrap<br>Stichprobe N"]
        
        B1 --> T1["🌳 Baum 1<br>Zufällige Features"]
        B2 --> T2["🌳 Baum 2<br>Zufällige Features"]
        BN --> TN["🌳 Baum N<br>Zufällige Features"]
        
        T1 --> V1["Vorhersage 1"]
        T2 --> V2["Vorhersage 2"]
        TN --> VN["Vorhersage N"]
        
        V1 --> AGG{{"Aggregation<br>(Voting/Mittelwert)"}}
        V2 --> AGG
        VN --> AGG
        
        AGG --> FINAL["Finale Vorhersage"]
    end
    
    style D fill:#e8f5e9,stroke:#4caf50
    style AGG fill:#fff9c4,stroke:#fbc02d
    style FINAL fill:#c8e6c9,stroke:#388e3c
```

**Was Random Forest "zufällig" macht:**

| Komponente | Zufälligkeit |
|------------|--------------|
| **Datensätze** | Jeder Baum erhält eine andere Bootstrap-Stichprobe |
| **Features** | An jedem Knoten wird nur eine zufällige Teilmenge der Features für den Split betrachtet |
| **Splits** | Die Auswahl des besten Splits erfolgt nur aus den zufällig gewählten Features |

### Vorteile von Random Forest

- Kann sowohl für **Klassifikation** als auch **Regression** verwendet werden
- Funktioniert mit **kategorialen und numerischen** Features
- Liefert automatisch **Feature Importance**
- Robust gegenüber **Overfitting**
- Benötigt wenig **Hyperparameter-Tuning**

---

## Boosting

Beim **Boosting** werden Modelle sequentiell trainiert, wobei jedes neue Modell versucht, die Fehler der vorherigen Modelle zu korrigieren.

### Funktionsweise

```mermaid
flowchart LR
    subgraph Seq["Sequentielles Lernen"]
        D[("Daten")] --> M1["Modell 1"]
        M1 --> E1["Fehler<br>analysieren"]
        E1 --> |"Gewichtung<br>anpassen"| M2["Modell 2"]
        M2 --> E2["Fehler<br>analysieren"]
        E2 --> |"Gewichtung<br>anpassen"| M3["Modell 3"]
        M3 --> EN["..."]
    end
    
    M1 --> K{{"Kombination"}}
    M2 --> K
    M3 --> K
    
    K --> P["Finale<br>Vorhersage"]
    
    style D fill:#e3f2fd,stroke:#1976d2
    style K fill:#fff9c4,stroke:#fbc02d
    style P fill:#c8e6c9,stroke:#388e3c
```

**Das Boosting-Prinzip:**

1. **Erstes Modell** macht erste Vorhersagen (oft noch ungenau)
2. **Fehleranalyse:** Falsch klassifizierte Datenpunkte werden identifiziert
3. **Gewichtung:** Schwer zu klassifizierende Muster erhalten höhere Gewichte
4. **Nächstes Modell** fokussiert sich auf die schwierigen Fälle
5. **Finale Vorhersage** kombiniert alle Modellbeiträge

> **Kernidee**
>
> Muster, die noch nicht gut klassifiziert werden, bekommen im nächsten Durchlauf ein **höheres Gewicht**. Bereits erkannte Muster bekommen ein **geringeres Gewicht**. So spezialisiert sich jedes neue Modell auf die verbleibenden Fehler.

### Vergleich: Bagging vs. Boosting

| Aspekt | Bagging | Boosting |
|--------|---------|----------|
| **Training** | Parallel | Sequentiell |
| **Fokus** | Varianzreduktion | Bias-Reduktion |
| **Fehlerbehandlung** | Gleichmäßig | Gewichtet (schwierige Fälle) |
| **Overfitting-Risiko** | Geringer | Höher (aber kontrollierbar) |
| **Rechenzeit** | Parallelisierbar | Nicht parallelisierbar |

---

## Weiterführende Ressourcen

| Ressource | Beschreibung |
|-----------|--------------|
| [KNIME Bagging & Boosting](https://www.knime.com/) | Visuelle Erklärung der Ensemble-Methoden |
| [StatQuest Random Forest](https://www.youtube.com/c/joshstarmer) | Video-Tutorial zu Random Forest |

---

## Zusammenfassung

```mermaid
mindmap
  root((Ensemble))
    Bagging
      Parallel
      Bootstrap-Samples
      Random Forest
      Varianzreduktion
    Boosting
      Sequentiell
      Fehlergewichtung
      XGBoost
      Bias-Reduktion
    Stacking
      Heterogen
      Voting
      Meta-Learning
```

**Die wichtigsten Erkenntnisse:**

- **Ensemble-Methoden** kombinieren mehrere Modelle für bessere Vorhersagen
- **Bagging** (z.B. Random Forest) reduziert Varianz durch parallele, unabhängige Modelle
- **Boosting** (z.B. XGBoost) reduziert Bias durch sequentielles Lernen aus Fehlern
- **Stacking** kombiniert verschiedenartige Modelle durch Voting oder Meta-Learning
- Die Wahl der Methode hängt vom Problem und den verfügbaren Ressourcen ab


---

**Version:** 1.0    
**Stand:** Januar 2026    
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.    
