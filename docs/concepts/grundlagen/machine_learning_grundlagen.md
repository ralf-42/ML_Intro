---
layout: default
title: Intro
parent: Grundlagen
grand_parent: Konzepte
nav_order: 1
description: "Einführung in die fundamentalen Konzepte des maschinellen Lernens: Lernparadigmen, Aufgabentypen und praktische Anwendungsbeispiele"
has_toc: true
---

# Machine Learning Grundlagen
{: .no_toc }

> **Dieses Kapitel vermittelt die fundamentalen Konzepte des maschinellen Lernens.**        
> Sie lernen, was Machine Learning ist, welche Lernparadigmen existieren und wie sich verschiedene ML-Aufgaben unterscheiden.

---

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Was ist Lernen?

Bevor wir uns dem maschinellen Lernen widmen, lohnt sich ein Blick auf das Konzept des Lernens selbst:

**Lernen** ist ein Prozess, bei dem:
- **Wissen**, **Fähigkeiten**, **Verhaltensweisen** oder **Einstellungen** erworben, verändert oder verstärkt werden
- Informationen aufgenommen, verarbeitet und behalten werden
- Anpassung an neue Situationen ermöglicht wird

Lernen kann auf verschiedene Weisen erfolgen:

```mermaid
mindmap
  root((Lernen))
    Erfahrung
      Trial & Error
      Beobachtung
    Unterricht
      Strukturierte Anleitung
      Feedback
    Training
      Wiederholung
      Optimierung
```

Diese menschlichen Lernprinzipien bilden die Grundlage für das maschinelle Lernen – übertragen auf Computer und Algorithmen.

---

## Was ist Machine Learning?

**Machine Learning** (maschinelles Lernen) bezeichnet einen Bereich der künstlichen Intelligenz, der es Computern ermöglicht:

1. **Automatisch aus Informationen und Erfahrung zu lernen**
2. **Ihre Leistung bei bestimmten Aufgaben kontinuierlich zu verbessern**
3. **Muster und Zusammenhänge zu erkennen** und diese in Vorhersagen, Entscheidungen oder Aktionen umzusetzen

```mermaid
flowchart LR
    subgraph input["Eingabe"]
        D[("Daten")]
    end
    
    subgraph process["Verarbeitung"]
        A["Algorithmus"]
        M["Modell"]
        D --> A
        A --> |"Training"| M
    end
    
    subgraph output["Ausgabe"]
        P["Vorhersagen"]
        E["Entscheidungen"]
        K["Aktionen"]
    end
    
    M --> P
    M --> E
    M --> K
    
    style D fill:#e3f2fd
    style A fill:#fff9c4
    style M fill:#c8e6c9
    style P fill:#f3e5f5
    style E fill:#f3e5f5
    style K fill:#f3e5f5
```

### Kernkonzepte

| Begriff | Beschreibung |
|---------|--------------|
| **Algorithmus** | Präzise, wohldefinierte Prozedur zur Lösung einer Aufgabe |
| **Modell** | Das Ergebnis des Lernprozesses – repräsentiert erkannte Muster |
| **Training** | Der Prozess, bei dem ein Algorithmus aus Daten lernt |
| **Vorhersage** | Anwendung des trainierten Modells auf neue Daten |

### Anwendungsgebiete

Machine Learning ist heute in vielen Bereichen verbreitet:

- **Bilderkennung**: Gesichtserkennung, medizinische Bildanalyse, autonomes Fahren
- **Spracherkennung**: Sprachassistenten, automatische Transkription
- **Datenanalyse**: Kundensegmentierung, Trendanalyse
- **Prognose**: Wettervorhersage, Aktienkurse, Nachfrageplanung
- **Automatisierte Entscheidungsfindung**: Kreditvergabe, Empfehlungssysteme

---

## Wie funktioniert Machine Learning?

Der grundlegende ML-Prozess folgt einem klaren Muster:

```mermaid
flowchart TB
    subgraph phase1["1. Datensammlung"]
        D1[("Rohdaten")]
        D2["Aufbereitete<br/>Daten"]
        D1 --> D2
    end
    
    subgraph phase2["2. Training"]
        D2 --> ALG["ML-Algorithmus"]
        ALG --> |"Lernt Muster"| MOD["Trainiertes<br/>Modell"]
    end
    
    subgraph phase3["3. Anwendung"]
        NEW[("Neue Daten")]
        NEW --> MOD
        MOD --> PRED["Vorhersage"]
    end
    
    style D1 fill:#e3f2fd
    style D2 fill:#bbdefb
    style ALG fill:#fff9c4
    style MOD fill:#c8e6c9
    style NEW fill:#e3f2fd
    style PRED fill:#f3e5f5
```

### Der Lernprozess im Detail

1. **Daten sammeln**: Relevante Beispieldaten für das Problem zusammentragen
2. **Daten aufbereiten**: Bereinigen, transformieren und für das Training vorbereiten
3. **Algorithmus wählen**: Passenden ML-Algorithmus für die Aufgabe auswählen
4. **Modell trainieren**: Algorithmus lernt Muster aus den Trainingsdaten
5. **Modell evaluieren**: Leistung auf ungesehenen Testdaten prüfen
6. **Modell anwenden**: Vorhersagen für neue Daten erstellen

---

## Lernparadigmen

Die Art der verfügbaren Daten bestimmt, welches Lernparadigma angewendet werden kann. Es gibt drei grundlegende Ansätze:

```mermaid
flowchart TB
    Q1{"Gibt es Daten<br/>zur Aufgabe?"}
    Q1 --> |"Nein"| NO["Keine ML-Lösung<br/>möglich"]
    Q1 --> |"Ja"| Q2{"Gibt es bekannte<br/>Lösungen zu<br/>den Daten?"}
    
    Q2 --> |"Ja"| SL["🎯 Supervised<br/>Learning"]
    Q2 --> |"Nein"| Q3{"Können Daten<br/>durch Simulation<br/>oder Experimente<br/>gewonnen werden?"}
    
    Q3 --> |"Ja"| RL["🎮 Reinforcement<br/>Learning"]
    Q3 --> |"Nein"| UL["🔍 Unsupervised<br/>Learning"]
    
    style SL fill:#c8e6c9
    style UL fill:#bbdefb
    style RL fill:#fff9c4
    style NO fill:#ffcdd2
```

### Übersicht der Lernparadigmen

| Paradigma | Daten | Lernziel | Typische Anwendung |
|-----------|-------|----------|-------------------|
| **Supervised Learning** | Daten mit bekannten Lösungen (Labels) | Vorhersage für neue Daten | Spam-Erkennung, Preisvorhersage |
| **Unsupervised Learning** | Daten ohne Labels | Strukturen und Muster entdecken | Kundensegmentierung, Anomalieerkennung |
| **Reinforcement Learning** | Interaktion mit Umgebung | Optimale Strategie lernen | Spielstrategien, Robotersteuerung |

---

## Lernparadigmen und Aufgabentypen

Jedes Lernparadigma umfasst verschiedene Aufgabentypen:

```mermaid
flowchart TB
    subgraph SL["Supervised Learning"]
        direction TB
        SL1["Klassifizierung"]
        SL2["Regression"]
    end
    
    subgraph UL["Unsupervised Learning"]
        direction TB
        UL1["Clustering"]
        UL2["Anomalieerkennung"]
        UL3["Dimensionsreduktion"]
        UL4["Assoziationsanalyse"]
    end
    
    subgraph RL["Reinforcement Learning"]
        direction TB
        RL1["Sequenzmodellierung"]
        RL2["Strategieoptimierung"]
    end
    
    style SL fill:#c8e6c9
    style UL fill:#bbdefb
    style RL fill:#fff9c4
```

---

## Supervised Learning (Überwachtes Lernen)

Beim **Supervised Learning** werden Modelle mit **gelabelten Daten** trainiert – also Daten, bei denen die richtige Antwort bekannt ist. Das Modell lernt, die Beziehung zwischen Eingabe (Features) und Ausgabe (Label/Target) zu erkennen.

### Die zwei Hauptaufgaben

```mermaid
flowchart LR
    subgraph SL["Supervised Learning"]
        direction TB
        IN["Features<br/>(Eingabe)"]
        
        subgraph tasks["Aufgabentypen"]
            CL["Klassifizierung<br/>📊"]
            RG["Regression<br/>📈"]
        end
        
        OUT_CL["Kategorie<br/>(diskret)"]
        OUT_RG["Zahlenwert<br/>(stetig)"]
        
        IN --> CL
        IN --> RG
        CL --> OUT_CL
        RG --> OUT_RG
    end
    
    style CL fill:#c8e6c9
    style RG fill:#bbdefb
    style OUT_CL fill:#e8f5e9
    style OUT_RG fill:#e3f2fd
```

### Klassifizierung

Ein **Klassifizierungsmodell** sagt **kategoriale Werte** voraus – es ordnet Datenpunkte einer von mehreren vordefinierten Klassen zu.

**Beispiele:**

| Anwendung | Features (Eingabe) | Klassen (Ausgabe) |
|-----------|-------------------|-------------------|
| **Spam-Erkennung** | E-Mail-Text, Absender, Betreff | Spam / Kein Spam |
| **Medizinische Diagnose** | Symptome, Laborwerte | Gesund / Krank |
| **Fahrprüfung** | Übungsstunden, Theorie-Tests | Bestanden / Nicht bestanden |
| **Bilderkennung** | Pixel-Werte | Katze / Hund / Vogel / ... |

```mermaid
flowchart LR
    subgraph beispiel["Beispiel: Spam-Klassifikation"]
        EMAIL["📧 E-Mail"]
        F1["Absender bekannt?"]
        F2["Verdächtige Links?"]
        F3["Typische Spam-Wörter?"]
        MODEL["🤖 Klassifikator"]
        
        EMAIL --> F1
        EMAIL --> F2
        EMAIL --> F3
        F1 --> MODEL
        F2 --> MODEL
        F3 --> MODEL
        
        MODEL --> SPAM["🚫 Spam"]
        MODEL --> OK["✅ Kein Spam"]
    end
    
    style SPAM fill:#ffcdd2
    style OK fill:#c8e6c9
```

### Regression

Ein **Regressionsmodell** sagt **stetige, numerische Werte** voraus.

**Beispiele:**

| Anwendung | Features (Eingabe) | Ausgabe (numerisch) |
|-----------|-------------------|---------------------|
| **Immobilienbewertung** | Lage, Größe, Baujahr, Ausstattung | Preis in € |
| **Temperaturvorhersage** | Historische Daten, Luftdruck, Jahreszeit | Temperatur in °C |
| **Umsatzprognose** | Vergangene Verkäufe, Marketing, Saison | Umsatz in € |
| **Speiseeis-Konsum** | Außentemperatur, Wochentag | Absatzmenge |

```mermaid
flowchart LR
    subgraph beispiel["Beispiel: Immobilienpreis"]
        HOUSE["🏠 Immobilie"]
        F1["Wohnfläche: 120m²"]
        F2["Baujahr: 2010"]
        F3["Lage: Stadtzentrum"]
        F4["Zimmer: 4"]
        MODEL["🤖 Regressor"]
        
        HOUSE --> F1
        HOUSE --> F2
        HOUSE --> F3
        HOUSE --> F4
        F1 --> MODEL
        F2 --> MODEL
        F3 --> MODEL
        F4 --> MODEL
        
        MODEL --> PRICE["💰 385.000 €"]
    end
    
    style PRICE fill:#c8e6c9
```

### Vergleich: Klassifizierung vs. Regression

| Aspekt | Klassifizierung | Regression |
|--------|----------------|------------|
| **Ausgabewert** | Kategorie (diskret) | Zahl (stetig) |
| **Beispiel-Frage** | "Ist es Spam?" | "Wie viel kostet es?" |
| **Anzahl möglicher Ausgaben** | Endlich viele Klassen | Unendlich viele Werte |
| **Typische Metriken** | Accuracy, Precision, Recall, F1 | MSE, RMSE, R² |
| **Beispiel-Algorithmen** | Logistische Regression, Decision Tree, Random Forest | Lineare Regression, Random Forest, XGBoost |

---

## Unsupervised Learning (Unüberwachtes Lernen)

Beim **Unsupervised Learning** arbeiten wir mit **ungelabelten Daten** – die "richtigen Antworten" sind nicht bekannt. Das Ziel ist es, **versteckte Strukturen und Muster** in den Daten zu entdecken.

```mermaid
flowchart TB
    subgraph UL["Unsupervised Learning"]
        direction LR
        
        subgraph CL["Clustering"]
            CL_DESC["Ähnliche Objekte<br/>gruppieren"]
            CL_EX["Kundensegmente,<br/>Dokumentgruppen"]
        end
        
        subgraph AN["Anomalieerkennung"]
            AN_DESC["Untypische<br/>Datenpunkte finden"]
            AN_EX["Betrugserkennung,<br/>Defekte Produkte"]
        end
        
        subgraph AS["Assoziationsanalyse"]
            AS_DESC["Zusammenhänge<br/>entdecken"]
            AS_EX["Warenkorbanalyse,<br/>Empfehlungen"]
        end
    end
    
    style CL fill:#bbdefb
    style AN fill:#fff9c4
    style AS fill:#f3e5f5
```

### Clustering (Segmentierung)

**Clustering** ist ein Verfahren zur Entdeckung von **Ähnlichkeitsstrukturen** in Daten. Die gefundenen Gruppen von "ähnlichen" Objekten werden als **Cluster** bezeichnet.

```mermaid
flowchart LR
    subgraph before["Vor dem Clustering"]
        D1[("🔵🔴🔵🔴<br/>🔵🔴🔵🔴<br/>Ungeordnete<br/>Datenpunkte")]
    end
    
    subgraph process["Clustering"]
        ALG["Clustering-<br/>Algorithmus"]
    end
    
    subgraph after["Nach dem Clustering"]
        C1["🔵🔵🔵<br/>Cluster 1"]
        C2["🔴🔴🔴<br/>Cluster 2"]
    end
    
    D1 --> ALG
    ALG --> C1
    ALG --> C2
    
    style D1 fill:#e3f2fd
    style C1 fill:#bbdefb
    style C2 fill:#ffcdd2
```

**Anwendungsbeispiele:**
- **Kundensegmentierung**: Gruppierung von Kunden nach Kaufverhalten
- **Dokumenten-Clustering**: Thematische Sortierung von Texten
- **Bildkompression**: Reduktion von Farbpaletten

### Anomalieerkennung

Die **Anomalieerkennung** identifiziert Datensätze, die für die gesamte Datenbasis **untypisch** sind.

```mermaid
flowchart TB
    subgraph data["Datenpunkte"]
        NORMAL["⚪⚪⚪⚪⚪⚪⚪⚪⚪<br/>Normale Datenpunkte"]
        ANOMALY["🔴<br/>Anomalie"]
    end
    
    subgraph result["Erkennung"]
        AN_ALG["Anomalie-<br/>Algorithmus"]
        NORMAL --> |"unauffällig"| AN_ALG
        ANOMALY --> |"auffällig!"| AN_ALG
        AN_ALG --> ALERT["⚠️ Warnung"]
    end
    
    style NORMAL fill:#e8f5e9
    style ANOMALY fill:#ffcdd2
    style ALERT fill:#fff9c4
```

**Anwendungsbeispiele:**
- **Betrugserkennung**: Ungewöhnliche Kreditkartentransaktionen
- **Qualitätskontrolle**: Defekte Produkte in der Fertigung
- **Netzwerksicherheit**: Verdächtige Aktivitäten erkennen
- **Medizin**: Abnormale Messwerte identifizieren

### Assoziationsanalyse

Die **Assoziationsanalyse** dient dem Auffinden von **Zusammenhängen** in transaktionsbasierten Daten. Die Ergebnisse werden als **Assoziationsregeln** dargestellt.

```mermaid
flowchart LR
    subgraph transactions["Transaktionen"]
        T1["🛒 Brot, Butter, Milch"]
        T2["🛒 Brot, Butter"]
        T3["🛒 Brot, Milch, Eier"]
        T4["🛒 Brot, Butter, Milch, Eier"]
    end
    
    subgraph analysis["Analyse"]
        ALG["Assoziations-<br/>Algorithmus"]
    end
    
    subgraph rules["Gefundene Regeln"]
        R1["Wer Brot kauft,<br/>kauft oft auch Butter"]
        R2["Brot + Butter<br/>→ oft auch Milch"]
    end
    
    T1 --> ALG
    T2 --> ALG
    T3 --> ALG
    T4 --> ALG
    ALG --> R1
    ALG --> R2
    
    style rules fill:#e8f5e9
```

**Das klassische Beispiel: Warenkorbanalyse**

> **"Wer Windeln kauft, kauft oft auch Bier"** – Diese berühmte (wenn auch umstrittene) Entdeckung zeigt, wie Assoziationsanalyse unerwartete Zusammenhänge aufdecken kann.

**Anwendungsbeispiele:**
- **Empfehlungssysteme**: "Kunden, die X kauften, kauften auch Y"
- **Cross-Selling**: Produktempfehlungen im E-Commerce
- **Angebotsgestaltung**: Produktbündel und Rabattaktionen

---

## Zusammenfassung

```mermaid
flowchart TB
    ML["Machine Learning"]
    
    ML --> SL["Supervised Learning<br/>📊"]
    ML --> UL["Unsupervised Learning<br/>🔍"]
    ML --> RL["Reinforcement Learning<br/>🎮"]
    
    SL --> SL1["Klassifizierung<br/>Kategorien vorhersagen"]
    SL --> SL2["Regression<br/>Zahlen vorhersagen"]
    
    UL --> UL1["Clustering<br/>Gruppen finden"]
    UL --> UL2["Anomalieerkennung<br/>Ausreißer finden"]
    UL --> UL3["Assoziationsanalyse<br/>Zusammenhänge finden"]
    
    RL --> RL1["Strategieoptimierung<br/>Optimales Handeln lernen"]
    
    style ML fill:#e1f5fe
    style SL fill:#c8e6c9
    style UL fill:#bbdefb
    style RL fill:#fff9c4
```

### Die wichtigsten Punkte

| Konzept | Kernaussage |
|---------|-------------|
| **Machine Learning** | Computer lernen aus Daten, Muster zu erkennen und Vorhersagen zu treffen |
| **Supervised Learning** | Lernen mit gelabelten Daten – die richtigen Antworten sind bekannt |
| **Unsupervised Learning** | Lernen ohne Labels – Strukturen und Muster selbst entdecken |
| **Klassifizierung** | Kategoriale Vorhersagen (z.B. Spam/Kein Spam) |
| **Regression** | Numerische Vorhersagen (z.B. Preis in €) |
| **Clustering** | Ähnliche Datenpunkte gruppieren |
| **Anomalieerkennung** | Ungewöhnliche Datenpunkte identifizieren |

### Entscheidungshilfe: Welcher Ansatz?

```mermaid
flowchart TB
    START["Welchen ML-Ansatz<br/>brauche ich?"]
    
    Q1{"Habe ich<br/>gelabelte Daten?"}
    START --> Q1
    
    Q1 --> |"Ja"| Q2{"Welche Art<br/>von Vorhersage?"}
    Q1 --> |"Nein"| Q3{"Was will ich<br/>herausfinden?"}
    
    Q2 --> |"Kategorien"| A1["Klassifizierung"]
    Q2 --> |"Zahlenwerte"| A2["Regression"]
    
    Q3 --> |"Gruppen"| A3["Clustering"]
    Q3 --> |"Ausreißer"| A4["Anomalieerkennung"]
    Q3 --> |"Zusammenhänge"| A5["Assoziationsanalyse"]
    
    style A1 fill:#c8e6c9
    style A2 fill:#c8e6c9
    style A3 fill:#bbdefb
    style A4 fill:#bbdefb
    style A5 fill:#bbdefb
```



---

**Version:** 1.0       
**Stand:** Januar 2026     
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.     
