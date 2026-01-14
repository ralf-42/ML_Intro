---
layout: default
title: K-Means & DBSCAN
parent: Modeling
grand_parent: Konzepte
nav_order: 4
description: "Clustering-Algorithmen zur Entdeckung von Strukturen in Daten: K-Means für partitionsbasiertes und DBSCAN für dichtebasiertes Clustering"
has_toc: true
---

# Clustering: K-Means & DBSCAN
{: .no_toc }

> **Clustering-Verfahren entdecken verborgene Strukturen in Daten, indem sie ähnliche Datenpunkte zu Gruppen zusammenfassen.**    
>  K-Means eignet sich für kompakte, gleichmäßige Cluster, während DBSCAN beliebig geformte Cluster erkennt und Ausreißer identifiziert.

---

## Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Einführung in Clustering

Clustering bezeichnet Verfahren zur Entdeckung von Ähnlichkeitsstrukturen in Datenbeständen. Die gefundenen Gruppen ähnlicher Objekte werden als **Cluster** bezeichnet.

Im Gegensatz zum überwachten Lernen sind beim Clustering keine Labels vorhanden – der Algorithmus muss die Struktur selbst entdecken. Dies macht Clustering zu einem zentralen Werkzeug des **unüberwachten Lernens**.

```mermaid
flowchart LR
    subgraph Input["Eingabedaten"]
        D1[("Unlabeled Data")]
    end
    
    subgraph Clustering["Clustering-Verfahren"]
        K["K-Means"]
        DB["DBSCAN"]
    end
    
    subgraph Output["Ergebnis"]
        C1["Cluster 1"]
        C2["Cluster 2"]
        C3["Cluster N"]
        N["Noise/Ausreißer"]
    end
    
    D1 --> K
    D1 --> DB
    K --> C1
    K --> C2
    K --> C3
    DB --> C1
    DB --> C2
    DB --> N
    
    style D1 fill:#e8f4f8,stroke:#1e88e5
    style K fill:#fff3e0,stroke:#ff9800
    style DB fill:#f3e5f5,stroke:#9c27b0
    style N fill:#ffebee,stroke:#f44336
```

---

## Clustering-Ansätze im Überblick

| Ansatz              | Methode              | Charakteristik                                                                   |
| ------------------- | -------------------- | -------------------------------------------------------------------------------- |
| **Partitionierend** | K-Means              | Teilt Daten in k vordefinierte Cluster; iterative Optimierung der Clusterzentren |
| **Dichtebasiert**   | DBSCAN               | Erkennt Cluster anhand der Datendichte; identifiziert Ausreißer automatisch      |


---

## K-Means Clustering

### Grundprinzip

K-Means ist ein **partitionierender** Clustering-Algorithmus, der einen Datensatz in **K verschiedene, nicht überlappende Cluster** aufteilt. Der Algorithmus gehört zu den am häufigsten verwendeten Techniken zur Gruppierung von Objekten, da er schnell die Zentren der Cluster findet.

```mermaid
flowchart TD
    subgraph Init["#1 Initialisierung"]
        A["K Clusterzentren<br/>zufällig wählen"]
    end
    
    subgraph Assign["#2 Zuweisung"]
        B["Jeden Datenpunkt<br/>nächstem Zentrum zuweisen"]
    end
    
    subgraph Update["#3 Update"]
        C["Neue Clusterzentren<br/>als Mittelwert berechnen"]
    end
    
    subgraph Check["#4 Konvergenz"]
        D{"Zentren<br/>verändert?"}
    end
    
    subgraph Result["#5 Ergebnis"]
        E["K finale Cluster"]
    end
    
    A --> B
    B --> C
    C --> D
    D -->|Ja| B
    D -->|Nein| E
    
    style A fill:#e3f2fd,stroke:#1976d2
    style B fill:#fff3e0,stroke:#ff9800
    style C fill:#e8f5e9,stroke:#4caf50
    style D fill:#fce4ec,stroke:#e91e63
    style E fill:#f3e5f5,stroke:#9c27b0
```

### Algorithmus-Schritte

1. **Initialisierung**: Wähle K initiale Clusterzentren (zufällig oder mit K-Means++)
2. **Zuweisung**: Ordne jeden Datenpunkt dem nächsten Clusterzentrum zu
3. **Update**: Berechne neue Clusterzentren als Mittelwert aller zugewiesenen Punkte
4. **Iteration**: Wiederhole Schritte 2-3 bis Konvergenz (keine Änderung mehr)

### Eigenschaften von K-Means

| Eigenschaft      | Beschreibung                                                                |
| ---------------- | --------------------------------------------------------------------------- |
| **Clusterform**  | Bevorzugt kugelförmige, kompakte Cluster                                    |
| **Clustergröße** | Tendiert zu ähnlich großen Clustern                                         |
| **Varianz**      | Minimiert die Varianz innerhalb der Cluster                                 |


### Optimale Clusterzahl mit der Elbow-Methode

Die Wahl der richtigen Anzahl K ist entscheidend. Die **Elbow-Methode** hilft dabei, indem sie die Inertia (Within-Cluster Sum of Squares) für verschiedene K-Werte visualisiert.


```mermaid
xychart-beta
    title "Elbow-Methode"
    x-axis "Anzahl Cluster (K)" [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y-axis "Inertia (WCSS)" 0 --> 1000
    line [950, 600, 350, 180, 150, 130, 115, 105, 98, 92]
```

> **Tipp:** Der "Ellbogen" im Plot zeigt die optimale Clusterzahl – dort, wo die Kurve abknickt und weitere Cluster nur noch geringe Verbesserungen bringen.

---

## DBSCAN Clustering

### Grundprinzip

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) ist ein dichtebasierter Algorithmus, der Cluster als Regionen hoher Datendichte definiert, getrennt durch Gebiete niedriger Dichte.

Im Gegensatz zu K-Means benötigt DBSCAN keine Vorgabe der Clusterzahl und kann:
- **Beliebig geformte Cluster** erkennen
- **Rauschen (Noise)** automatisch identifizieren
- Mit **Ausreißern** umgehen

```mermaid
flowchart TD
    subgraph Params["Parameter"]
        E["ε (eps): Radius"]
        M["min_samples: Mindestpunkte"]
    end
    
    subgraph Points["Punkttypen"]
        CP["🔴 Core Point<br/>≥ min_samples in ε-Radius"]
        BP["🟡 Border Point<br/>In ε-Radius eines Core Points"]
        NP["⚫ Noise Point<br/>Weder Core noch Border"]
    end
    
    subgraph Process["Algorithmus"]
        P1["#1 Core Points identifizieren"]
        P2["#2 Core Points verbinden<br/>(wenn in ε-Distanz)"]
        P3["#3 Border Points zuweisen"]
        P4["#4 Noise Points markieren"]
    end
    
    E --> P1
    M --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> CP
    P4 --> BP
    P4 --> NP
    
    style CP fill:#ffcdd2,stroke:#c62828
    style BP fill:#fff9c4,stroke:#f9a825
    style NP fill:#e0e0e0,stroke:#424242
```

### Schlüsselkonzepte

| Konzept | Definition |
|---------|------------|
| **ε (epsilon)** | Der Radius, innerhalb dessen Nachbarn gesucht werden |
| **min_samples** | Minimale Anzahl Punkte für einen Core Point |
| **Core Point** | Punkt mit mindestens min_samples Nachbarn im ε-Radius |
| **Border Point** | Punkt im ε-Radius eines Core Points, aber selbst kein Core Point |
| **Noise Point** | Punkt, der weder Core noch Border Point ist (Ausreißer) |

### Algorithmus-Ablauf

1. **Core Points finden**: Alle Punkte mit ≥ min_samples Nachbarn im ε-Radius markieren
2. **Cluster bilden**: Verbundene Core Points gehören zum selben Cluster
3. **Border Points zuweisen**: Jedem erreichbaren Core Point-Cluster zuordnen
4. **Noise klassifizieren**: Übrige Punkte als Rauschen markieren


---

## Distanzmetriken

Die Wahl der Distanzmetrik beeinflusst das Clustering-Ergebnis erheblich. Beide Algorithmen können verschiedene Metriken verwenden.

### Übersicht gängiger Distanzmaße

```mermaid
flowchart LR
    subgraph Metriken["Distanzmetriken"]
        E["Euklidisch<br/>√Σ(xi-yi)²"]
        M["Manhattan<br/>Σ|xi-yi|"]
        C["Cosinus<br/>1 - (x·y)/(||x||·||y||)"]
        Min["Minkowski<br/>ᵖ√Σ|xi-yi|ᵖ"]
    end
    
    subgraph Anwendung["Typische Anwendung"]
        E --> E1["Geometrische Daten"]
        M --> M1["Taxi-Distanzen,<br/>kategorische Daten"]
        C --> C1["Textdaten,<br/>Empfehlungssysteme"]
        Min --> Min1["Generalisierung<br/>(p=1: Manhattan, p=2: Euklidisch)"]
    end
    
    style E fill:#e3f2fd,stroke:#1976d2
    style M fill:#fff3e0,stroke:#ff9800
    style C fill:#e8f5e9,stroke:#4caf50
    style Min fill:#f3e5f5,stroke:#9c27b0
```

### Vergleich der Distanzmetriken

| Metrik | Formel | Charakteristik | Anwendung |
|--------|--------|----------------|-----------|
| **Euklidisch** | √Σ(xᵢ-yᵢ)² | Direkte Luftlinie | Standard für kontinuierliche Daten |
| **Manhattan** | Σ\|xᵢ-yᵢ\| | Summe der Achsenabstände | Robuster bei Ausreißern |
| **Cosinus** | 1 - cos(θ) | Winkel zwischen Vektoren | Textähnlichkeit, Embeddings |
| **Minkowski** | (Σ\|xᵢ-yᵢ\|ᵖ)^(1/p) | Generalisierung | Parameter p anpassbar |


---

## K-Means vs. DBSCAN: Vergleich

```mermaid
flowchart TD
    subgraph KMeans["K-Means"]
        K1["✓ Schnell & effizient"]
        K2["✓ Einfach zu verstehen"]
        K3["✗ K muss vorgegeben werden"]
        K4["✗ Nur kugelförmige Cluster"]
        K5["✗ Empfindlich bei Ausreißern"]
    end
    
    subgraph DBSCAN_Box["DBSCAN"]
        D1["✓ Findet beliebige Formen"]
        D2["✓ Erkennt Ausreißer"]
        D3["✓ Keine Clusterzahl nötig"]
        D4["✗ Parameter ε schwer zu wählen"]
        D5["✗ Probleme bei variierender Dichte"]
    end
    
    style K1 fill:#e8f5e9,stroke:#4caf50
    style K2 fill:#e8f5e9,stroke:#4caf50
    style K3 fill:#ffebee,stroke:#f44336
    style K4 fill:#ffebee,stroke:#f44336
    style K5 fill:#ffebee,stroke:#f44336
    style D1 fill:#e8f5e9,stroke:#4caf50
    style D2 fill:#e8f5e9,stroke:#4caf50
    style D3 fill:#e8f5e9,stroke:#4caf50
    style D4 fill:#ffebee,stroke:#f44336
    style D5 fill:#ffebee,stroke:#f44336
```

### Entscheidungshilfe

| Kriterium | K-Means | DBSCAN |
|-----------|---------|--------|
| **Clusterzahl bekannt** | ✅ Ideal | ❌ Nicht nötig |
| **Kugelförmige Cluster** | ✅ Optimal | ⚠️ Möglich |
| **Beliebige Clusterformen** | ❌ Ungeeignet | ✅ Ideal |
| **Ausreißer im Datensatz** | ❌ Problematisch | ✅ Werden erkannt |
| **Große Datensätze** | ✅ Sehr schnell | ⚠️ Kann langsam sein |
| **Variierende Clusterdichte** | ⚠️ Problematisch | ❌ Problematisch |

---


---

## Best Practices


### Checkliste für erfolgreiches Clustering

- [ ] Daten auf fehlende Werte prüfen
- [ ] Features skalieren (StandardScaler oder MinMaxScaler)
- [ ] Bei K-Means: Optimales K mit Elbow-Methode bestimmen
- [ ] Bei DBSCAN: ε mit k-Distanz-Graph bestimmen
- [ ] Ergebnis mit Silhouette-Score evaluieren
- [ ] Cluster visualisieren und interpretieren

### Häufige Fehler vermeiden

| Fehler | Problem | Lösung |
|--------|---------|--------|
| Keine Skalierung | Unterschiedliche Feature-Skalen dominieren | StandardScaler verwenden |
| Falsches K | Over-/Underfitting | Elbow-Methode + Silhouette |
| ε zu klein/groß | Zu viele/wenige Cluster | k-Distanz-Graph analysieren |
| Nur ein Algorithmus | Suboptimale Ergebnisse | Beide Methoden vergleichen |

---

**Version:** 1.0    
**Stand:** Januar 2026    
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.    
