---
layout: default
title: PCA und LDA
parent: Modeling
grand_parent: Konzepte
nav_order: 7
description: "Dimensionsreduktion durch Principal Component Analysis (PCA) und Linear Discriminant Analysis (LDA) - Konzepte, Unterschiede und praktische Implementierung"
has_toc: true
---

# PCA und LDA
{: .no_toc }

> **Dimensionsreduktion ist eine Schlüsseltechnik im Machine Learning, um hochdimensionale Daten auf ihre wesentlichen Merkmale zu reduzieren.**
> PCA und LDA sind zwei fundamentale Ansätze, die unterschiedliche Ziele verfolgen: PCA maximiert die Varianz, während LDA die Klassentrennung optimiert.

---

## Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Grundidee

### Das Problem: Zu viele Informationen

Heutzutage werden enorme Datenmengen gesammelt. Für das Training von KI-Systemen ist das einerseits gut, da mehr Daten oft zu besseren Ergebnissen führen. Andererseits benötigen diese riesigen Mengen extrem viel **Speicherplatz und Rechenzeit**, was besonders bei Echtzeitanwendungen zum Problem wird.

### Die Lösung: Dimensionsreduktion

Anstatt einfach Datenpunkte (Beispiele) zu löschen, verringert man bei der **Dimensionsreduktion** die Anzahl der **Merkmale** (Dimensionen). Es gibt zwei Wege, dies zu tun:

1. **Merkmalsauswahl**: Man pickt sich nur einen Teil der vorhandenen Merkmale heraus.
2. **Merkmalsprojektion**: Man rechnet alle Merkmale so um, dass sie durch weniger, neue Merkmale beschrieben werden können. Die **PCA** oder LDA gehört zu dieser zweiten Kategorie.

### Wie funktioniert die PCA? (Das Schatten-Beispiel)

Das Prinzip der PCA lässt sich vereinfacht mit einem **Schattenwurf** erklären:

- Stellen Sie sich einen dreidimensionalen Würfel vor. Wenn man ihn mit einer Taschenlampe anstrahlt, wirft er einen zweidimensionalen Schatten auf den Boden.
- Der Schatten ist eine **Projektion** des Würfels: Er hat weniger Dimensionen (2D statt 3D), zeigt aber immer noch die wesentlichen Eigenschaften des Objekts.
- Die PCA sucht nun genau den Winkel („die beste Lichtquelle“), bei dem der Schatten die meiste Information des Originals beibehält. Mathematisch bedeutet das, dass die **Varianz** (die Streuung der Daten) im Schattenwurf maximal sein soll.

<img src="https://raw.githubusercontent.com/ralf-42/ML_Intro/main/07_image/wuerfel.png" class="logo" width="650"/>

### Hauptkomponenten und Nutzen

- **Hauptkomponenten**: Dies sind die neuen Koordinatenachsen, auf die die Daten projiziert werden. Die erste Hauptkomponente zeigt immer in die Richtung der größten Streuung.
- **Vorteile**: Neben der Ersparnis von Rechenzeit und Speicher kann die PCA auch dabei helfen, unerwünschtes **Rauschen** (Störsignale) in den Daten zu unterdrücken.
- **Risiko**: Man muss vorsichtig sein. In manchen Fällen kann die Reduktion dazu führen, dass verschiedene Datenklassen (zum Beispiel unterschiedliche Pflanzentypen) so stark überlappen, dass man sie nach der PCA nicht mehr voneinander unterscheiden kann.

Zusammenfassend ist die PCA also eine Methode, um die **Komplexität von Daten zu verringern**, indem man sie auf ihre wichtigsten Bestandteile reduziert, ähnlich wie ein aussagekräftiger Schatten ein komplexes Objekt vereinfacht darstellt.


---

[PCA - Eine Methode zur Dimensionsreduktion » Lamarr-Institut](https://lamarr-institute.org/de/blog/dimensionsreduktion-pca/)

---

## Principal Component Analysis (PCA)

Die **Hauptkomponentenanalyse (PCA)** ist die am häufigsten verwendete Methode zur Dimensionsreduktion. Sie projiziert Datenpunkte in einen Unterraum mit weniger Dimensionen, wobei die **Varianz der Daten maximiert** wird.

### Funktionsweise

```mermaid
flowchart TD
    A["Originaldaten<br/>(n Dimensionen)"] --> B["Kovarianzmatrix<br/>berechnen"]
    B --> C["Eigenwerte &<br/>Eigenvektoren"]
    C --> D["Hauptkomponenten<br/>auswählen"]
    D --> E["Daten auf neue<br/>Achsen projizieren"]
    E --> F["Reduzierte Daten<br/>(k Dimensionen)"]
    
    style A fill:#ffcdd2,stroke:#d32f2f
    style F fill:#c8e6c9,stroke:#388e3c
```

**Schrittweise Erklärung:**

1. **Varianzmaximierung:** Der Unterraum wird so gewählt, dass die Varianz der projizierten Datenpunkte maximal ist
2. **Erste Hauptkomponente:** Eine Gerade durch die Daten, welche die Varianz der orthogonal projizierten Punkte maximiert
3. **Weitere Komponenten:** Jede zusätzliche Achse steht senkrecht zur vorherigen und erklärt die verbleibende Varianz

### Eigenschaften von PCA

| Eigenschaft             | Beschreibung                                      |
| ----------------------- | ------------------------------------------------- |
| **Lerntyp**             | Unüberwacht (keine Labels erforderlich)           |
| **Ziel**                | Maximierung der Gesamtvarianz                     |
| **Interpretierbarkeit** | Neue Achsen sind interpretierbar                  |
| **Anwendung**           | Datenkompression, Visualisierung, Rauschreduktion |

---

 *Beispiel Use Case "Cancer": siehe b240_ul_pca_special.ipynb*

---

## Linear Discriminant Analysis (LDA)

Die **Lineare Diskriminanzanalyse (LDA)** ist eine überwachte Methode zur Dimensionsreduktion. Im Gegensatz zu PCA nutzt LDA die **Klasseninformationen**, um eine optimale Trennung zwischen den Klassen zu finden.

### Funktionsweise

```mermaid
flowchart TD
    A["Daten mit<br/>Klassenlabels"] --> B["Berechne Varianz<br/>ZWISCHEN Klassen"]
    A --> C["Berechne Varianz<br/>INNERHALB Klassen"]
    B --> D["Maximiere<br/>Zwischen/Innerhalb"]
    C --> D
    D --> E["Optimale<br/>Trennachse"]
    E --> F["Projizierte<br/>Daten"]
    
    style A fill:#e1bee7,stroke:#7b1fa2
    style F fill:#c8e6c9,stroke:#388e3c
```

**Kernidee:** LDA sucht nach linearen Kombinationen der Merkmale, die:
- Die **Varianz zwischen den Klassen** maximieren
- Die **Varianz innerhalb jeder Klasse** minimieren

### Eigenschaften von LDA

| Eigenschaft | Beschreibung |
|-------------|--------------|
| **Lerntyp** | Überwacht (Labels erforderlich) |
| **Ziel** | Maximierung der Klassentrennung |
| **Max. Komponenten** | Anzahl Klassen - 1 |
| **Anwendung** | Klassifikation, Vorverarbeitung |


---

## Vergleich: PCA vs. LDA

Die Wahl zwischen PCA und LDA hängt vom Anwendungsfall und den verfügbaren Daten ab.

```mermaid
flowchart TD
    START["Dimensionsreduktion<br/>benötigt"] --> Q1{"Labels<br/>vorhanden?"}
    
    Q1 -->|Nein| PCA["✅ PCA verwenden"]
    Q1 -->|Ja| Q2{"Ziel?"}
    
    Q2 -->|"Klassifikation<br/>optimieren"| LDA["✅ LDA verwenden"]
    Q2 -->|"Allgemeine<br/>Kompression"| Q3{"Klassenverteilung<br/>balanciert?"}
    
    Q3 -->|Ja| BOTH["PCA oder LDA<br/>ausprobieren"]
    Q3 -->|Nein| PCA2["✅ PCA bevorzugen"]
    
    style PCA fill:#c8e6c9,stroke:#388e3c
    style LDA fill:#bbdefb,stroke:#1976d2
    style PCA2 fill:#c8e6c9,stroke:#388e3c
    style BOTH fill:#fff3e0,stroke:#f57c00
```

### Gegenüberstellung

| Kriterium | PCA | LDA |
|-----------|-----|-----|
| **Lerntyp** | Unüberwacht | Überwacht |
| **Ziel** | Maximierung der Gesamtvarianz | Maximierung der Klassentrennung |
| **Labels erforderlich** | ❌ Nein | ✅ Ja |
| **Max. Komponenten** | min(n_samples, n_features) | n_classes - 1 |
| **Overfitting-Risiko** | Gering | Höher (bei wenig Daten) |
| **Interpretierbarkeit** | Hoch | Moderat |
| **Typische Anwendung** | Datenkompression, Visualisierung | Klassifikation, Vorverarbeitung |

### Visuelle Unterschiede

```mermaid
flowchart LR
    subgraph pca_result["PCA Ergebnis"]
        direction TB
        P1["Maximiert<br/>Gesamtvarianz"]
        P2["Klassen können<br/>überlappen"]
    end
    
    subgraph lda_result["LDA Ergebnis"]
        direction TB
        L1["Maximiert<br/>Klassentrennung"]
        L2["Klassen optimal<br/>separiert"]
    end
    
    style pca_result fill:#e8f5e9,stroke:#388e3c
    style lda_result fill:#e3f2fd,stroke:#1976d2
```

---

## Wann welche Methode verwenden?

### PCA empfohlen bei:

- Keine Klassenlabels vorhanden
- Ziel ist allgemeine Datenkompression
- Visualisierung hochdimensionaler Daten
- Rauschreduktion gewünscht
- Vorverarbeitung für unüberwachtes Lernen

### LDA empfohlen bei:

- Klassenlabels verfügbar
- Nachfolgende Klassifikationsaufgabe
- Klassen sollen optimal getrennt werden
- Wenige Klassen im Verhältnis zu Features
- Klassenverteilung einigermaßen balanciert

> **Best Practice**
>
> In der Praxis werden PCA und LDA oft kombiniert: Zuerst reduziert PCA die Dimensionen auf ein handhabbares Maß, dann optimiert LDA die Klassentrennung in diesem reduzierten Raum.

---

## Zusammenfassung

```mermaid
mindmap
  root((Dimensionsreduktion))
    PCA
      Unüberwacht
      Maximiert Varianz
      Keine Labels nötig
      Datenkompression
    LDA
      Überwacht
      Maximiert Klassentrennung
      Labels erforderlich
      Klassifikation
    Gemeinsamkeiten
      Lineare Projektion
      Reduzieren Dimensionen
      Interpretierbare Komponenten
```

| Aspekt | PCA | LDA |
|--------|-----|-----|
| **Kernfrage** | "Welche Richtung erklärt die meiste Variation?" | "Welche Richtung trennt die Klassen am besten?" |
| **Stärke** | Universell einsetzbar | Optimal für Klassifikation |
| **Schwäche** | Ignoriert Klassenzugehörigkeit | Braucht Labels, max. k-1 Komponenten |


---

**Version:** 1.0    
**Stand:** Januar 2026    
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.    