---
layout: default
title: Kodierung
parent: Prepare
grand_parent: Konzepte
nav_order: 5
description: "Kodierung kategorialer Daten - OrdinalEncoder, OneHotEncoder und TargetEncoder für Machine Learning"
has_toc: true
---

# Kodierung kategorialer Daten
{: .no_toc }

> **Transformation kategorialer Merkmale in numerische Formate**      
> Nominale vs. ordinale Daten, OrdinalEncoder, OneHotEncoder und TargetEncoder

---

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---


## Einführung

Die **meisten** Machine-Learning-Algorithmen können **nur** mit **numerischen Daten** arbeiten. Kategoriale Daten – also Merkmale mit diskreten Ausprägungen wie Farben, Kategorien oder Bewertungen – müssen daher in numerische Formate umgewandelt werden. Dieser Prozess wird als **Kodierung** (Encoding) bezeichnet.

Die Wahl der richtigen Kodierungsmethode hängt dabei maßgeblich vom **Skalenniveau** der Daten ab.

## Skalenniveaus kategorialer Daten

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px'}}}%%
flowchart TB
    subgraph Kategoriale["Kategoriale Skalen"]
        direction TB
        N["<b>Nominale Skalen</b><br/>Keine Rangfolge<br/><i>Beispiele: Blutgruppe,<br/>Geschlecht, Farbe</i>"]
        O["<b>Ordinale Skalen</b><br/>Mit Rangfolge<br/><i>Beispiele: Schulnoten,<br/>Rating, Bildungsgrad</i>"]
    end
    
    subgraph Numerische["Numerische Skalen"]
        direction TB
        I["<b>Intervallskalen</b><br/>Definierte Abstände<br/><i>Beispiele: Temperatur,<br/>Zeitrechnung, IQ</i>"]
        V["<b>Verhältnisskalen</b><br/>Absoluter Nullpunkt<br/><i>Beispiele: Gewicht,<br/>Einkommen, Alter</i>"]
    end
    
    K["Kategoriale Daten<br/>(diskret)"] --> Kategoriale
    NUM["Numerische Daten<br/>(meist stetig)"] --> Numerische
```

### Nominale Daten
- Merkmalsausprägungen haben **keine unterschiedliche Wertigkeit**
- Reihenfolge ist beliebig und bedeutungslos
- Beispiele: Blutgruppe (A, B, AB, 0), Geschlecht, Automarke, Farbe

### Ordinale Daten
- Merkmalsausprägungen haben eine **natürliche Rangfolge**
- Abstände zwischen den Kategorien sind nicht definiert
- Beispiele: Schulnoten (1-6), Kundenzufriedenheit (sehr unzufrieden bis sehr zufrieden), Bildungsgrad

## Entscheidungsbaum: Welche Kodierung verwenden?

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px'}}}%%
flowchart TD
    Start["Kategoriale<br/>Daten"] --> Q1{"Haben die Kategorien<br/>eine natürliche<br/>Rangfolge?"}
    
    Q1 -->|"Ja<br/>(ordinal)"| OE["<b>OrdinalEncoder</b><br/>Ganzzahlen: 0, 1, 2, ..."]
    
    Q1 -->|"Nein<br/>(nominal)"| Q2{"Wie viele<br/>Kategorien?"}
    
    Q2 -->|"Wenige<br/>(< 10-15)"| OHE["<b>OneHotEncoder</b><br/>Binäre Spalten"]
    
    Q2 -->|"Viele<br/>(> 15)"| Q3{"Supervised<br/>Learning?"}
    
    Q3 -->|"Ja"| TE["<b>TargetEncoder</b><br/>Zielwert-basiert"]
    Q3 -->|"Nein"| OHE2["<b>OneHotEncoder</b><br/>oder Dimensionsreduktion"]
    
    style OE fill:#e1f5fe
    style OHE fill:#fff3e0
    style OHE2 fill:#fff3e0
    style TE fill:#f3e5f5
```

## Kodierungsmethoden im Detail

### 1. OrdinalEncoder (für ordinale Daten)

Der **OrdinalEncoder** konvertiert kategoriale Merkmale in ganzzahlige Werte (0 bis n-1). Diese Methode ist geeignet, wenn die Kategorien eine **natürliche Rangfolge** besitzen.

#### Funktionsweise

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px'}}}%%
flowchart LR
    subgraph Vorher["Originaldaten"]
        V1["schlecht"]
        V2["mittel"]
        V3["gut"]
        V4["sehr gut"]
    end
    
    subgraph Nachher["Nach OrdinalEncoder"]
        N1["0"]
        N2["1"]
        N3["2"]
        N4["3"]
    end
    
    V1 --> N1
    V2 --> N2
    V3 --> N3
    V4 --> N4
```

#### Beispiel in scikit-learn

```python
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

# Beispieldaten mit ordinaler Kategorie
df = pd.DataFrame({
    'bildung': ['Hauptschule', 'Realschule', 'Abitur', 'Bachelor', 'Master']
})

# Encoder mit definierter Reihenfolge
encoder = OrdinalEncoder(
    categories=[['Hauptschule', 'Realschule', 'Abitur', 'Bachelor', 'Master']]
)

# Transformation
df['bildung_encoded'] = encoder.fit_transform(df[['bildung']])

# Ergebnis:
# Hauptschule -> 0
# Realschule  -> 1
# Abitur      -> 2
# Bachelor    -> 3
# Master      -> 4
```

#### Wann verwenden?
- Bildungsgrad, Einkommensklassen, Zufriedenheitsskalen
- Größenangaben (S, M, L, XL)
- Qualitätsstufen (niedrig, mittel, hoch)

#### Vorteile und Nachteile

| Vorteile | Nachteile |
|----------|-----------|
| Erhält die Rangfolge | Impliziert gleiche Abstände |
| Speichereffizient (eine Spalte) | Nur für ordinale Daten geeignet |
| Keine Dimensionserhöhung | Kann bei nominalen Daten irreführend sein |

### 2. OneHotEncoder (für nominale Daten)

Der **OneHotEncoder** erstellt für jede Kategorie eine eigene binäre Spalte. Diese Methode ist ideal für **nominale Daten ohne natürliche Rangfolge**.

#### Funktionsweise

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px'}}}%%
flowchart LR
    subgraph Vorher["Originaldaten"]
        direction TB
        F["Farbe"]
        F1["rot"]
        F2["blau"]
        F3["grün"]
    end
    
    subgraph Nachher["Nach OneHotEncoder"]
        direction TB
        H["Farbe_rot | Farbe_blau | Farbe_grün"]
        H1["1 | 0 | 0"]
        H2["0 | 1 | 0"]
        H3["0 | 0 | 1"]
    end
    
    F1 --> H1
    F2 --> H2
    F3 --> H3
```

#### Beispiel in scikit-learn

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Beispieldaten mit nominaler Kategorie
df = pd.DataFrame({
    'farbe': ['rot', 'blau', 'grün', 'rot', 'blau']
})

# OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded = encoder.fit_transform(df[['farbe']])

# Als DataFrame mit Spaltennamen
encoded_df = pd.DataFrame(
    encoded, 
    columns=encoder.get_feature_names_out(['farbe'])
)

# Ergebnis:
#    farbe_blau  farbe_grün  farbe_rot
# 0         0.0         0.0        1.0
# 1         1.0         0.0        0.0
# 2         0.0         1.0        0.0
# 3         0.0         0.0        1.0
# 4         1.0         0.0        0.0
```

#### Parameter-Optionen

```python
# Drop='first' vermeidet Multikollinearität (für lineare Modelle)
encoder = OneHotEncoder(sparse_output=False, drop='first')

# handle_unknown='ignore' für unbekannte Kategorien in neuen Daten
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
```

#### Wann verwenden?
- Farben, Länder, Produktkategorien
- Zahlungsmethoden, Versandarten
- Alle nominalen Daten mit **wenigen Kategorien** (< 10-15)

#### Vorteile und Nachteile

| Vorteile | Nachteile |
|----------|-----------|
| Keine implizite Rangfolge | Dimensionserhöhung (Curse of Dimensionality) |
| Funktioniert mit allen Algorithmen | Speicherintensiv bei vielen Kategorien |
| Vermeidet ordinale Beziehungen | Sparse Matrizen können Probleme verursachen |

### 3. TargetEncoder (für nominale Daten mit vielen Kategorien)

Der **TargetEncoder** ersetzt kategoriale Werte durch einen Wert, der aus dem Zielmerkmal (Target) berechnet wird – typischerweise der Mittelwert des Targets für diese Kategorie.

#### Funktionsweise

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px'}}}%%
flowchart TB
    subgraph Input["<b>Eingabedaten"]
        direction LR
        Stadt["Stadt"]
        Preis["Preis (Target)"]
    end
    
    subgraph Berechnung["<b>Berechnung Mittelwert je Kategorie"]
        direction TB
        B["Berlin: (300k + 350k + 280k) / 3 = 310k"]
        M["München: (450k + 500k + 480k) / 3 = 477k"]
        H["Hamburg: (320k + 340k) / 2 = 330k"]
    end
    
    subgraph Output["<b>Kodierte Daten"]
        direction LR
        E["Stadt_encoded"]
        E1["310000"]
        E2["477000"]
        E3["330000"]
    end
    
    Stadt --> Berechnung
    Preis --> Berechnung
    Berechnung --> Output
```

#### Beispiel in scikit-learn

```python
from sklearn.preprocessing import TargetEncoder
import pandas as pd

# Beispieldaten
df = pd.DataFrame({
    'stadt': ['Berlin', 'München', 'Berlin', 'Hamburg', 'München', 'Berlin'],
    'preis': [300000, 450000, 350000, 320000, 500000, 280000]
})

# TargetEncoder
encoder = TargetEncoder(smooth='auto')
df['stadt_encoded'] = encoder.fit_transform(
    df[['stadt']], 
    df['preis']
)

# Ergebnis: Jede Stadt wird durch den mittleren Preis ersetzt
```

#### Wann verwenden?
- Postleitzahlen, Städte mit vielen Ausprägungen
- Produkt-IDs, Kunden-IDs
- Kategorien mit **mehr als 15-20 Ausprägungen**

#### Vorteile und Nachteile

| Vorteile | Nachteile |
|----------|-----------|
| Keine Dimensionserhöhung | Risiko von Overfitting |
| Effektiv bei vielen Kategorien | Mögliche Data Leakage |
| Bezieht Zielinformation ein | Nur für Supervised Learning |

## Vergleich der Kodierungsmethoden

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px'}}}%%
flowchart TB
    subgraph Vergleich["<b>Übersicht Kodierungsmethoden"]
        direction TB
        
        subgraph OE["<b>OrdinalEncoder"]
            OE_T["Typ: Ordinal"]
            OE_D["Dimensionen: 1"]
            OE_R["Rangfolge: ✓ Erhalten"]
        end
        
        subgraph OH["<b>OneHotEncoder"]
            OH_T["Typ: Nominal"]
            OH_D["Dimensionen: n Kategorien"]
            OH_R["Rangfolge: ✗ Keine"]
        end
        
        subgraph TE["<b>TargetEncoder"]
            TE_T["Typ: Nominal (viele)"]
            TE_D["Dimensionen: 1"]
            TE_R["Nutzt: Zielwert"]
        end
    end
```

| Kriterium              | OrdinalEncoder | OneHotEncoder    | TargetEncoder        |
| ---------------------- | -------------- | ---------------- | -------------------- |
| **Datentyp**           | Ordinal        | Nominal          | Nominal              |
| **Kategorien**         | Beliebig       | Wenige (< 15)    | Viele (> 15)         |
| **Dimensionen**        | 1              | n Kategorien     | 1                    |
| **Rangfolge**          | Erhalten       | Keine impliziert | Keine                |
| **Supervised**         | Nein           | Nein             | Ja (benötigt Target) |
| **Overfitting-Risiko** | Gering         | Gering           | Höher                |


Die Wahl der richtigen Kodierung ist entscheidend für die Modellperformance. Grundregel: **Ordinale Daten** mit OrdinalEncoder, **nominale Daten** mit OneHotEncoder (wenige Kategorien) oder TargetEncoder (viele Kategorien).


---

**Version:** 1.0     
**Stand:** Januar 2026     
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.     
