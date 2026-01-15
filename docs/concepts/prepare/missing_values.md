---
layout: default
title: Missing Values
parent: Prepare
grand_parent: Konzepte
nav_order: 1
description: "Missing Values - Fehlende Werte erkennen und behandeln mit SimpleImputer, KNNImputer und IterativeImputer"
has_toc: true
---

# Missing Values – Fehlende Werte behandeln
{: .no_toc }

> **Behandlung fehlender Werte in Machine Learning Projekten**    
> MCAR, MAR, MNAR - Deletion vs. Imputation - Strategien und Best Practices

---

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Überblick

Fehlende Werte (Missing Values) sind ein häufiges Problem in realen Datensätzen. Sie entstehen durch unvollständige Datenerfassung, Übertragungsfehler, Systemausfälle oder bewusst nicht beantwortete Fragen. Die korrekte Behandlung fehlender Werte ist entscheidend für die Qualität eines Machine-Learning-Modells.

## Entscheidungsbaum zur Behandlung

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px'}}}%%
flowchart TD
    start([Start: Fehlende Werte erkannt]) --> analyse["Analyse des Anteils<br/>fehlender Werte"]
    
    analyse --> check_percent{Anteil fehlender<br/>Werte prüfen}
    
    check_percent -->|"> 50% in Spalte"| drop_col["Spalte löschen<br/>❌ Column Drop"]
    check_percent -->|"< 5% in Zeilen"| drop_row["Zeilen löschen<br/>❌ Row Drop"]
    check_percent -->|"5-50%"| impute_decision{"Art der<br/>Variable?"}
    
    impute_decision -->|"Numerisch"| num_methods{"Verteilung<br/>prüfen"}
    impute_decision -->|"Kategorial"| cat_methods["Modus oder<br/>eigene Kategorie<br/>'Unbekannt'"]
    
    num_methods -->|"Normalverteilt"| mean_imp["Mittelwert<br/>Imputation"]
    num_methods -->|"Schief verteilt"| median_imp["Median<br/>Imputation"]
    num_methods -->|"Komplex"| advanced["Erweiterte<br/>Imputation"]
    
    advanced --> knn["KNN Imputer"]
    advanced --> iterative["Iterative Imputer"]
    advanced --> model["Modellbasierte<br/>Imputation"]
    
    drop_col --> end_node([Ende])
    drop_row --> end_node
    mean_imp --> end_node
    median_imp --> end_node
    cat_methods --> end_node
    knn --> end_node
    iterative --> end_node
    model --> end_node

    style start fill:#2196F3,color:#fff
    style end_node fill:#4CAF50,color:#fff
    style drop_col fill:#f44336,color:#fff
    style drop_row fill:#f44336,color:#fff
```

## Strategien im Detail

### 1. Löschen (Deletion)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px'}}}%%
flowchart LR
    subgraph deletion["<b>Löschstrategien"]
        direction TB
        listwise["<b>Listwise Deletion</b><br/>Komplette Zeile löschen"]
        pairwise["<b>Pairwise Deletion</b><br/>Nur für betroffene<br/>Berechnungen ausschließen"]
        column["<b>Column Deletion</b><br/>Gesamte Spalte löschen"]
    end
    
    listwise --> lw_pro["✅ Einfach umzusetzen<br/>✅ Konsistente Datensätze"]
    listwise --> lw_con["❌ Datenverlust<br/>❌ Bias bei nicht-MCAR"]
    
    pairwise --> pw_pro["✅ Mehr Daten nutzbar"]
    pairwise --> pw_con["❌ Inkonsistente n<br/>❌ Komplexer"]
    
    column --> col_pro["✅ Schnell bei vielen NaN"]
    column --> col_con["❌ Informationsverlust<br/>❌ Evtl. wichtige Features weg"]

    style listwise fill:#e3f2fd
    style pairwise fill:#e3f2fd
    style column fill:#e3f2fd
```

### 2. Imputation (Auffüllen)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px'}}}%%
flowchart TB
    subgraph simple["<b>Einfache Imputation"]
        mean["<b>Mittelwert</b><br/>Durchschnitt der Spalte"]
        median["<b>Median</b><br/>Zentralwert der Spalte"]
        mode["<b>Modus</b><br/>Häufigster Wert"]
        constant["<b>Konstante</b><br/>Fester Platzhalterwert"]
    end
    
    subgraph advanced["<b>Erweiterte Imputation"]
        knn["<b>KNN Imputer</b><br/>k nächste Nachbarn<br/>zur Schätzung"]
        iterative["<b>Iterative Imputer</b><br/>Mehrfache Schätzung<br/>mit ML-Modellen"]
        mice["<b>MICE</b><br/>Multiple Imputation by<br/>Chained Equations"]
    end
    
    mean --> mean_use["Normalverteilte<br/>numerische Daten"]
    median --> median_use["Schiefe Verteilungen<br/>mit Ausreißern"]
    mode --> mode_use["Kategoriale<br/>Variablen"]
    constant --> constant_use["Wenn Fehlen<br/>inhaltliche Bedeutung hat"]
    
    knn --> knn_use["Lokale Muster<br/>wichtig"]
    iterative --> iterative_use["Komplexe Abhängigkeiten<br/>zwischen Features"]
    mice --> mice_use["Statistische Analysen<br/>mit Unsicherheit"]

    style simple fill:#e8f5e9
    style advanced fill:#fff3e0
```


## Konsequenzen falscher Behandlung

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px'}}}%%
flowchart TD
    wrong["Falsche Behandlung<br/>fehlender Werte"]
    
    wrong --> bias["Verzerrung (Bias)<br/>Systematische Fehler<br/>im Modell"]
    wrong --> overfit["Overfitting<br/>Modell lernt<br/>Artefakte"]
    wrong --> underfit["Underfitting<br/>Zu wenig<br/>Trainingsdaten"]
    wrong --> invalid["Ungültige Ergebnisse<br/>Falsche Schluss-<br/>folgerungen"]
    
    bias --> result1["Diskriminierende<br/>Vorhersagen"]
    overfit --> result2["Schlechte<br/>Generalisierung"]
    underfit --> result3["Niedrige<br/>Modellgüte"]
    invalid --> result4["Fehlerhafte<br/>Entscheidungen"]

    style wrong fill:#f44336,color:#fff
    style result1 fill:#ffcdd2
    style result2 fill:#ffcdd2
    style result3 fill:#ffcdd2
    style result4 fill:#ffcdd2
```

## Best Practices

| Empfehlung | Beschreibung |
|------------|--------------|
| **Dokumentieren** | Anzahl und Verteilung fehlender Werte vor der Behandlung festhalten |
| **Visualisieren** | Muster in fehlenden Werten erkennen (z.B. mit `missingno`) |
| **Kontext beachten** | Fachliche Bedeutung fehlender Werte berücksichtigen |
| **Mehrere Strategien testen** | Auswirkung verschiedener Methoden auf Modellgüte vergleichen |
| **Pipeline nutzen** | Imputation als Teil der sklearn-Pipeline für konsistente Anwendung |
| **Train/Test trennen** | Imputation nur auf Trainingsdaten fitten, dann auf Testdaten anwenden |

## Scikit-learn Klassen

| Klasse | Verwendung |
|--------|------------|
| `SimpleImputer` | Einfache Strategien (mean, median, most_frequent, constant) |
| `KNNImputer` | K-Nearest-Neighbors basierte Imputation |
| `IterativeImputer` | Multivariate Imputation mit ML-Modellen |


---

**Version:** 1.0    
**Stand:** Januar 2026    
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.    
