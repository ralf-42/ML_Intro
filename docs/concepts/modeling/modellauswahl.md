---
layout: default
title: Modellauswahl
parent: Modeling
grand_parent: Konzepte
nav_order: 0
description: "Systematische Kriterien und Strategien zur Auswahl des optimalen Machine-Learning-Modells für verschiedene Problemstellungen"
has_toc: true
---

# Modellauswahl
{: .no_toc }

> **Die Wahl des richtigen Modells ist entscheidend für den Erfolg eines ML-Projekts.**     Diese Seite zeigt systematische Kriterien aus Daten- und Use-Case-Perspektive sowie praktische Entscheidungshilfen.

---

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Übersicht

Die Modellauswahl ist einer der kritischsten Schritte im Machine-Learning-Prozess. Ein falsch gewähltes Modell kann trotz perfekter Datenaufbereitung zu unbefriedigenden Ergebnissen führen. Die Auswahl erfolgt anhand zweier Hauptperspektiven: den **Dateneigenschaften** und den **Anforderungen des Use Cases**.

```mermaid
flowchart TB
    subgraph Entscheidungsfaktoren
        A[Modellauswahl]
        A --> B[Perspektive Daten]
        A --> C[Perspektive Use Case]
    end
    
    subgraph Daten["Dateneigenschaften"]
        B --> D[Target-Typ]
        B --> E[Lernstrategie]
        B --> F[Datenumfang]
        B --> G[Datenqualität]
    end
    
    subgraph UseCase["Use-Case-Anforderungen"]
        C --> H[Performance]
        C --> I[Einfachheit]
        C --> J[Interpretierbarkeit]
        C --> K[Rechenkosten]
        C --> L[Modellvergleich]
    end
    
    style A fill:#4a90d9,stroke:#2c5282,color:#fff
    style B fill:#48bb78,stroke:#276749,color:#fff
    style C fill:#ed8936,stroke:#c05621,color:#fff
```

---

## Perspektive Daten

Die Eigenschaften der verfügbaren Daten bestimmen maßgeblich, welche Modelle überhaupt in Frage kommen.

### Target-Typ: Kategorial vs. Numerisch

Der Typ der Zielvariable ist das erste Auswahlkriterium:

| Target-Typ | Aufgabe | Geeignete Modelle |
|------------|---------|-------------------|
| **Kategorial (binär)** | Klassifikation | Logistische Regression, Decision Tree, Random Forest, SVM |
| **Kategorial (mehrere Klassen)** | Multi-Class-Klassifikation | Random Forest, XGBoost, Neuronale Netze |
| **Numerisch (stetig)** | Regression | Lineare Regression, Random Forest Regressor, XGBoost Regressor |
| **Kein Target** | Clustering/Anomalie | K-Means, DBSCAN, Isolation Forest |

### Lernstrategie

```mermaid
flowchart TD
    A[Gibt es gelabelte Daten?]
    A -->|Ja| B[Supervised Learning]
    A -->|Nein| C[Unsupervised Learning]
    
    B --> D{Target-Typ?}
    D -->|Kategorial| E[Klassifikation]
    D -->|Numerisch| F[Regression]
    
    C --> G{Ziel?}
    G -->|Gruppen finden| H[Clustering]
    G -->|Ausreißer erkennen| I[Anomalieerkennung]
    G -->|Features reduzieren| J[Dimensionsreduktion]
    
    style A fill:#4a90d9,stroke:#2c5282,color:#fff
    style B fill:#48bb78,stroke:#276749,color:#fff
    style C fill:#ed8936,stroke:#c05621,color:#fff
```

### Umfang des Datenbestands

Die Datenmenge beeinflusst die Modellwahl erheblich:

| Datenmenge | Empfohlene Modelle | Begründung |
|------------|-------------------|------------|
| **Klein (<1.000)** | Lineare Modelle, Decision Trees, Ridge/Lasso | Weniger anfällig für Overfitting |
| **Mittel (1.000-100.000)** | Random Forest, XGBoost, SVM | Gute Balance aus Performance und Robustheit |
| **Groß (>100.000)** | XGBoost, Neuronale Netze, LightGBM | Können komplexe Muster lernen |

### Datenqualität: Outlier und Missing Values

Verschiedene Modelle reagieren unterschiedlich auf Datenqualitätsprobleme:

| Problem | Robuste Modelle | Sensible Modelle |
|---------|-----------------|------------------|
| **Ausreißer** | Decision Trees, Random Forest | Lineare Regression, K-Means |
| **Missing Values** | XGBoost (nativ), Random Forest | SVM, Neuronale Netze |
| **Unbalancierte Klassen** | XGBoost (mit scale_pos_weight), Random Forest | Logistische Regression |

---

## Perspektive Use Case

Neben den Daten spielen die Anforderungen des konkreten Anwendungsfalls eine zentrale Rolle.

### Performance: Was ist akzeptabel?

Definieren Sie vorab, welche Metriken relevant sind und welche Schwellenwerte akzeptabel sind:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Performance-Anforderung definieren
min_accuracy = 0.85
min_f1_score = 0.80

# Modelle vergleichen
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    accuracy = cross_val_score(model, data_train, target_train, cv=5, scoring='accuracy').mean()
    f1 = cross_val_score(model, data_train, target_train, cv=5, scoring='f1').mean()
    
    status = "✓" if accuracy >= min_accuracy and f1 >= min_f1_score else "✗"
    print(f"{name}: Accuracy={accuracy:.3f}, F1={f1:.3f} {status}")
```

### Simplicity: Das richtige Maß an Komplexität

> **Grundregel:** Verwenden Sie kein komplexes Modell für ein einfaches Problem.

```mermaid
flowchart LR
    A[Einfaches Problem]
    B[Komplexes Problem]
    
    A --> C[Lineare Modelle<br/>Decision Trees]
    B --> D[Ensemble-Methoden<br/>Neuronale Netze]
    
    C --> E[Schnell trainierbar<br/>Leicht interpretierbar]
    D --> F[Höhere Genauigkeit<br/>Mehr Ressourcen nötig]
    
    style A fill:#48bb78,stroke:#276749,color:#fff
    style B fill:#ed8936,stroke:#c05621,color:#fff
```

**Empfehlung:** Starten Sie immer mit einem einfachen Baseline-Modell und erhöhen Sie die Komplexität nur bei Bedarf:

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Baseline: Einfachstes Modell
baseline = DummyClassifier(strategy='most_frequent')

# 2. Einfach: Lineares Modell
simple_model = LogisticRegression(max_iter=1000)

# 3. Komplex: Ensemble-Modell
complex_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Schrittweise Komplexität erhöhen
for name, model in [('Baseline', baseline), ('Einfach', simple_model), ('Komplex', complex_model)]:
    model.fit(data_train, target_train)
    score = model.score(data_test, target_test)
    print(f"{name}: {score:.3f}")
```

### Interpretability: Nachvollziehbarkeit der Entscheidungen

Je nach Anwendungsbereich ist die Interpretierbarkeit unterschiedlich wichtig:

| Anwendungsbereich | Interpretierbarkeit | Empfohlene Modelle |
|-------------------|--------------------|--------------------|
| **Medizin/Gesundheit** | Sehr hoch | Decision Trees, Logistische Regression |
| **Finanzen/Kredit** | Hoch | Lineare Modelle, Rule-based Systems |
| **Marketing** | Mittel | Random Forest mit Feature Importance |
| **Bilderkennung** | Niedrig akzeptabel | Neuronale Netze, Deep Learning |

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Interpretierbares Modell: Decision Tree
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(data_train, target_train)

# Feature Importance anzeigen
feature_names = data_train.columns if hasattr(data_train, 'columns') else [f'Feature_{i}' for i in range(data_train.shape[1])]
importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance (Top 5):")
print(importance.head())
```

### Computation Costs: Zeit- und Ressourcenbedarf

Das Modell muss in angemessener Zeit trainierbar und mit verfügbarer Hardware anwendbar sein:

| Modell | Trainingszeit | Inferenzzeit | Speicherbedarf |
|--------|--------------|--------------|----------------|
| Logistische Regression | Sehr schnell | Sehr schnell | Gering |
| Decision Tree | Schnell | Sehr schnell | Gering |
| Random Forest | Mittel | Schnell | Mittel |
| XGBoost | Mittel-Lang | Schnell | Mittel |
| Neuronale Netze | Lang | Variabel | Hoch |

```python
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest (100 Trees)': RandomForestClassifier(n_estimators=100, random_state=42),
    'Random Forest (500 Trees)': RandomForestClassifier(n_estimators=500, random_state=42)
}

print("Trainingszeiten vergleichen:")
for name, model in models.items():
    start = time.time()
    model.fit(data_train, target_train)
    train_time = time.time() - start
    
    start = time.time()
    predictions = model.predict(data_test)
    predict_time = time.time() - start
    
    print(f"{name}: Training={train_time:.3f}s, Inferenz={predict_time:.4f}s")
```

### Competition: Systematischer Modellvergleich

Wenn mehrere Modelle die Grundanforderungen erfüllen, hilft ein systematischer Vergleich:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd

# Kandidaten definieren
candidates = {
    'Logistische Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

# Systematischer Vergleich
results = []
for name, model in candidates.items():
    accuracy = cross_val_score(model, data_train, target_train, cv=5, scoring='accuracy')
    results.append({
        'Modell': name,
        'Accuracy (Mean)': accuracy.mean(),
        'Accuracy (Std)': accuracy.std()
    })

comparison = pd.DataFrame(results).sort_values('Accuracy (Mean)', ascending=False)
print(comparison.to_string(index=False))
```

---

## Entscheidungsbaum zur Modellauswahl

Der folgende Entscheidungsbaum bietet eine praktische Orientierungshilfe:

```mermaid
flowchart TD
    A[Start: Modellauswahl]
    A --> B{Gelabelte<br/>Daten?}
    
    B -->|Ja| C{Target-Typ?}
    B -->|Nein| D[Unsupervised]
    
    C -->|Kategorial| E{Interpretierbarkeit<br/>wichtig?}
    C -->|Numerisch| F{Lineare<br/>Beziehung?}
    
    E -->|Ja| G[Decision Tree<br/>Logistische Regression]
    E -->|Nein| H{Datenmenge?}
    
    H -->|Klein| I[Random Forest<br/>SVM]
    H -->|Groß| J[XGBoost<br/>Neural Networks]
    
    F -->|Ja| K[Lineare Regression<br/>Ridge/Lasso]
    F -->|Nein| L[Random Forest<br/>XGBoost Regressor]
    
    D --> M{Ziel?}
    M -->|Gruppierung| N[K-Means<br/>DBSCAN]
    M -->|Anomalien| O[Isolation Forest]
    M -->|Dimensionsreduktion| P[PCA<br/>LDA]
    
    style A fill:#4a90d9,stroke:#2c5282,color:#fff
    style G fill:#48bb78,stroke:#276749,color:#fff
    style I fill:#48bb78,stroke:#276749,color:#fff
    style J fill:#48bb78,stroke:#276749,color:#fff
    style K fill:#48bb78,stroke:#276749,color:#fff
    style L fill:#48bb78,stroke:#276749,color:#fff
    style N fill:#48bb78,stroke:#276749,color:#fff
    style O fill:#48bb78,stroke:#276749,color:#fff
    style P fill:#48bb78,stroke:#276749,color:#fff
```

---

## Best Practices

### Do's ✓

- **Baseline zuerst:** Starten Sie immer mit einem einfachen Modell als Referenz
- **Cross-Validation nutzen:** Verwenden Sie CV für robuste Leistungsschätzungen
- **Mehrere Metriken:** Bewerten Sie Modelle anhand mehrerer relevanter Metriken
- **Dokumentieren:** Halten Sie fest, warum Sie sich für ein Modell entschieden haben
- **Iterativ vorgehen:** Erhöhen Sie die Komplexität schrittweise

### Don'ts ✗

- **Nicht das komplexeste Modell wählen:** Komplexität ≠ bessere Performance
- **Nicht nur Accuracy betrachten:** Besonders bei unbalancierten Daten irreführend
- **Nicht blind kopieren:** Was bei anderen funktioniert, passt nicht automatisch zu Ihren Daten
- **Nicht Trainingszeit ignorieren:** Produktionsanforderungen berücksichtigen
- **Nicht Interpretierbarkeit vernachlässigen:** Stakeholder müssen Ergebnisse verstehen

---

## Zusammenfassung

Die Modellauswahl basiert auf zwei Perspektiven:

**Perspektive Daten:**
- Target-Typ (kategorial vs. numerisch)
- Lernstrategie (supervised vs. unsupervised)
- Umfang des Datenbestands
- Datenqualität (Outlier, Missing Values)

**Perspektive Use Case:**
- Performance-Anforderungen
- Einfachheit (Simplicity)
- Interpretierbarkeit
- Rechenkosten
- Systematischer Modellvergleich

> **Merke:** Das beste Modell ist nicht das komplexeste, sondern das, welches die Anforderungen mit der geringsten Komplexität erfüllt.


---

**Version:** 1.0    
**Stand:** Januar 2026    
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.    