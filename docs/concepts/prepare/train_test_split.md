---
layout: default
title: Train-Test-Split
parent: Prepare
grand_parent: Konzepte
nav_order: 5
description: "Train-Test-Split - Aufteilung von Daten in Trainings- und Testsets für Machine Learning"
has_toc: true
---

# Train-Test-Split
{: .no_toc }

> **Aufteilung von Daten für Training und Evaluation**
> 80-20-Split, Stratified Split, Random State und Best Practices

---

## Überblick

Der Train-Test-Split ist ein fundamentales Verfahren im Machine Learning, um die Leistungsfähigkeit von Modellen zu überprüfen. Das Prinzip ist einfach: Die verfügbaren Daten werden in zwei separate Mengen aufgeteilt – eine zum Trainieren des Modells und eine zum Testen seiner Vorhersagefähigkeit.

```mermaid
flowchart LR
    subgraph input[" "]
        D[("Dataset")]
    end
    
    subgraph split["Train-Test-Split"]
        D --> |"z.B. 80%"| TR[("Training Set")]
        D --> |"z.B. 20%"| TE[("Test Set")]
    end
    
    subgraph usage["Verwendung"]
        TR --> M["Modell<br/>Training"]
        M --> P["Trainiertes<br/>Modell"]
        P --> E["Evaluation"]
        TE --> E
    end
    
    style D fill:#e1f5fe
    style TR fill:#c8e6c9
    style TE fill:#ffecb3
    style P fill:#f3e5f5
```

## Warum ist der Split notwendig?

Das zentrale Ziel eines Machine-Learning-Modells ist die **Generalisierungsfähigkeit** – die Fähigkeit, auf neuen, unbekannten Daten gute Vorhersagen zu treffen. Ein Modell, das nur auf bereits gesehenen Daten gut funktioniert, hat keinen praktischen Nutzen.

**Das Problem ohne Split:**
- Ein Modell könnte die Trainingsdaten auswendig lernen (Overfitting)
- Die gemessene Leistung wäre unrealistisch hoch
- Im Produktiveinsatz würde das Modell versagen

**Die Lösung mit Split:**
- Das Modell wird nur mit Trainingsdaten trainiert
- Die Testdaten simulieren "neue, unbekannte" Daten
- Die Evaluation auf Testdaten zeigt die realistische Leistung

## Gängige Aufteilungsverhältnisse

| Verhältnis | Training | Test | Typischer Einsatz |
|------------|----------|------|-------------------|
| **80-20** | 80% | 20% | Standardwahl für die meisten Datensätze |
| **70-30** | 70% | 30% | Bei kleineren Datensätzen oder höherem Validierungsbedarf |
| **60-40** | 60% | 40% | Bei sehr kleinen Datensätzen |
| **90-10** | 90% | 10% | Bei sehr großen Datensätzen (>100.000 Samples) |

**Faustregel:** Je größer der Datensatz, desto kleiner kann der Testanteil sein, da auch ein kleinerer Prozentsatz noch statistisch aussagekräftig ist.

## Implementierung mit scikit-learn

### Grundlegende Verwendung

```python
from sklearn.model_selection import train_test_split

# Beispieldaten
data = df.drop('target', axis=1)  # Features
target = df['target']              # Zielvariable

# Split durchführen
data_train, data_test, target_train, target_test = train_test_split(
    data, target,
    test_size=0.2,      # 20% für Test
    random_state=42     # Reproduzierbarkeit
)

print(f"Training: {len(data_train)} Samples")
print(f"Test:     {len(data_test)} Samples")
```

### Wichtige Parameter

```python
data_train, data_test, target_train, target_test = train_test_split(
    data, target,
    test_size=0.2,       # Anteil oder absolute Anzahl für Test
    train_size=None,     # Optional: Anteil für Training
    random_state=42,     # Seed für Reproduzierbarkeit
    shuffle=True,        # Daten vor Split mischen
    stratify=target      # Klassenverteilung beibehalten
)
```

### Stratifizierter Split bei Klassifikation

Bei unausgewogenen Klassen sollte der Split die Klassenverteilung in beiden Mengen beibehalten:

```python
# Ohne Stratifizierung (problematisch bei unausgewogenen Klassen)
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

# Mit Stratifizierung (empfohlen für Klassifikation)
data_train, data_test, target_train, target_test = train_test_split(
    data, target,
    test_size=0.2,
    stratify=target,     # Klassenverteilung beibehalten
    random_state=42
)

# Überprüfung der Verteilung
print("Original:", target.value_counts(normalize=True))
print("Training:", target_train.value_counts(normalize=True))
print("Test:    ", target_test.value_counts(normalize=True))
```

## Train-Validate-Test-Split

Für eine robustere Modellentwicklung wird oft ein dreifacher Split verwendet:

```mermaid
flowchart TB
    subgraph phase1["Phase 1: Initiale Aufteilung"]
        D[("Dataset")] --> TR1[("Training + Validation<br/>80%")]
        D --> TE[("Test Set<br/>20%")]
    end
    
    subgraph phase2["Phase 2: Weitere Aufteilung"]
        TR1 --> TR2[("Training Set<br/>64%")]
        TR1 --> VA[("Validation Set<br/>16%")]
    end
    
    subgraph usage["Verwendung"]
        TR2 --> |"Modell trainieren"| M["Training"]
        VA --> |"Hyperparameter tunen<br/>Modell auswählen"| M
        M --> P["Finales Modell"]
        TE --> |"Finale Bewertung<br/>nur einmal!"| P
    end
    
    style D fill:#e1f5fe
    style TR2 fill:#c8e6c9
    style VA fill:#fff9c4
    style TE fill:#ffecb3
    style P fill:#f3e5f5
```

### Zweck der drei Mengen

| Menge | Zweck | Verwendung |
|-------|-------|------------|
| **Training Set** | Modell trainieren | Lernen der Muster und Parameter |
| **Validation Set** | Modell optimieren | Hyperparameter-Tuning, Modellauswahl |
| **Test Set** | Finale Bewertung | Einmalige, unvoreingenommene Evaluation |

### Implementierung

```python
from sklearn.model_selection import train_test_split

# Erster Split: Trainings-/Validierungs-Set vs. Test-Set
data_temp, data_test, target_temp, target_test = train_test_split(
    data, target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

# Zweiter Split: Training-Set vs. Validation-Set
data_train, data_val, target_train, target_val = train_test_split(
    data_temp, target_temp,
    test_size=0.2,       # 20% von 80% = 16% des Originals
    random_state=42,
    stratify=target_temp
)

print(f"Training:   {len(data_train)} Samples ({len(data_train)/len(data)*100:.1f}%)")
print(f"Validation: {len(data_val)} Samples ({len(data_val)/len(data)*100:.1f}%)")
print(f"Test:       {len(data_test)} Samples ({len(data_test)/len(data)*100:.1f}%)")
```

## Data Leakage vermeiden

**Data Leakage** bezeichnet die unbeabsichtigte Verwendung von Informationen aus dem Test-Set während des Trainings. Dies führt zu unrealistisch guten Ergebnissen, die sich nicht in der Praxis bestätigen.

```mermaid
flowchart TB
    subgraph wrong["❌ Falsch: Data Leakage"]
        D1[("Dataset")] --> S1["Skalierung/<br/>Encoding"]
        S1 --> SP1["Split"]
        SP1 --> TR1["Training"]
        SP1 --> TE1["Test"]
    end
    
    subgraph correct["✅ Richtig: Kein Data Leakage"]
        D2[("Dataset")] --> SP2["Split"]
        SP2 --> TR2["Training"]
        SP2 --> TE2["Test"]
        TR2 --> S2["Fit & Transform<br/>(nur Training)"]
        S2 --> M["Modell<br/>Training"]
        TE2 --> S3["Transform<br/>(mit Training-Params)"]
        S3 --> E["Evaluation"]
        M --> E
    end
    
    style wrong fill:#ffcdd2
    style correct fill:#c8e6c9
```

### Häufige Quellen von Data Leakage

1. **Skalierung vor dem Split**
   ```python
   # ❌ FALSCH: Skalierung auf gesamtem Dataset
   scaler = StandardScaler()
   data_scaled = scaler.fit_transform(data)  # Lernt von allen Daten!
   data_train, data_test, target_train, target_test = train_test_split(data_scaled, target)

   # ✅ RICHTIG: Skalierung nur auf Training-Daten
   data_train, data_test, target_train, target_test = train_test_split(data, target)
   scaler = StandardScaler()
   data_train_scaled = scaler.fit_transform(data_train)  # Nur Training
   data_test_scaled = scaler.transform(data_test)        # Nur Transform!
   ```

2. **Imputation vor dem Split**
   ```python
   # ❌ FALSCH: Mittelwert aus allen Daten
   data['feature'].fillna(data['feature'].mean(), inplace=True)

   # ✅ RICHTIG: Mittelwert nur aus Training-Daten
   train_mean = data_train['feature'].mean()
   data_train['feature'].fillna(train_mean, inplace=True)
   data_test['feature'].fillna(train_mean, inplace=True)
   ```

3. **Feature Engineering vor dem Split**
   ```python
   # ❌ FALSCH: Encoding auf gesamtem Dataset
   encoder = LabelEncoder()
   df['category_encoded'] = encoder.fit_transform(df['category'])
   
   # ✅ RICHTIG: Mit Pipeline nach dem Split
   ```

### Lösung: Pipelines verwenden

Pipelines in scikit-learn verhindern Data Leakage automatisch:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Pipeline definieren
model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Split durchführen
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

# Pipeline auf Training-Daten fitten
model.fit(data_train, target_train)  # Alle Schritte lernen nur von data_train

# Vorhersage auf Test-Daten
target_pred = model.predict(data_test)  # Transformation mit Training-Parametern
```

## Workflow: Vollständiges Beispiel

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Daten laden
df = pd.read_csv('daten.csv')
data = df.drop('target', axis=1)
target = df['target']

# 2. Feature-Typen identifizieren
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object', 'category']).columns

# 3. Preprocessing-Pipeline definieren
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. Vollständige Pipeline mit Modell
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. Train-Test-Split
data_train, data_test, target_train, target_test = train_test_split(
    data, target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

# 6. Training
model.fit(data_train, target_train)

# 7. Evaluation
target_pred = model.predict(data_test)
print(classification_report(target_test, target_pred))
```

## Best Practices

### Dos ✅

- **Immer `random_state` setzen** für Reproduzierbarkeit
- **Stratifizierung verwenden** bei Klassifikation mit unausgewogenen Klassen
- **Pipelines nutzen** um Data Leakage zu vermeiden
- **Test-Set nur einmal verwenden** für die finale Bewertung
- **Shuffle aktiviert lassen** (Standard), außer bei Zeitreihen

### Don'ts ❌

- **Niemals auf Test-Daten trainieren** oder Hyperparameter tunen
- **Keine Vorverarbeitung vor dem Split** (Skalierung, Encoding, Imputation)
- **Test-Set nicht mehrfach verwenden** für Modellauswahl
- **Shuffle nicht bei Zeitreihen** – zeitliche Reihenfolge beachten

## Sonderfälle

### Zeitreihen-Daten

Bei Zeitreihen darf nicht zufällig gesplittet werden:

```python
# ❌ FALSCH für Zeitreihen
data_train, data_test = train_test_split(data, shuffle=True)

# ✅ RICHTIG für Zeitreihen: Chronologischer Split
split_index = int(len(data) * 0.8)
data_train, data_test = data[:split_index], data[split_index:]
target_train, target_test = target[:split_index], target[split_index:]

# Oder mit TimeSeriesSplit für Cross-Validation
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### Sehr kleine Datensätze

Bei kleinen Datensätzen (<1000 Samples) ist Cross-Validation oft besser:

```python
from sklearn.model_selection import cross_val_score

# Statt einem einzelnen Split
scores = cross_val_score(model, data, target, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

### Gruppierte Daten

Wenn Samples zu Gruppen gehören (z.B. mehrere Messungen pro Patient):

```python
from sklearn.model_selection import GroupShuffleSplit

# Sicherstellen, dass alle Samples einer Gruppe im selben Set landen
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(data, target, groups=patient_ids))

data_train, data_test = data.iloc[train_idx], data.iloc[test_idx]
target_train, target_test = target.iloc[train_idx], target.iloc[test_idx]
```

## Zusammenfassung

```mermaid
flowchart TB
    subgraph entscheidung["Entscheidungsbaum: Welcher Split?"]
        Q1{Zeitreihen-<br/>Daten?}
        Q1 -->|Ja| A1["Chronologischer Split<br/>oder TimeSeriesSplit"]
        Q1 -->|Nein| Q2{Gruppierte<br/>Daten?}
        Q2 -->|Ja| A2["GroupShuffleSplit"]
        Q2 -->|Nein| Q3{Sehr kleine<br/>Datenmenge?}
        Q3 -->|Ja| A3["Cross-Validation<br/>statt einfachem Split"]
        Q3 -->|Nein| Q4{Klassifikation mit<br/>unausgewogenen<br/>Klassen?}
        Q4 -->|Ja| A4["Stratifizierter<br/>Train-Test-Split"]
        Q4 -->|Nein| A5["Standard<br/>Train-Test-Split"]
    end
    
    style A1 fill:#e3f2fd
    style A2 fill:#e3f2fd
    style A3 fill:#e3f2fd
    style A4 fill:#e3f2fd
    style A5 fill:#e3f2fd
```

Der Train-Test-Split ist die Grundlage für zuverlässige Modellbewertung. Die korrekte Anwendung – insbesondere die Vermeidung von Data Leakage – ist entscheidend für aussagekräftige Ergebnisse und erfolgreiche ML-Projekte.

## Weiterführende Themen

- **Cross-Validation**: Robustere Bewertung durch mehrfache Splits
- **Nested Cross-Validation**: Kombination von Modellauswahl und Evaluation
- **Bootstrapping**: Resampling-Technik für Konfidenzintervalle

---

*Referenzen:*
- scikit-learn Dokumentation: [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- KNIME: Data Preprocessing for Machine Learning Part 1 & Part 2

---

**Version:** 1.0
**Stand:** Januar 2026
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.
