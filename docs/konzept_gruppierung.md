# Konzept: GitHub Pages Gruppierung für ML_Intro

*Erstellt: 2026-01-15*

---

## Aktuelle Struktur (docs/)

```
├── index.md
├── concepts.md          → concepts/01_grundlagen, 02_prepare, 03_modeling, 04_evaluate, 05_deployment, 08_xai
├── ressourcen.md        → ressourcen/links, interaktive-visualisierung
├── frameworks.md
├── projekte.md
├── regulatorisches.md   → EU AI Act, Ethik
└── legal/
```

---

## Vorschlag: 3 Optionen zur Gruppierung

### Option A: Nach ML-Workflow (5 Phasen)

*Orientiert sich am bereits etablierten Workflow-Konzept*

```
📚 Grundlagen
   ├── ML Einführung
   ├── Python/Pandas Basics
   └── Datasets-Übersicht

🔍 1. Understand (Daten verstehen)
   ├── Explorative Datenanalyse
   └── Datenvisualisierung

🔧 2. Prepare (Daten vorbereiten)
   ├── Fehlende Werte
   ├── Outlier-Behandlung
   ├── Kategoriale Kodierung
   ├── Skalierung
   └── Feature Engineering

🤖 3. Model (Modellierung)
   ├── Supervised Learning
   │   ├── Decision Trees
   │   ├── Random Forests
   │   ├── Lineare/Logistische Regression
   │   └── XGBoost
   ├── Unsupervised Learning
   │   ├── Clustering (K-means, DBSCAN)
   │   ├── PCA
   │   └── Association Rules
   └── Deep Learning
       ├── Neuronale Netze Basics
       ├── CNN (Computer Vision)
       ├── RNN/LSTM (Time Series)
       └── NLP

📊 4. Evaluate (Bewerten)
   ├── Metriken (Klassifikation/Regression)
   ├── Cross-Validation
   ├── Hyperparameter-Tuning
   └── Overfitting vermeiden

🚀 5. Deploy (Bereitstellen)
   ├── Model Persistence
   ├── Pipelines
   ├── Web Apps (Gradio)
   └── XAI (Explainability)
```

**Vorteile:** Didaktisch sinnvoll, folgt dem natürlichen Lernpfad
**Nachteile:** Algorithmen verteilt über mehrere Kategorien

---

### Option B: Nach Algorithmus-Kategorien

*Klassische ML-Lehrbuch-Struktur*

```
📚 Grundlagen
   ├── ML Einführung & Begriffe
   ├── Workflow-Übersicht
   └── Datenvorbereitung (alle Prepare-Themen)

🎯 Supervised Learning
   ├── Klassifikation
   │   ├── Decision Trees
   │   ├── Random Forests
   │   ├── Logistische Regression
   │   └── XGBoost
   └── Regression
       ├── Lineare Regression
       ├── Random Forest Regression
       └── XGBoost Regression

🔮 Unsupervised Learning
   ├── Clustering
   │   ├── K-means
   │   └── DBSCAN
   ├── Dimensionsreduktion (PCA)
   └── Association Rules (Apriori)

🧠 Deep Learning
   ├── Neuronale Netze Grundlagen
   ├── CNN (Computer Vision)
   ├── RNN/LSTM (Time Series)
   └── Autoencoders

⚙️ Model Engineering
   ├── Hyperparameter-Tuning
   ├── Cross-Validation
   ├── Ensemble Methods
   └── Pipelines

🚀 Deployment & Produktion
   ├── Model Persistence
   ├── Web Apps
   └── XAI
```

**Vorteile:** Intuitive Suche nach Algorithmen, Referenzcharakter
**Nachteile:** Weniger workflow-orientiert

---

### Option C: Hybrid (Workflow + Algorithmus-Referenz) ⭐ EMPFOHLEN

*Kombination beider Ansätze*

```
🎯 Lernpfad (Workflow)
   ├── 1. Daten verstehen
   ├── 2. Daten vorbereiten
   ├── 3. Modell trainieren
   ├── 4. Modell bewerten
   └── 5. Modell deployen

📖 Algorithmen-Referenz
   ├── Supervised Learning
   │   ├── Decision Trees
   │   ├── Random Forests
   │   ├── Regression
   │   └── Gradient Boosting
   ├── Unsupervised Learning
   │   ├── Clustering
   │   └── Dimensionsreduktion
   └── Deep Learning
       ├── Grundlagen
       └── Spezialisierungen

🛠️ Techniken & Tools
   ├── Evaluation & Metriken
   ├── Hyperparameter-Tuning
   ├── Pipelines
   ├── XAI
   └── Deployment

📚 Ressourcen
   ├── Interaktive Tools
   ├── Externe Links
   └── Datasets
```

**Vorteile:** Sowohl Lernpfad als auch Nachschlagewerk
**Nachteile:** Etwas komplexere Navigation

---

## Empfehlung: Option C (Hybrid)

### Begründung

1. Der **Lernpfad** unterstützt Einsteiger mit strukturiertem Vorgehen
2. Die **Algorithmen-Referenz** dient als Nachschlagewerk für spezifische Themen
3. **Techniken & Tools** bündelt querschnittliche Themen (unabhängig vom Algorithmus)
4. Entspricht der vorhandenen Notebook-Struktur (Module 00-09)

### Mapping zu vorhandenen Modulen

| Navigation | Notebook-Module |
|------------|-----------------|
| Lernpfad | Modul 00, 05, 06 |
| Supervised | Modul 01, 04 |
| Unsupervised | Modul 02 |
| Deep Learning | Modul 03, 07 |
| Techniken | Modul 05, 09 |

---

## Status

- [ ] Option auswählen
- [ ] Navigation anpassen
- [ ] Seiten umstrukturieren
- [ ] Links aktualisieren
