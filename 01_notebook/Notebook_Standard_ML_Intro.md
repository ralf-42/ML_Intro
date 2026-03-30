# Notebook-Standard ML_Intro

> Projektspezifischer Standard für den produktiven Notebook-Bestand in `ML_Intro/01_notebook`.

Dieser Standard leitet sich aus den vorhandenen Haupt-Notebooks in `00_general` bis `09_diverse` ab. `_misc` und Fremd-/Archivbestände sind ausdrücklich nicht Referenz. Ziel ist kein Bruch mit dem existierenden Material, sondern eine belastbare Vereinheitlichung des tatsächlichen Kursstils.

---

## 1. Geltungsbereich

Dieser Standard gilt für produktive Kurs-Notebooks in:

- `00_general`
- `01_supervised`
- `02_unsupervised`
- `03_network`
- `04_ensemble`
- `05_tuning`
- `06_workflow`
- `07_special`
- `09_diverse`

Nicht Teil des Standards:

- `_misc`
- `.ipynb_checkpoints`
- importierte oder fremde Demo-Notebooks

---

## 2. Grundprinzip

`ML_Intro` ist ein klassischer Machine-Learning-Kurs mit starker Workflow-Logik. Notebooks sollen deshalb vor allem drei Dinge leisten:

1. Ein Problem fachlich einordnen.
2. Den Weg von Daten zu Modell und Bewertung nachvollziehbar machen.
3. Die Umsetzung an einem konkreten Datensatz zeigen.

Die Notebooks sind keine Agenten-Tutorials. Vorgaben aus dem Agenten-Projekt gelten nur dort, wo sie für ML_Intro wirklich passen.

---

## 3. Minimalstruktur

### 3.1 Header

Die erste Markdown-Zelle enthält Kurstitel und Notebook-Titel im etablierten HTML-Stil:

```html
<p><font size="6" color='grey'> <b>
Machine Learning
</b></font> </br></p>
<p><font size="5" color='grey'> <b>
[Notebook-Titel]
</b></font> </br></p>

---
```

Regel:

- Kurstitel und Notebook-Titel dürfen in derselben Markdown-Zelle stehen.
- Der bestehende graue HTML-Stil ist Projektstandard.

### 3.2 Kapitelstruktur

Der Standardaufbau orientiert sich am dominanten Muster im Bestand:

```markdown
# 0 | Install & Import

# 1 | Understand

# 2 | Prepare

# 3 | Modeling

# 4 | Evaluate

# 5 | Deploy
```

Regeln:

- Kapitel folgen dem Format `# [Nummer] | [Titel]`.
- Hauptkapitel werden immer mit `#` gesetzt.
- Die Kapitel `1` bis `5` bilden den didaktischen Kern des Kurses.
- `0 | Install & Import` ist üblich, aber nicht in jedem Notebook zwingend.
- Nicht jedes Notebook braucht jedes Kapitel, die Logik soll jedoch erkennbar bleiben.

### 3.3 Unterkapitel

Unterkapitel folgen im Projekt nicht als weitere Markdown-Überschrift, sondern als schwarzer HTML-Block:

```html
<p><font color='black' size="5">
[Unterkapitel]
</font></p>
```

Regeln:

- Unterkapitel werden mit `<p><font color='black' size="5"> ... </font></p>` gesetzt.
- Unterkapitel strukturieren Abschnitte innerhalb eines Hauptkapitels.
- Die Kombination aus `#` für Hauptkapitel und HTML-Block für Unterkapitel ist der projektspezifische Standard für `ML_Intro`.

### 3.4 Optionale Abweichungen

Folgende Abweichungen sind zulässig, wenn der Notebook-Typ es erfordert:

- Spezial- oder Analyse-Notebooks mit reduziertem Kapitelumfang
- Themenblöcke mit mehreren Datensätzen oder Vergleichsfällen
- Notebooks mit stärkerem Tool- oder Explorationscharakter in `00_general` und `09_diverse`

Nicht mehr als Standardbruch gewertet:

- keine eigene `# A | Aufgabe`-Section
- keine verpflichtende `🛠️ Umgebung einrichten`-Zelle

---

## 4. Inhaltliche Funktion der Kapitel

### `1 | Understand`

Enthält Problemtyp, Datensatzkontext, Zielvariable, fachliche Einordnung und typische Risiken. Hier wird erklärt, worum es geht und warum das Problem so modelliert wird.

### `2 | Prepare`

Enthält Datenzugriff, Bereinigung, Feature-Auswahl, Kodierung, Skalierung, Split und ähnliche Vorverarbeitungsschritte.

### `3 | Modeling`

Enthält die Modellerstellung inklusive Parameterwahl, Trainingslogik und gegebenenfalls Modellvergleich.

### `4 | Evaluate`

Enthält Metriken, Visualisierungen, Fehleranalyse und die Einordnung, ob das Modell brauchbar ist.

### `5 | Deploy`

Enthält Ausblick, Speicherung, Inferenz, App-/Demo-Bezug oder Transfer in eine praktisch nutzbare Form. In vielen ML_Intro-Notebooks ist dies eher ein Abschluss- oder Transferkapitel als ein echtes Produktionsdeployment.

---

## 5. Code-Stil

### 5.1 Installationen und Imports

Der reale Projektstandard ist schlicht und klassisch:

```python
# Install
!uv pip install ...

# Import
from pandas import read_excel
from sklearn.model_selection import train_test_split
```

Regeln:

- `uv pip install` ist erlaubt und bevorzugt, wenn zusätzliche Pakete nötig sind.
- Install- und Import-Blöcke dürfen getrennt bleiben.
- Ein Agenten-Setup mit `#@title` oder `setup_api_keys()` ist für ML_Intro nicht verpflichtend.

### 5.2 Reihenfolge

Die Code-Logik folgt dem ML-Workflow:

1. Daten laden
2. Daten vorbereiten
3. Modell trainieren
4. Modell bewerten
5. Ergebnis interpretieren

### 5.3 Bibliotheken

Typische Bibliotheken des Kurses:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`
- `xgboost`
- `tensorflow` / `keras`
- fallweise `gradio`, `pycaret`, `shap`, `lime`

---

## 6. Markdown-Qualität

### 6.1 Ton

Die Notebooks sollen sachlich, knapp und lehrbar formuliert sein. Direkte Ansprache wird vermieden.

Zu vermeiden:

- `Sie können ...`
- `Verwenden Sie ...`
- `Stellen Sie sich vor ...`

Besser:

- `Verwendet wird ...`
- `Zum Einsatz kommt ...`
- `Als Analogie dient ...`

### 6.2 Fließtext vor Stichpunkten

Listen sind erlaubt, aber Erklärungen gehören bevorzugt in kurze Absätze. Aufzählungen dienen der Struktur, nicht als Ersatz für Formulierung.

### 6.3 Grenzen benennen

Jedes zentrale Konzept soll mindestens einen Satz zu Grenzen, Fehlerquellen oder typischen Missverständnissen enthalten. Das ist für ML_Intro wichtiger als Werbesprache oder Vollständigkeitsanspruch.

---

## 7. Visualisierung

Mermaid ist in ML_Intro derzeit nicht gelebter Projektstandard und daher nicht verpflichtend. Visualisierung erfolgt primär über:

- `matplotlib`
- `seaborn`
- `plotly`
- modellnahe Diagramme und Metrikplots

Empfehlung:

- Prozess- oder Entscheidungslogik darf künftig mit Mermaid ergänzt werden.
- Für bestehende Notebook-Serien ist das eine Qualitätsverbesserung, aber keine Mindestanforderung.

---

## 8. Callouts und Hervorhebungen

GitHub-Alert-Callouts sind in ML_Intro derzeit nicht etabliert und deshalb nicht verpflichtend.

Erlaubte Hervorhebungen im Bestand:

- kurze Markdown-Hinweise
- fette Zwischenüberschriften
- Warn- oder Infozeilen im Text

Empfehlung:

- Bei Überarbeitungen sparsam standardisierte Callouts einführen, aber nicht rückwirkend als Pflicht behandeln.

---

## 9. Notebook-Typen im Kurs

### 9.1 `00_general`

Orientierung, Datenzugang, Werkzeuge, Snippets, Grundlagen.

### 9.2 `01_supervised`

Klassische Supervised-Learning-Fälle wie Decision Tree oder lineare/logistische Regression.

### 9.3 `02_unsupervised`

Clustering, Anomalieerkennung, Apriori, PCA.

### 9.4 `03_network`

MLP- und Keras-basierte neuronale Netze.

### 9.5 `04_ensemble`

Random Forest, XGBoost, Stacking.

### 9.6 `05_tuning`

Cross-Validation, Bootstrapping, Hyperparameter-Tuning, Thresholds.

### 9.7 `06_workflow`

Pipeline- und Workflow-Logik.

### 9.8 `07_special`

Vision, NLP, Zeitreihen, Autoencoder und andere Spezialthemen.

### 9.9 `09_diverse`

XAI, Apps, Save/Load, KI-gestützte Datenanalyse.

---

## 10. Mindeststandard für neue oder überarbeitete Notebooks

Ein neues produktives Notebook in ML_Intro erfüllt mindestens diese Punkte:

- Header im bestehenden HTML-Stil mit `Machine Learning`
- Hauptkapitel im Format `# N | Titel`
- Unterkapitel im Format `<p><font color='black' size="5"> ... </font></p>`
- Workflow-Logik von Problem zu Bewertung nachvollziehbar
- Markdown ohne direkte Ansprache
- klare Trennung von Vorbereitung, Modellierung und Evaluation
- fachliche Einordnung des Datensatzes und der Modellwahl
- mindestens ein Satz zu Grenzen, Fehlerquellen oder typischen Fehlinterpretationen

Empfohlen, aber nicht verpflichtend:

- Mermaid für Prozesslogik
- standardisierte Callouts
- expliziter Übungs- oder Transferblock am Ende

---

## 11. Kurzcheck vor der Freigabe

Vor der Ablage eines neuen oder überarbeiteten Notebooks:

1. Ist sofort erkennbar, welches ML-Problem bearbeitet wird?
2. Folgen die Kapitel der Logik `Understand -> Prepare -> Modeling -> Evaluate`?
3. Ist die Sprache sachlich und ohne direkte Ansprache formuliert?
4. Wird nicht nur gezeigt, dass etwas funktioniert, sondern auch, wann es problematisch wird?
5. Ist das Notebook klar einem Kursbereich zuordenbar?

---

**Version:** 1.0  
**Stand:** März 2026  
**Projekt:** ML_Intro
