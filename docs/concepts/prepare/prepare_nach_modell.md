---
layout: default
title: Prepare nach Modell
parent: Prepare
grand_parent: Konzepte
nav_order: 7
description: "Welche Vorverarbeitungsschritte fuer die im Kurs eingesetzten ML-Algorithmen notwendig sind"
has_toc: true
---

# Prepare nach Modell
{: .no_toc }

> **Nicht jedes Modell braucht die gleiche Vorverarbeitung.**
> Diese Seite ordnet die im Kurs eingesetzten Algorithmen danach ein, welche Prepare-Schritte zwingend, wichtig oder nur fallweise relevant sind.

---

## Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Zuerst lesen

Vor dieser Seite sollten diese Grundlagen klar sein:

| Dokument | Warum zuerst? |
|----------|---------------|
| [Train-Test-Split](./train_test_split.html) | Alle lernenden Vorverarbeitungsschritte muessen nur auf Trainingsdaten gefittet werden. |
| [Missing Values](./missing_values.html) | Fehlende Werte sind je nach Modell erlaubt oder muessen imputiert werden. |
| [Kodierung](./kodierung_kategorialer_daten.html) | Die meisten Kursmodelle erwarten numerische Eingaben. |
| [Skalierung](./skalierung.html) | Distanz-, Gradienten- und Varianzverfahren reagieren stark auf unterschiedliche Wertebereiche. |
| [Outlier](./outlier.html) | Ausreisser sind kein Pflichtschritt fuer jedes Modell, koennen aber mehrere Verfahren stark verzerren. |

## Lesepfad

Empfohlene Reihenfolge fuer den Prepare-Block:

1. [Train-Test-Split](./train_test_split.html)
2. [Missing Values](./missing_values.html)
3. [Kodierung](./kodierung_kategorialer_daten.html)
4. [Skalierung](./skalierung.html)
5. [Outlier](./outlier.html)
6. [Feature Engineering](./feature-engineering.html)
7. **Prepare nach Modell**

Diese Seite ist als Entscheidungshilfe gedacht: Erst die einzelnen Techniken verstehen, dann pro Modell entscheiden, welche davon wirklich gebraucht werden.

## Legende

| Eintrag | Bedeutung |
|---------|-----------|
| **Ja** | In der Regel notwendig oder sehr empfehlenswert. |
| **Nein** | Normalerweise nicht erforderlich. |
| **Bedingt** | Haengt von Daten, Implementierung, Pipeline oder Ziel ab. |
| **Kritisch** | Ohne diesen Schritt sind Ergebnisse oft deutlich schlechter oder instabil. |

## Ueberblick

| Algorithmus im Kurs | Fehlende Werte erlaubt? | Skalierung noetig? | Kodierung noetig? | Ausreisser kritisch? | Besondere Prepare-Hinweise |
|---------------------|-------------------------|--------------------|-------------------|----------------------|----------------------------|
| Decision Tree Classifier/Regressor | Ja, in aktuellen scikit-learn-Versionen | Nein | Ja | Eher nein | Fuer aeltere Versionen oder Pipeline-Schritte trotzdem imputieren. |
| Random Forest Classifier/Regressor | Ja, in aktuellen scikit-learn-Versionen | Nein | Ja | Eher nein | Robust, aber Feature-Auswahl und saubere Kodierung bleiben sinnvoll. |
| XGBoost Classifier/Regressor | Ja | Nein | Ja | Mittel | Missing Values werden intern behandelt; Kategorien im Kurs meist vorher kodieren. |
| Explainable Boosting Classifier | Bedingt | Nein | Ja | Mittel | Explainability profitiert von sauber benannten, stabilen Features. |
| Linear Regression | Nein | Meist ja | Ja | Ja | Ausreisser und Multikollinearitaet besonders pruefen. |
| Logistic Regression | Nein | Ja | Ja | Ja | Skalierung ist besonders bei Regularisierung wichtig. |
| KNN | Nein | Kritisch | Ja | Ja | Distanzbasiert: Skalierung und Ausreisserbehandlung stark relevant. |
| Linear SVC / SVM | Nein | Kritisch | Ja | Ja | Ohne Skalierung dominieren Merkmale mit grossem Wertebereich. |
| MLP Classifier/Regressor | Nein | Ja | Ja | Ja | Skalierung stabilisiert Optimierung und Konvergenz. |
| Keras Dense-Netze | Nein | Ja | Ja | Ja | Eingaben numerisch, skaliert und als Arrays/Tensoren vorbereiten. |
| Keras CNN | Nein | Ja | Bedingt | Mittel | Bilddaten skalieren/normalisieren; Kategorien meist als Labels oder One-Hot-Ziele. |
| Keras LSTM / Sequenzmodelle | Nein | Ja | Bedingt | Ja | Reihenfolge, Fensterbildung und zeitlicher Split sind entscheidend. |
| Autoencoder | Nein | Ja | Ja / bedingt | Ja | Bei Anomalieerkennung sind Ausreisser oft Zielsignal, nicht automatisch Fehler. |
| PCA | Nein | Ja | Ja | Ja | Varianzbasiert: Skalierung ist in der Regel Pflicht. |
| K-Means | Nein | Kritisch | Ja | Ja | Cluster werden ueber Distanzen gebildet; Ausreisser verschieben Zentren. |
| DBSCAN | Nein | Kritisch | Ja | Mittel | Skalierung bestimmt direkt `eps`; Ausreisser koennen bewusst als Noise erkannt werden. |
| Apriori / Association Rules | Nein | Nein | Speziell | Bedingt | Erwartet transaktionale, binarisiert/one-hot kodierte Warenkorb-Daten. |
| Stacking Classifier | Haengt von Base Models ab | Haengt von Base Models ab | Ja | Haengt von Base Models ab | Prepare muss zu allen Basis- und Meta-Modellen passen. |
| Voting Regressor | Haengt von Einzelmodellen ab | Haengt von Einzelmodellen ab | Ja | Haengt von Einzelmodellen ab | Gemeinsame Pipeline so bauen, dass das empfindlichste Modell korrekt versorgt wird. |
| PyCaret AutoML | Wird automatisiert behandelt | Wird automatisiert behandelt | Wird automatisiert behandelt | Pruefen | Automatisierung ersetzt keine fachliche Kontrolle von Datenqualitaet und Leakage. |
| CatBoost / LightGBM / Extra Trees / Gradient Boosting in PyCaret-Vergleichen | Bedingt | Nein | Bedingt | Mittel | PyCaret kann viel vorbereiten; bei manueller Nutzung die jeweilige Library beachten. |

## Modellgruppen

### Baum- und Ensemble-Modelle

Decision Trees, Random Forest, Extra Trees, Gradient Boosting, XGBoost und Explainable Boosting sind vergleichsweise robust gegen unterschiedliche Skalen. Skalierung ist deshalb normalerweise nicht erforderlich.

Trotzdem gilt:

- Kategoriale Features muessen fuer scikit-learn-Modelle numerisch kodiert werden.
- Fehlende Werte sind nicht fuer jede Library und jede Version gleich geregelt.
- Ausreisser muessen nicht automatisch geloescht werden, sollten aber bei Regression und stark verzerrten Zielvariablen geprueft werden.

### Lineare Modelle

Lineare Regression und logistische Regression reagieren empfindlicher auf Ausreisser und Feature-Skalen als Baumverfahren. Besonders bei regularisierten Modellen ist Skalierung praktisch Standard.

Wichtige Schritte:

- Missing Values imputieren oder Zeilen begruendet entfernen.
- Kategoriale Daten kodieren, oft mit One-Hot-Encoding.
- Numerische Features skalieren, besonders bei Logistic Regression.
- Ausreisser und stark schiefe Verteilungen pruefen.

### Distanzbasierte Modelle

KNN, SVM, K-Means und DBSCAN vergleichen Datenpunkte ueber Abstaende. Darum ist Skalierung hier nicht nur Kosmetik, sondern modellbestimmend.

Wichtige Schritte:

- Numerische Features skalieren.
- Ausreisser pruefen, weil sie Nachbarschaften und Distanzen verzerren.
- Kategoriale Features nur mit sinnvoller Kodierung verwenden.
- Bei DBSCAN `eps` erst nach der Skalierung interpretieren.

### Neuronale Netze

MLP, Keras Dense-Netze, CNNs, LSTMs und Autoencoder brauchen numerische, konsistente Eingaben. Skalierung oder Normalisierung ist fast immer sinnvoll, weil die Optimierung sonst langsam oder instabil werden kann.

Zusaetzlich wichtig:

- Zielvariable passend kodieren: binar, Klassenindex oder One-Hot.
- Bei Sequenzen zeitliche Reihenfolge erhalten.
- Bei Bildern Pixelwerte normalisieren.
- Bei Autoencodern klaeren, ob Ausreisser Fehler oder das eigentliche Zielsignal sind.

### Dimensionsreduktion und Association Rules

PCA ist varianzbasiert und braucht deshalb skalierte numerische Daten. Apriori arbeitet anders: Es lernt keine kontinuierlichen Modellparameter, sondern sucht haeufige Item-Kombinationen. Dafuer muessen Daten transaktional und meist binaer vorliegen.

## Entscheidungsregeln

Wenn unklar ist, welche Prepare-Schritte gebraucht werden, helfen diese Regeln:

| Frage | Konsequenz |
|-------|------------|
| Gibt es Text- oder Kategorie-Spalten? | Kodieren. |
| Gibt es `NaN` oder leere Werte? | Imputieren oder begruendet entfernen. |
| Nutzt das Modell Distanzen, Gradienten oder Varianz? | Skalieren. |
| Gibt es extreme Werte oder Messfehler? | Ausreisser pruefen; nicht blind loeschen. |
| Werden mehrere Modelle kombiniert? | Prepare am empfindlichsten Modell ausrichten. |
| Wird Cross-Validation oder Tuning genutzt? | Vorverarbeitung in die Pipeline legen. |

## Typische Fehler

| Fehler | Problem |
|--------|---------|
| Skalierung vor dem Train-Test-Split | Testdaten beeinflussen die Transformation: Data Leakage. |
| Encoding auf allen Daten fitten | Seltene Kategorien aus Testdaten koennen ins Training leaken. |
| Outlier automatisch entfernen | Echte, fachlich wichtige Extremfaelle gehen verloren. |
| Baumverfahren unnoetig skalieren | Meist kein Schaden, aber unnoetige Komplexitaet. |
| Distanzverfahren unskaliert trainieren | Grosse Zahlenbereiche dominieren das Modell. |
| AutoML ungeprueft vertrauen | Automatisierung erkennt nicht jedes fachliche Datenproblem. |

## Abgrenzung zu verwandten Dokumenten

| Thema | Abgrenzung |
|-------|------------|
| [Modellauswahl](../modeling/modellauswahl.html) | Waehlt das Modell zur Aufgabe; diese Seite leitet daraus Prepare-Anforderungen ab. |
| [Modell-Steckbriefe](../modeling/modell-steckbriefe.html) | Beschreibt Staerken und Grenzen der Modelle; diese Seite fokussiert die Datenvorbereitung. |
| [Workflow Design](../grundlagen/workflow-design.html) | Erklaert Pipelines und Leakage; diese Seite zeigt, welche Pipeline-Schritte je Modell relevant sind. |
| [Cross-Validation](../evaluate/cross_validation.html) | Bewertet Modelle robust; Prepare-Schritte muessen dabei innerhalb der CV-Schleife liegen. |

---

**Version:** 1.0<br>
**Stand:** Mai 2026<br>
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.
