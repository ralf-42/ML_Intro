---
layout: default
title: Konzepte
nav_order: 2
has_children: true
description: "Theoretische Grundlagen und technische Konzepte"
has_toc: true
---

# Konzepte

Die Konzeptseiten erklären nicht möglichst viel auf einmal, sondern genau die Stellen, an denen im Kurs die wichtigsten Unterschiede entstehen: Welche Art von Problem liegt vor, wie werden Daten sinnvoll vorbereitet, woran erkennt sich ein geeignetes Modell und wie wird Qualität belastbar bewertet? Die Sammlung ist deshalb entlang des typischen ML-Workflows aufgebaut.

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Grundlagen

Hier beginnt der Kurs fachlich. Die Dokumente klären, was Machine Learning von klassischer Programmierung unterscheidet, wie der Workflow von der Problemklärung bis zur Auswertung aussieht und welche Arbeitsweisen in Notebooks oder Python-Projekten tragfähig bleiben.

- **[Grundlagen](https://ralf-42.github.io/ML_Intro/concepts/01_grundlagen.html)** – *Wie beginnt der ML-Workflow sinnvoll?* Einstieg in Lernparadigmen, Workflow und praktische Arbeitsweisen.

## Daten vorbereiten

Viele Qualitätsprobleme entstehen vor dem ersten Modelltraining. Dieser Block behandelt deshalb die Schritte, die in Projekten oft unterschätzt werden: fehlende Werte, Ausreißer, Skalierung, Kodierung und die saubere Trennung von Trainings- und Testdaten.

- **[Prepare](https://ralf-42.github.io/ML_Intro/concepts/02_prepare.html)** – *Wie werden Rohdaten modellfähig gemacht?* Datenaufbereitung, Transformation und typische Fehlerquellen.

## Modelle auswählen

Die Modellsektion dient nicht als Katalog aller Verfahren, sondern als Entscheidungshilfe. Relevant ist, welches Modell zu welchem Datentyp, welcher Fragestellung und welchem Qualitätsanspruch passt.

- **[Modeling](https://ralf-42.github.io/ML_Intro/concepts/03_modeling.html)** – *Welches Modell passt zur Aufgabe?* Klassische Verfahren, Ensembles, Deep Learning und AutoML im Vergleich.

## Qualität bewerten

Ein Modell ist erst dann brauchbar, wenn seine Qualität sauber eingeordnet wurde. Hier geht es um Metriken, Cross-Validation, Overfitting, Bootstrapping und die Unterschiede zwischen Klassifikation, Regression, Clustering und Anomalieerkennung.

- **[Evaluate](https://ralf-42.github.io/ML_Intro/concepts/04_evaluate.html)** – *Wie wird Modellqualität belastbar gemessen?* Bewertungslogik und typische Fehlinterpretationen.

## Erklärbarkeit

Erklärbarkeit wird relevant, sobald Modelle begründet, geprüft oder gegenüber Fachbereichen erläutert werden müssen. Die XAI-Seiten zeigen, was Verfahren wie SHAP oder LIME leisten und wo ihre Grenzen liegen.

- **[XAI](https://ralf-42.github.io/ML_Intro/concepts/08_xai.html)** – *Wie wird ein Modell nachvollziehbar gemacht?* Methoden der erklärbaren KI im ML-Kontext.

---

**Version:** 1.0
**Stand:** März 2026
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.
