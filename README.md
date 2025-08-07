


# 1 🎯 Machine Learning Kurs

Dieses Repository enthält umfassende Lehrmaterialien für einen Machine Learning-Kurs mit Schwerpunkt auf Neuronale Netze und ML-Anwendungen. 
Dieser Kurs bietet einen strukturierten Lernpfad von grundlegenden Machine Learning-Konzepten bis hin zu fortgeschrittenen Deep Learning-Anwendungen und umfasst sowohl überwachte als auch unüberwachte Lerntechniken, neuronale Netze, Ensemble-Methoden und moderne generative KI-Anwendungen.

# 2 📚 Repository-Struktur

Der Kursinhalt ist in 10 Hauptmodule unter `01 ipynb/` organisiert:

## 2.1 🟢 Modul 00: Allgemeine Konzepte
**Pfad**: `00 general/`
- Grundlegende ML-Konzepte und pandas-Grundlagen
- Datensatzbehandlung und -exploration
- Google Colab-Integration mit Gemini

## 2.2 🔵 Modul 01: Überwachtes Lernen
**Pfad**: `01 supervised/`
- Entscheidungsbäume (Titanic-Datensatz)
- Lineare Regression (MPG-Vorhersage)
- Random Forests (Diamantpreisvorhersage)

## 2.3 🟣 Modul 02: Unüberwachtes Lernen
**Pfad**: `02 unsupervised/`
- K-means und DBSCAN-Clustering
- Isolation Forest für Anomalieerkennung
- PCA für Dimensionsreduktion
- Assoziationsregeln (Apriori-Algorithmus)

## 2.4 🟡 Modul 03: Neuronale Netze
**Pfad**: `03 network/`
- Multi-Layer Perceptron (MLP) Implementierungen
- Keras/TensorFlow neuronale Netze
- Anwendungen auf Krebs- und Diamant-Datensätzen

## 2.5 🟠 Modul 04: Ensemble-Methoden
**Pfad**: `04 ensemble/`
- XGBoost-Implementierung
- Stacking-Ensemble-Techniken

## 2.6 ⚪ Modul 05: Modell-Tuning & Validierung
**Pfad**: `05 tuning/`
- Kreuzvalidierungstechniken
- Hyperparameter-Optimierung (Grid Search, Random Search)
- ROC-AUC-Analyse und Schwellenwertoptimierung
- AutoML mit PyCaret
- Lernkurven und Validierungsstrategien

## 2.7 🔴 Modul 06: ML-Workflows
**Pfad**: `06 workflow/`
- Scikit-learn Pipelines
- End-to-End ML-Workflow-Automatisierung

## 2.8 🟤 Modul 07: Spezialisierte Anwendungen
**Pfad**: `07 special/`
- Computer Vision (MNIST, YOLO)
- Natural Language Processing (Spam-Erkennung)
- Zeitreihenanalyse (Wettervorhersage)
- Autoencoder für Dimensionsreduktion

## 2.9 ⚫ Modul 08: Generative KI
**Pfad**: `08 genai/`
- LangChain-Integration mit OpenAI
- PDF-Zusammenfassung mit LLMs
- Interaktive Chat-Anwendungen mit Gradio

## 2.10 🔵 Modul 09: Vielfältige Anwendungen
**Pfad**: `09 diverse/`
- Erklärbare KI (XAI) Techniken
- Gradio-Webanwendungen
- Modellpersistierung (Speichern/Laden)
- Business Intelligence mit Gemini AI


# 3 🔄 Standardisierter ML-Workflow

Jedes Modul folgt einer konsistenten 5-Phasen Machine Learning-Workflow-Vorlage:


## 3.1 🔍 Verstehen
**Checkliste**:
- ✅ Aufgabenverständnis
- ✅ Datensammlung
- ✅ Statistische Analyse (Min, Max, Mittelwert, Korrelation)
- ✅ Datenvisualisierung (Streudiagramme, Box-Plots)
- ✅ Definition der Vorverarbeitungsstrategie

## 3.2 🛠️ Vorbereiten
**Checkliste**:
- ✅ Entfernung unnötiger Features
- ✅ Datentypkonvertierung
- ✅ Duplikatserkennung und -entfernung
- ✅ Behandlung fehlender Werte
- ✅ Ausreißerbehandlung
- ✅ Kodierung kategorischer Features
- ✅ Skalierung numerischer Features
- ✅ Feature Engineering
- ✅ Dimensionsreduktion
- ✅ Resampling (Über-/Unterabtastung)
- ✅ Pipeline-Erstellung
- ✅ Train-Test-Split

## 3.3 🤖 Modellierung
**Checkliste**:
- ✅ Modellauswahl
- ✅ Pipeline-Konfiguration
- ✅ Trainingsausführung
- ✅ Hyperparameter-Tuning
- ✅ Kreuzvalidierung
- ✅ Bootstrapping
- ✅ Regularisierung

## 3.4 📊 Bewerten
**Checkliste**:
- ✅ Vorhersagegenerierung (Training, Test)
- ✅ Modellleistungsbewertung
- ✅ Residualanalyse
- ✅ Feature-Wichtigkeitsanalyse
- ✅ Robustheitstests
- ✅ Modellinterpretation
- ✅ Sensitivitätsanalyse
- ✅ Kommunikation (Wichtigste Erkenntnisse)

## 3.5 🚀 Bereitstellen
**Checkliste**:
- ✅ Modellexport und -speicherung
- ✅ Abhängigkeiten und Umgebungssetup
- ✅ Sicherheits- und Datenschutzüberlegungen
- ✅ Produktionsintegration
- ✅ Testen und Validierung
- ✅ Dokumentation und Wartung

# 4 🛠️ Technologie-Stack

## 4.1 Kernbibliotheken
- **Machine Learning**: scikit-learn, pandas, numpy
- **Deep Learning**: Keras, TensorFlow
- **Datenvisualisierung**: matplotlib, plotly
- **Spezialisierte ML**: XGBoost, PyCaret
- **Generative KI**: LangChain, OpenAI API
- **Webanwendungen**: Gradio
- **Datenverarbeitung**: pandas, numpy

## 4.2 Entwicklungsumgebung
- **Laufzeit**: Python 3.7+
- **IDE**: Jupyter Lab/Notebook
- **Cloud-Integration**: Google Colab-Kompatibilität



# 5 🗂️ Datensatz-Sammlung

Der Kurs verwendet verschiedene reale Datensätze für praktisches Lernen:
- **Titanic**: Überlebensvorhersage (Klassifikation)
- **Diamonds**: Preisvorhersage (Regression)
- **Cancer**: Medizinische Diagnose (Klassifikation)
- **MPG**: Kraftstoffeffizienzvorhersage (Regression)
- **MNIST**: Handschriftenerkennung (Computer Vision)
- **Weather**: Zeitreihenvorhersage
- Und viele weitere spezialisierte Datensätze

# 6 🎓 Lernziele

Nach Abschluss dieses Kurses können Studierende:

- ✅ Grundlegende ML-Konzepte und Workflows verstehen
- ✅ Überwachte und unüberwachte Lernalgorithmen implementieren
- ✅ Neuronale Netze mit Keras/TensorFlow erstellen und optimieren
- ✅ Ensemble-Methoden und fortgeschrittene ML-Techniken anwenden
- ✅ Hyperparameter-Tuning und Modellvalidierung durchführen
- ✅ Mit spezialisierten Anwendungen arbeiten (CV, NLP, Zeitreihen)
- ✅ Einsetzbare ML-Anwendungen mit modernen Tools erstellen
- ✅ Machine Learning-Modelle interpretieren und erklären


# 7 🔧 Technische Anforderungen

- **Python**: 3.7 oder höher
- **RAM**: Mindestens 8GB (16GB empfohlen für Deep Learning-Module)
- **Speicher**: Mindestens 2GB freier Speicherplatz
- **Internet**: Erforderlich für Datensatz-Downloads und API-Integrationen
- **GPU**: Optional, aber empfohlen für Deep Learning-Module (07-09)

# 8 🌟 Besondere Merkmale

- **Zweisprachiger Ansatz**: Deutsche Anweisungen mit englischen Fachbegriffen
- **Praktischer Fokus**: Reale Datensätze und Geschäftsanwendungen
- **Moderne Integration**: Enthält neueste KI-Entwicklungen (GPT, Gemini)
- **Interaktive Anwendungen**: Gradio-basierte Web-Apps für Modellbereitstellung
- **Umfassende Abdeckung**: Von grundlegendem ML bis hin zu modernster generativer KI


# 9 ⚠️ Wichtige Hinweise

- Dies ist ein Bildungsrepository, das für sequenzielles Lernen konzipiert ist
- Einige Notebooks erfordern möglicherweise API-Schlüssel (OpenAI, Gemini) für volle Funktionalität
- Notebooks können Abhängigkeiten zu vorherigen Modulen haben


# 10 📄 Lizenz

Dieser Bildungsinhalt wird für akademische und Lernzwecke bereitgestellt. Bitte respektiere die Lizenzbestimmungen einzelner Datensätze und Bibliotheken, die im gesamten Kurs verwendet werden. MIT License.

---

*Dieser Kurs bietet eine umfassende Reise durch modernes Machine Learning, von traditionellen Algorithmen bis hin zu modernsten generativen KI-Anwendungen, mit Fokus auf praktische Implementierung und reale Anwendungen.*