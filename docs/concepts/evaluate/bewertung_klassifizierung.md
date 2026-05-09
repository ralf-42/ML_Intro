---
layout: default
title: Klassifizierung
parent: Evaluate
grand_parent: Konzepte
nav_order: 2
description: Metriken und Methoden zur Bewertung von Klassifikationsmodellen – von der Confusion Matrix bis zur ROC-Kurve
has_toc: true
---

# Bewertung Klassifizierung
{: .no_toc }

> **Klassifikationsmodelle erfordern spezifische Metriken zur Leistungsbewertung.**<br>
>  Dieses Kapitel behandelt die wichtigsten Werkzeuge: Confusion Matrix, Precision, Recall, F1-Score, Cohen's Kappa, ROC-Kurve und AUC. Besondere Aufmerksamkeit gilt dem Umgang mit unausgewogenen Datensätzen.

---

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Confusion Matrix

Die Konfusionsmatrix (Confusion Matrix) ist das fundamentale Werkzeug zur Bewertung von Klassifikationsmodellen. Sie zeigt, wie die Vorhersagen des Modells mit den tatsächlichen Werten übereinstimmen.

> [!NOTE] Ausgangspunkt<br>
> Viele Klassifikationsmetriken sind direkte Ableitungen aus TP, FP, TN und FN der Confusion Matrix.

### Aufbau der binären Confusion Matrix

```mermaid
flowchart TB
    subgraph matrix["<b>Confusion Matrix"]
    
		subgraph row2["Predicted: <b>Negative"]
            FN["✗ False Negative<br/>(FN)<br/>Fälschlich als negativ erkannt<br/><i>Typ-II-Fehler</i>"]
            TN["✓ True Negative<br/>(TN)<br/>Richtig als negativ erkannt"]
        end
        subgraph row1["Predicted: <b>Positive"]
            TP["✓ True Positive<br/>(TP)<br/>Richtig als positiv erkannt"]
            FP["✗ False Positive<br/>(FP)<br/>Fälschlich als positiv erkannt<br/><i>Typ-I-Fehler</i>"]
        end

    end
    
 
    
    
    style TP fill:#c8e6c9,stroke:#2e7d32
    style TN fill:#c8e6c9,stroke:#2e7d32
    style FP fill:#ffcdd2,stroke:#c62828
    style FN fill:#ffcdd2,stroke:#c62828
```

### Die vier Kategorien

| Kategorie | Beschreibung | Beispiel (Spam-Erkennung) |
|-----------|--------------|---------------------------|
| **True Positive (TP)** | Positives Ergebnis vorhergesagt, tatsächlich positiv | Spam korrekt als Spam erkannt |
| **False Positive (FP)** | Positives Ergebnis vorhergesagt, tatsächlich negativ | Normale E-Mail fälschlich als Spam markiert |
| **True Negative (TN)** | Negatives Ergebnis vorhergesagt, tatsächlich negativ | Normale E-Mail korrekt durchgelassen |
| **False Negative (FN)** | Negatives Ergebnis vorhergesagt, tatsächlich positiv | Spam fälschlich als normale E-Mail durchgelassen |

### Implementierung mit scikit-learn

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix berechnen
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negativ', 'Positiv'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

---

## Metriken aus der Confusion Matrix

Aus den vier Grundwerten der Confusion Matrix lassen sich verschiedene Leistungsmetriken ableiten, die unterschiedliche Aspekte der Modellqualität bewerten.


### Accuracy (Genauigkeit)

Die Accuracy misst den Anteil aller **korrekten Vorhersagen** an der Gesamtzahl der Fälle.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(target_test, target_pred)
print(f"Accuracy: {accuracy:.4f}")
```

> [!WARNING] Accuracy-Falle bei Imbalance<br>
> Ein Modell kann hohe Accuracy erreichen und trotzdem die relevante Minderheitsklasse kaum erkennen.

### Precision (Relevanz/Präzision)

Die Precision gibt an, wie viele der als **positiv vorhergesagten Fälle tatsächlich positiv** sind.

$$\text{Precision} = \frac{TP}{TP + FP}$$

```python
from sklearn.metrics import precision_score

precision = precision_score(target_test, target_pred)
print(f"Precision: {precision:.4f}")
```

**Wann ist Precision wichtig?**
- Wenn **False Positives teuer oder problematisch** sind
- Beispiel: Spam-Filter (normale E-Mails sollen nicht blockiert werden)
- Beispiel: Qualitätskontrolle (gute Produkte sollen nicht aussortiert werden)

### Recall (Sensitivität/Trefferquote)

Der Recall gibt an, wie viele der **tatsächlich positiven Fälle korrekt** erkannt wurden.

$$\text{Recall} = \frac{TP}{TP + FN}$$

```python
from sklearn.metrics import recall_score

recall = recall_score(target_test, target_pred)
print(f"Recall: {recall:.4f}")
```

**Wann ist Recall wichtig?**
- Wenn **False Negatives teuer oder gefährlich** sind
- Beispiel: Krankheitsdiagnose (Kranke sollen nicht übersehen werden)
- Beispiel: Betrugserkennung (Betrug soll nicht unentdeckt bleiben)

### F1-Score

Der F1-Score ist das **harmonische Mittel aus Precision und Recall** und bietet eine ausgewogene Bewertung.

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

```python
from sklearn.metrics import f1_score

f1 = f1_score(target_test, target_pred)
print(f"F1-Score: {f1:.4f}")
```

**Interpretation:** Der F1-Score ist besonders nützlich, wenn ein Gleichgewicht zwischen Precision und Recall gewünscht ist und die Klassen unausgewogen sind.

### Beispielrechnung

Gegeben sei folgende Confusion Matrix:

|  | Predicted: Positiv | Predicted: Negativ |
|---|:---:|:---:|
| **Actual: Positiv** | TP = 20 | FN = 3 |
| **Actual: Negativ** | FP = 5 | TN = 15 |

**Berechnungen:**

- **Accuracy:** (20 + 15) / (20 + 15 + 5 + 3) = 35 / 43 ≈ **0.81**
- **Precision:** 20 / (20 + 5) = 20 / 25 = **0.80**
- **Recall:** 20 / (20 + 3) = 20 / 23 ≈ **0.87**
- **F1-Score:** 2 × (0.80 × 0.87) / (0.80 + 0.87) ≈ **0.83**

### Kompletter Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(target_test, target_pred, target_names=['Negativ', 'Positiv']))
```

**Beispielausgabe:**

```
              precision    recall  f1-score   support

     Negativ       0.83      0.75      0.79        20
     Positiv       0.80      0.87      0.83        23

    accuracy                           0.81        43
   macro avg       0.82      0.81      0.81        43
weighted avg       0.82      0.81      0.81        43
```

---

## Multi-Class Confusion Matrix

Bei Klassifikationsproblemen mit mehr als zwei Klassen wird die Confusion Matrix entsprechend erweitert. Die Berechnung der Metriken erfolgt dann klassenweise.

<img src="https://raw.githubusercontent.com/ralf-42/ML_Intro/main/07_image/multi_confusion_matrix.png" class="logo" width="650"/>

### Berechnung für einzelne Klassen

Bei Multi-Class-Problemen werden TP, TN, FP und FN für jede Klasse einzeln berechnet:

**Beispiel für Klasse "Apple" aus der obigen Matrix:**

| Metrik | Berechnung                                  | Ergebnis          |
| ------ | ------------------------------------------- | ----------------- |
| **TP** | Korrekt als Apple klassifiziert             | 7                 |
| **FN** | Apple, aber als andere Klasse klassifiziert | 1 + 3 = 4         |
| **FP** | Andere Klasse, aber als Apple klassifiziert | 8 + 9 = 17        |
| **TN** | Andere Klasse, korrekt nicht als Apple      | 2 + 3 + 2 + 1 = 8 |

### Implementierung

```python
from sklearn.metrics import confusion_matrix, classification_report

# Multi-Class Confusion Matrix
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report für alle Klassen
print("\nClassification Report:")
print(classification_report(target_test, target_pred, target_names=['Apple', 'Orange', 'Banana']))
```

### Aggregationsstrategien

Bei Multi-Class-Problemen gibt es verschiedene Möglichkeiten, die Metriken zu aggregieren:

| Strategie | Beschreibung | Verwendung |
|-----------|--------------|------------|
| **Macro Average** | Ungewichteter Durchschnitt aller Klassen | Wenn alle Klassen gleich wichtig sind |
| **Weighted Average** | Gewichtet nach Klassenhäufigkeit | Bei unausgewogenen Klassen |
| **Micro Average** | Gesamtsumme aller TP, FP, FN | Für Gesamtleistung |

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix berechnen
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negativ', 'Positiv'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```0

---

## Cohen's Kappa

Cohen's Kappa ist eine robustere Metrik als die Accuracy, da sie die zufällige Übereinstimmung berücksichtigt. Sie ist besonders wertvoll bei unausgewogenen Klassen.

### Interpretation

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix berechnen
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negativ', 'Positiv'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```1

| Kappa-Wert | Interpretation |
|------------|----------------|
| κ < 0.0 | Schlechte Übereinstimmung (schlechter als Zufall) |
| κ ≤ 0.2 | Leichte Übereinstimmung |
| κ ≤ 0.4 | Ausreichende Übereinstimmung |
| κ ≤ 0.6 | Moderate Übereinstimmung |
| κ ≤ 0.8 | Beachtliche Übereinstimmung |
| κ > 0.8 | (Fast) vollständige Übereinstimmung |

### Berechnung

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Wobei:
- $p_o$ = beobachtete Übereinstimmung (Accuracy)
- $p_e$ = erwartete zufällige Übereinstimmung

### Implementierung

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix berechnen
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negativ', 'Positiv'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```2

### Vorteile gegenüber Accuracy

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix berechnen
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negativ', 'Positiv'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```3

---

## ROC-Kurve (Receiver Operating Characteristic)

Die ROC-Kurve ist ein leistungsstarkes Werkzeug zur Visualisierung der **Klassifikationsleistung über verschiedene Schwellenwerte** hinweg.

### Grundkonzept

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix berechnen
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negativ', 'Positiv'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```4

<img src="```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix berechnen
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negativ', 'Positiv'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```5 class="logo" width="750"/>


### Sensitivität und Spezifität

| Metrik                  | Formel                          | Beschreibung                                                |
| ----------------------- | ------------------------------- | ----------------------------------------------------------- |
| **Sensitivität (TPR)**  | TP / (TP + FN)                  | Anteil der korrekt erkannten positiven Fälle                |
| **Spezifität (TNR)**    | TN / (TN + FP)                  | Anteil der korrekt erkannten negativen Fälle                |
| **False Positive Rate** | FP / (FP + TN) = 1 - Spezifität | Anteil der fälschlich als positiv erkannten negativen Fälle |

### Implementierung

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix berechnen
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negativ', 'Positiv'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```6

### Schwellenwert-Optimierung

Der Standard-Schwellenwert von 0.5 ist nicht immer optimal. Je nach Anwendungsfall kann ein anderer Schwellenwert sinnvoller sein.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix berechnen
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negativ', 'Positiv'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```7

**Beispiel: Krebsdiagnose**

Bei einem Test auf eine ernsthafte, aber behandelbare Krankheit wie Krebs:

- **Hohe Sensitivität (TPR):** Fast alle tatsächlichen Fälle werden erkannt
- **Akzeptierte höhere FPR:** Einige Gesunde werden fälschlich positiv getestet

Dies ist vertretbar, weil:
1. Ein übersehener Krebsfall (FN) kann tödlich sein
2. Ein falscher Alarm (FP) führt "nur" zu weiteren Tests

### Optimalen Schwellenwert finden

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix berechnen
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negativ', 'Positiv'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```8

---

## Area Under the Curve (AUC)

Die AUC (Area Under the ROC Curve) fasst die **Leistung eines Klassifikators** in einer einzigen Zahl zusammen.

### Interpretation

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix berechnen
cm = confusion_matrix(target_test, target_pred)
print("Confusion Matrix:")
print(cm)

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negativ', 'Positiv'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```9


<img src="```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(target_test, target_pred)
print(f"Accuracy: {accuracy:.4f}")
```0 class="logo" width="750"/>


| AUC-Wert  | Interpretation                          |
| --------- | --------------------------------------- |
| 0.5       | Keine Unterscheidungsfähigkeit (Zufall) |
| 0.6 - 0.7 | Schwache Trennschärfe/Diskriminierung   |
| 0.7 - 0.8 | Akzeptable Trennschärfe                 |
| 0.8 - 0.9 | Gute Trennschärfe                       |
| > 0.9     | Exzellente Trennschärfe                 |

### Vorteile der AUC

1. **Schwellenwert-unabhängig:** Bewertet die gesamte ROC-Kurve
2. **Vergleichbarkeit:** Ermöglicht einfachen Modellvergleich
3. **Robust:** Weniger anfällig für Klassenungleichgewichte als Accuracy

### Implementierung

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(target_test, target_pred)
print(f"Accuracy: {accuracy:.4f}")
```1

### Modellvergleich mit AUC

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(target_test, target_pred)
print(f"Accuracy: {accuracy:.4f}")
```2

---

## Umgang mit unausgewogenen Klassen

> [!WARNING] Geschäftsrisiko<br>
> Bei unausgewogenen Klassen führt die falsche Metrik oft zu teuren Fehlentscheidungen im Betrieb.

> [!TIP] Metrik nach Kostenmodell wählen<br>
> Wenn False Negatives kritisch sind, Recall priorisieren; wenn False Positives teuer sind, Precision priorisieren.

Bei unausgewogenen Datensätzen (Imbalanced Classification) sind Standard-Metriken oft irreführend. Die richtige Metrikauswahl ist entscheidend.

### Entscheidungsbaum zur Metrikauswahl

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(target_test, target_pred)
print(f"Accuracy: {accuracy:.4f}")
```3

### Empfehlungen nach Anwendungsfall

| Anwendungsfall | Empfohlene Metrik | Begründung |
|----------------|-------------------|------------|
| **Medizinische Diagnose** | Recall + Spezifität | Kranke nicht übersehen |
| **Spam-Erkennung** | Precision + F1 | Normale E-Mails nicht blockieren |
| **Betrugserkennung** | Recall + AUC | Betrug nicht durchlassen |
| **Qualitätskontrolle** | Precision + Recall | Balance je nach Kosten |
| **Kreditwürdigkeit** | AUC + F1 | Ranking und Gesamtleistung |

### Precision-Recall-Kurve als Alternative

Bei stark unausgewogenen Daten kann die Precision-Recall-Kurve informativer sein als die ROC-Kurve:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(target_test, target_pred)
print(f"Accuracy: {accuracy:.4f}")
```4

---

## Zusammenfassung: Metrik-Übersicht

> [!SUCCESS] Mindeststandard<br>
> Neben Accuracy immer mindestens Precision, Recall, F1 und eine klassenbezogene Auswertung berichten.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(target_test, target_pred)
print(f"Accuracy: {accuracy:.4f}")
```5

### Quick Reference: Wann welche Metrik?

| Situation | Empfohlene Metrik(en) |
|-----------|----------------------|
| Ausgewogene Klassen | Accuracy, F1-Score |
| Unausgewogene Klassen | F1-Score, AUC, Cohen's Kappa |
| Kosten unterschiedlich | Precision oder Recall (je nach Kosten) |
| Modellvergleich | AUC |
| Schwellenwert-Optimierung | ROC-Kurve mit Youden's J |
| Sehr seltene positive Klasse | PR-Kurve, Average Precision |

### Komplettes Evaluations-Template

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(target_test, target_pred)
print(f"Accuracy: {accuracy:.4f}")
```6

---

## Weiterführende Themen

- **Cross-Validation:** Robustere Schätzung der Metriken durch mehrfache Splits
- **Learning Curves:** Diagnose von Over-/Underfitting
- **Calibration:** Zuverlässigkeit der Wahrscheinlichkeitsvorhersagen
- **Cost-Sensitive Learning:** Berücksichtigung unterschiedlicher Fehlerkosten

---

## Abgrenzung zu verwandten Dokumenten

| Thema                                                | Abgrenzung                                                                                                    |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| [Bewertung: Regression](./bewertung_regression.html) | Klassifikations-Metriken für kategoriale Vorhersagen; Regressions-Metriken für kontinuierliche Werte        |
| [Overfitting](./overfitting.html)                    | Klassifikations-Metriken quantifizieren Vorhersageguete; Overfitting erkennt man an der Train-Test-Diskrepanz |
| [Cross-Validation](./cross_validation.html)          | Cross-Validation ist die Evaluierungsmethodik; Klassifikations-Metriken sind die Messgroessen dabei           |


---

**Version:** 1.1<br>
**Stand:** April 2026<br>
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.