---
layout: default
title: Neuronale Netze
parent: Modeling
grand_parent: Konzepte
nav_order: 8
description: "Grundlagen künstlicher neuronaler Netze: Architektur, Aktivierungsfunktionen, Training und Loss Functions"
has_toc: true
---

# Neuronale Netze
{: .no_toc }

> **Künstliche neuronale Netze (KNN) sind dem biologischen Nervensystem nachempfundene Modelle, die komplexe nichtlineare Zusammenhänge zwischen Ein- und Ausgaben lernen können. Sie bilden das Fundament moderner Deep-Learning-Systeme.**

---

## Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Überblick

Künstliche neuronale Netze (englisch: Artificial Neural Networks, ANN) sind Netze aus künstlichen Neuronen, die nach dem biologischen Vorbild des Nervensystems modelliert sind. Sie können sowohl für **Klassifikation** als auch für **Regression** eingesetzt werden.

```mermaid
flowchart LR
    subgraph Input["Eingabeschicht"]
        I1["Input 1"]
        I2["Input 2"]
        I3["Input n"]
    end
    
    subgraph Hidden["Verborgene Schicht(en)"]
        H1((("N")))
        H2((("N")))
        H3((("N")))
        H4((("N")))
    end
    
    subgraph Output["Ausgabeschicht"]
        O1["Output"]
    end
    
    I1 --> H1 & H2
    I2 --> H1 & H2 & H3 & H4
    I3 --> H3 & H4
    
    H1 --> O1
    H2 --> O1
    H3 --> O1
    H4 --> O1
    
    style I1 fill:#e3f2fd
    style I2 fill:#e3f2fd
    style I3 fill:#e3f2fd
    style H1 fill:#fff9c4
    style H2 fill:#fff9c4
    style H3 fill:#fff9c4
    style H4 fill:#fff9c4
    style O1 fill:#c8e6c9
```

### Grundlegende Bestandteile

Ein neuronales Netz besteht aus:

| Komponente            | Beschreibung                        | Funktion                                        |
| --------------------- | ----------------------------------- | ----------------------------------------------- |
| **Neuronen/Knoten**   | Grundbausteine des Netzwerks        | Verarbeiten Eingaben und erzeugen Ausgaben      |
| **Verbindungen**      | Gewichtete Kanten zwischen Neuronen | Übertragen Signale mit unterschiedlicher Stärke |
| **Schichten (Layer)** | Gruppierte Neuronen                 | Input → Hidden → Output                         |

---

## Aufbau eines Neuronalen Netzes

Ein künstliches Neuron kann durch vier Basiselemente beschrieben werden:

```mermaid
flowchart LR
    subgraph Eingaben["<b>Eingaben"]
        X1["x₁"]
        X2["x₂"]
        X3["x₃"]
    end
    
    subgraph Gewichte["<b>Gewichte & Bias"]
        W1["w₁"]
        W2["w₂"]
        W3["w₃"]
        B["Bias b"]
    end
    
    subgraph Verarbeitung["<b>Neuron"]
        SUM["Σ<br/>Übertragungsfunktion"]
        ACT["f(x)<br/>Aktivierungsfunktion"]
    end
    
    Y["y<br/><b>Ausgabe"]
    
    X1 --> W1 --> SUM
    X2 --> W2 --> SUM
    X3 --> W3 --> SUM
    B --> SUM
    SUM --> ACT --> Y
    
    style SUM fill:#e3f2fd
    style ACT fill:#fff9c4
    style Y fill:#c8e6c9
```

### Die vier Grundelemente

| Element | Beschreibung | Mathematisch |
|---------|--------------|--------------|
| **Gewichtung (Weights)** | Jeder Eingang erhält ein Gewicht, das den Einfluss der Eingabe bestimmt | w₁, w₂, ..., wₙ |
| **Bias** | Konstante additive Komponente, die den Schwellenwert verschiebt | b |
| **Übertragungsfunktion** | Berechnet die gewichtete Summe aller Eingaben | z = Σ(wᵢ · xᵢ) + b |
| **Aktivierungsfunktion** | Transformiert die Summe in die Ausgabe | y = f(z) |

### Beispiel: Einfaches Neuron

```python
import numpy as np

def neuron(inputs, weights, bias, activation_fn):
    """
    Berechnung eines einzelnen Neurons
    
    Args:
        inputs: Eingabewerte [x1, x2, x3]
        weights: Gewichte [w1, w2, w3]
        bias: Bias-Wert
        activation_fn: Aktivierungsfunktion
    
    Returns:
        Ausgabe des Neurons
    """
    # Übertragungsfunktion: gewichtete Summe + Bias
    z = np.dot(inputs, weights) + bias
    
    # Aktivierungsfunktion anwenden
    output = activation_fn(z)
    
    return output

# Beispiel
inputs = np.array([0.5, 0.3, 0.2])
weights = np.array([0.4, 0.6, 0.8])
bias = 0.1

# Mit ReLU-Aktivierung
relu = lambda x: max(0, x)
result = neuron(inputs, weights, bias, relu)
print(f"Ausgabe: {result}")  # 0.54
```

---

## Aktivierungsfunktionen

Die Aktivierungsfunktion bestimmt den möglichen Wertebereich der Ausgabe eines Neurons. Ohne Aktivierungsfunktionen wäre ein tiefes Netzwerk mathematisch äquivalent zu einem einfachen linearen Modell.

```mermaid
flowchart TD
    subgraph Aktivierungsfunktionen["<b>Aktivierungsfunktionen"]
        direction TB
        
        subgraph Linear["Linear (Identity)"]
            L["f(x) = x<br/>Wertebereich: (-∞, ∞)"]
        end
        
        subgraph Sigmoid["Sigmoid"]
            S["f(x) = 1/(1+e⁻ˣ)<br/>Wertebereich: (0, 1)"]
        end
        
        subgraph Tanh["Tanh"]
            T["f(x) = tanh(x)<br/>Wertebereich: (-1, 1)"]
        end
        
        subgraph ReLU["ReLU"]
            R["f(x) = max(0, x)<br/>Wertebereich: (0, ∞)"]
        end
    end
    
    style L fill:#e0e0e0
    style S fill:#ffcdd2
    style T fill:#c8e6c9
    style R fill:#bbdefb
```

### Übersicht der wichtigsten Aktivierungsfunktionen

| Funktion | Formel | Wertebereich | Typischer Einsatz |
|----------|--------|--------------|-------------------|
| **Linear (Identity)** | f(x) = x | (-∞, ∞) | Output-Layer bei Regression |
| **Sigmoid** | f(x) = 1/(1+e⁻ˣ) | (0, 1) | Binäre Klassifikation (Output) |
| **Tanh** | f(x) = tanh(x) | (-1, 1) | Hidden Layers, RNNs |
| **ReLU** | f(x) = max(0, x) | [0, ∞) | Hidden Layers (Standard) |
| **Softmax** | f(xᵢ) = eˣⁱ/Σeˣʲ | (0, 1), Summe = 1 | Multi-Class Klassifikation (Output) |

### Wann welche Aktivierungsfunktion?

```mermaid
flowchart TD
    START["Welche Aktivierungsfunktion?"]
    
    LAYER{"Welche Schicht?"}
    
    START --> LAYER
    
    LAYER -->|Hidden Layer| HIDDEN["ReLU<br/>(oder Leaky ReLU, SELU)"]
    LAYER -->|Output Layer| OUTPUT{"Aufgabe?"}
    
    OUTPUT -->|Regression| REG["Linear"]
    OUTPUT -->|Binäre Klassifikation| BIN["Sigmoid"]
    OUTPUT -->|Multi-Class| MULTI["Softmax"]
    
    style HIDDEN fill:#bbdefb
    style REG fill:#e0e0e0
    style BIN fill:#ffcdd2
    style MULTI fill:#fff9c4
```

### Python-Implementierung

```python
import numpy as np

# Aktivierungsfunktionen
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerische Stabilität
    return exp_x / exp_x.sum()

# Beispiel
x = np.array([-2, -1, 0, 1, 2])
print(f"Sigmoid: {sigmoid(x)}")
print(f"Tanh:    {tanh(x)}")
print(f"ReLU:    {relu(x)}")
```

---

## Training: Forward und Backward Pass

Das Training eines neuronalen Netzes erfolgt in zwei Phasen, die iterativ wiederholt werden:

```mermaid
flowchart LR
    subgraph Forward["Forward Pass"]
        direction TB
        DATA["Trainingsdaten"] --> NN["Neuronales Netz"]
        NN --> PRED["Vorhersage ŷ"]
    end
    
    subgraph Loss["Loss-Berechnung"]
        PRED --> LOSS["Loss/Error<br/>L(y, ŷ)"]
        ACTUAL["Tatsächlicher Wert y"] --> LOSS
    end
    
    subgraph Backward["Backward Pass"]
        LOSS --> GRAD["Gradienten<br/>berechnen"]
        GRAD --> UPDATE["Gewichte<br/>aktualisieren"]
        UPDATE --> NN
    end
    
    style DATA fill:#e3f2fd
    style PRED fill:#fff9c4
    style LOSS fill:#ffcdd2
    style UPDATE fill:#c8e6c9
```

### Der Trainingszyklus

1. **Forward Pass**: Eingabedaten werden durch das Netz propagiert → Vorhersage
2. **Loss-Berechnung**: Abweichung zwischen Vorhersage und tatsächlichem Wert
3. **Backward Pass**: Gradienten werden rückwärts durch das Netz berechnet
4. **Update**: Gewichte werden angepasst, um den Loss zu minimieren

<br>

<img src="https://raw.githubusercontent.com/ralf-42/ML_Intro/main/07_image/forward_backward.png" class="logo" width="950"/>

### Rechenbeispiel: Forward Pass

Betrachten wir ein einfaches Netz mit zwei Eingaben und einem Ausgabeneuron:

```python
import numpy as np

# Netzwerk-Parameter
inputs = np.array([9, 12])  # Besuchte Kurstermine, bearbeitete Fallstudien
weights_hidden = np.array([[0.11, 0.21],    # Gewichte zu Hidden-Neuron 1
                           [0.12, 0.08]])    # Gewichte zu Hidden-Neuron 2
bias_hidden = np.array([0.0, 0.0])

weights_output = np.array([0.14, 0.15])      # Gewichte zum Output
bias_output = 0.0

# Forward Pass
hidden = np.dot(weights_hidden, inputs) + bias_hidden
# hidden[0] = 0.11*9 + 0.21*12 = 0.99 + 2.52 = 3.51
# hidden[1] = 0.12*9 + 0.08*12 = 1.08 + 0.96 = 2.04

output = np.dot(weights_output, hidden) + bias_output
# output = 0.14*3.51 + 0.15*2.04 = 0.49 + 0.31 = 0.80

print(f"Hidden Layer: {hidden}")  # [3.51, 2.04]
print(f"Output: {output}")         # 0.80
```

### Backward Pass: Gewichtsanpassung

Der Backward Pass verwendet den **Gradientenabstieg** zur Optimierung:

```python
# Parameter
learning_rate = 0.05
target = 1.0  # Erwarteter Wert
prediction = 0.80

# Error
error = target - prediction  # 0.2

# Gewichte aktualisieren (vereinfacht, ohne vollständige Kettenregel)
# w_neu = w_alt + learning_rate * hidden * error
weights_output_new = weights_output + learning_rate * hidden * error

print(f"Alte Gewichte: {weights_output}")
print(f"Neue Gewichte: {weights_output_new}")
# [0.14, 0.15] → [0.175, 0.170] (ungefähr)
```

---

## Initialisierung der Gewichte

Die Wahl der initialen Gewichte hat großen Einfluss auf das Training. Schlechte Initialisierung kann zu langsamer Konvergenz oder gar keinem Lernen führen.

### Probleme bei falscher Initialisierung

| Problem | Ursache | Konsequenz |
|---------|---------|------------|
| **Nullgewichte** | Alle Gewichte = 0 | Neuronen bleiben inaktiv |
| **Zu große Werte** | Gewichte >> 1 | Sättigung der Aktivierungsfunktion |
| **Symmetrische Gewichte** | Alle Gewichte identisch | Neuronen lernen dasselbe |

### Moderne Initialisierungsmethoden

```mermaid
flowchart TD
    INIT["Gewichts-Initialisierung"]
    
    INIT --> XAVIER["Xavier/Glorot<br/>Varianz = 2/(n_in + n_out)"]
    INIT --> HE["He-Initialisierung<br/>Varianz = 2/n_in"]
    
    XAVIER --> |"Sigmoid, Tanh"| SIGMOID_NET["Für Sigmoid/Tanh-Netze"]
    HE --> |"ReLU"| RELU_NET["Für ReLU-Netze"]
    
    style XAVIER fill:#c8e6c9
    style HE fill:#bbdefb
```

### Python-Implementierung mit Keras

```python
from tensorflow import keras
from tensorflow.keras import layers

# Xavier/Glorot-Initialisierung (Standard für Dense-Layer)
model = keras.Sequential([
    layers.Dense(64, activation='tanh', 
                 kernel_initializer='glorot_uniform'),  # Xavier
    layers.Dense(32, activation='tanh',
                 kernel_initializer='glorot_uniform'),
    layers.Dense(1, activation='linear')
])

# He-Initialisierung für ReLU-Netze
model_relu = keras.Sequential([
    layers.Dense(64, activation='relu',
                 kernel_initializer='he_uniform'),  # He
    layers.Dense(32, activation='relu',
                 kernel_initializer='he_uniform'),
    layers.Dense(1, activation='linear')
])
```

---

## Loss Functions

Die Loss-Funktion (Verlustfunktion) quantifiziert, wie weit die Vorhersagen des Modells von den tatsächlichen Werten abweichen. Sie ist das Optimierungsziel beim Training.

### Übersicht: Aufgabe → Loss Function

| Aufgabe | Loss Function | Beschreibung |
|---------|---------------|--------------|
| **Regression** | MSE (Mean Squared Error) | Mittlerer quadratischer Fehler |
| **Regression** | MAE (Mean Absolute Error) | Mittlerer absoluter Fehler |
| **Binäre Klassifikation** | Binary Cross-Entropy | Logarithmischer Verlust für 2 Klassen |
| **Multi-Class Klassifikation** | Categorical Cross-Entropy | Logarithmischer Verlust für n Klassen |

### Kombinationen: Aktivierung + Loss

```mermaid
flowchart TD
    TASK{"Aufgabe?"}
    
    TASK -->|Regression| REG["Output: Linear<br/>Loss: MSE oder MAE"]
    TASK -->|Binäre Klassifikation| BIN["Output: Sigmoid<br/>Loss: Binary Cross-Entropy"]
    TASK -->|Multi-Class| MULTI["Output: Softmax<br/>Loss: Categorical Cross-Entropy"]
    
    style REG fill:#e3f2fd
    style BIN fill:#ffcdd2
    style MULTI fill:#fff9c4
```

### Empfohlene Kombinationen für Supervised Learning

<br>


<img src="https://raw.githubusercontent.com/ralf-42/ML_Intro/main/07_image/activation_loss_function.png" class="logo" width="750"/>


### Keras-Implementierung

```python
from tensorflow import keras
from tensorflow.keras import layers

# Binäre Klassifikation
model_binary = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: Wahrscheinlichkeit 0-1
])
model_binary.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Loss für binäre Klassifikation
    metrics=['accuracy']
)

# Multi-Class Klassifikation (z.B. 10 Klassen)
model_multi = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output: Wahrscheinlichkeiten
])
model_multi.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Loss für Multi-Class
    metrics=['accuracy']
)

# Regression
model_regression = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')  # Output: beliebiger Wert
])
model_regression.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error
    metrics=['mae']
)
```

---

## Kombinationen für Unsupervised Learning

Neuronale Netze werden auch im unüberwachten Lernen eingesetzt:

| Use Case | Hidden Aktivierung | Output Aktivierung | Loss Function |
|----------|-------------------|-------------------|---------------|
| **Clustering** | ReLU, Leaky ReLU | Softmax | KL-Divergenz, Cosine Similarity |
| **Anomalieerkennung** | ReLU, Tanh, Sigmoid | Sigmoid | MSE, Binary Cross-Entropy |
| **Dimensionsreduktion** | ReLU, Leaky ReLU, Tanh | Linear, Sigmoid | MSE |
| **Generative Modelle (GAN)** | Leaky ReLU, SELU | Tanh (Generator), Sigmoid (Discriminator) | Binary Cross-Entropy |

---

## Best Practices

### Checkliste für neuronale Netze

- [ ] **Daten normalisieren**: StandardScaler oder MinMaxScaler verwenden
- [ ] **Passende Architektur**: Starte klein, erweitere bei Bedarf
- [ ] **ReLU in Hidden Layers**: Standard-Wahl für die meisten Aufgaben
- [ ] **Passende Output-Aktivierung**: Sigmoid (binär), Softmax (multi-class), Linear (Regression)
- [ ] **Passende Loss-Funktion**: Muss zur Aufgabe passen
- [ ] **Gewichtsinitialisierung**: He für ReLU, Xavier für Sigmoid/Tanh

### Häufige Fehler vermeiden

| Fehler | Problem | Lösung |
|--------|---------|--------|
| Sigmoid in allen Hidden Layers | Vanishing Gradients | ReLU verwenden |
| Softmax für binäre Klassifikation | Unnötige Komplexität | Sigmoid verwenden |
| Linear-Aktivierung in Hidden Layers | Keine Nichtlinearität | ReLU, Tanh verwenden |
| Falsche Loss-Funktion | Training konvergiert nicht | Loss an Aufgabe anpassen |
| Zu große Lernrate | Training instabil | Lernrate reduzieren |

---
## Weiterführende Ressourcen

- **StatQuest**: [Neural Networks Playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1) – Intuitive Erklärungen zu allen Grundlagen
- **3Blue1Brown**: [Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) – Visuelle Mathematik
- **Keras Dokumentation**: [Getting Started](https://keras.io/getting_started/) – Offizielle Anleitungen


---

**Version:** 1.0    
**Stand:** Januar 2026    
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.    