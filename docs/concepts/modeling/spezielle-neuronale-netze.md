---
layout: default
title: Spezielle NN
parent: Modeling
grand_parent: Konzepte
nav_order: 9
description: Computer Vision mit CNNs, Sequenzmodellierung mit RNNs und LSTMs sowie AutoEncoder für Dimensionsreduktion und Anomalieerkennung
has_toc: true
---

# Spezielle Neuronale Netze
{: .no_toc }

> **Computer Vision mit CNNs, Sequenzmodellierung mit RNNs und LSTMs sowie AutoEncoder für Dimensionsreduktion und Anomalieerkennung**

---

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Computer Vision

Computer Vision analysiert Bilder und Videos, um deren Inhalt zu verstehen oder geometrische Informationen zu extrahieren. Typische Aufgaben sind die Objekterkennung, Bildklassifizierung und Segmentierung.

### Bilder als Daten

Technisch interpretiert ein Computer-Vision-Modell Bilder als eine Reihe von **Pixeln**. Ein Pixel (Bildpunkt) bezeichnet den einzelnen Farbwert einer digitalen Rastergrafik.

<img src="https://raw.githubusercontent.com/ralf-42/ML_Intro/main/07_image/image_6.png" class="logo" width="350"/>

```mermaid
flowchart LR
    subgraph Bild["Digitales Bild"]
        A["28 × 28 Pixel"]
    end
    
    subgraph Matrix["Pixel-Matrix"]
        B["784 Werte<br/>(0-255)"]
    end
    
    subgraph Vektor["Input-Vektor"]
        C["Flattened<br/>Array"]
    end
    
    Bild --> Matrix --> Vektor
```

**Beispiel: MNIST-Datensatz**

Der MNIST-Datensatz enthält handgeschriebene Ziffern als monochromes Bild mit einer Größe von 28 × 28 Pixel. Jedes Pixel hat einen Grauwert zwischen 0 (schwarz) und 255 (weiß).

```python
from tensorflow import keras
import matplotlib.pyplot as plt

# MNIST-Datensatz laden
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Bildform prüfen
print(f"Bildgröße: {X_train[0].shape}")  # (28, 28)
print(f"Anzahl Trainingsbilder: {X_train.shape[0]}")  # 60000

# Beispielbild anzeigen
plt.imshow(X_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()
```

### Farbbilder und Kanäle

Während Graustufenbilder nur einen Kanal haben, bestehen Farbbilder aus mehreren Kanälen:

| Bildtyp    | Kanäle | Shape (Beispiel) |
| ---------- | ------ | ---------------- |
| Graustufen | 1      | (28, 28, 1)      |
| RGB        | 3      | (224, 224, 3)    |


```python
# Normalisierung der Pixelwerte (0-1 statt 0-255)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape für CNN (Kanal hinzufügen)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```

---

## Convolutional Neural Networks (CNN)

Convolutional Neural Networks sind mehrschichtige neuronale Netze, die besonders gut darin sind, Merkmale aus Bilddaten zu extrahieren. Sie funktionieren hervorragend mit Bildern und benötigen wenig Vorverarbeitung.

### Grundprinzip

CNNs nutzen zwei Kernkonzepte:

1. **Faltungen (Convolutions)**: Erkennen lokale Muster wie Kanten, Texturen oder komplexe Formen
2. **Pooling**: Reduzieren die räumliche Größe und extrahieren dominante Merkmale

```mermaid
flowchart LR
    subgraph Input["Input Layer"]
        A["Bild<br/>(28×28×1)"]
    end
    
    subgraph Feature["Feature Learning"]
        B["Convolution"]
        C["Pooling"]
        D["Convolution"]
        E["Pooling"]
    end
    
    subgraph Class["Classification"]
        F["Flatten"]
        G["Dense"]
        H["Output"]
    end
    
    A --> B --> C --> D --> E --> F --> G --> H
```

### CNN-Architektur im Detail

```mermaid
flowchart TD
    subgraph InputLayer["Input Layer"]
        I["Bild-Input<br/>Höhe × Breite × Kanäle"]
    end
    
    subgraph ConvBlock1["Convolutional Block 1"]
        C1["Conv2D<br/>Filter anwenden"]
        A1["Aktivierung<br/>(ReLU)"]
        P1["MaxPooling<br/>Downsampling"]
    end
    
    subgraph ConvBlock2["Convolutional Block 2"]
        C2["Conv2D<br/>Tiefere Features"]
        A2["Aktivierung<br/>(ReLU)"]
        P2["MaxPooling<br/>Weitere Reduktion"]
    end
    
    subgraph DenseBlock["Dense Block"]
        F["Flatten<br/>2D → 1D"]
        D1["Dense Layer<br/>Vollverbunden"]
        D2["Output Layer<br/>Softmax"]
    end
    
    I --> C1 --> A1 --> P1
    P1 --> C2 --> A2 --> P2
    P2 --> F --> D1 --> D2
```

### Layer-Typen erklärt

#### Input Layer

Die Eingabeschicht definiert die Bildmaße: Höhe, Breite und Anzahl der Farbkanäle.

```python
from tensorflow.keras.layers import Input

# Für MNIST (Graustufen)
input_layer = Input(shape=(28, 28, 1))

# Für Farbbilder
input_layer_rgb = Input(shape=(224, 224, 3))
```

#### Convolutional Layer

Der Convolutional Layer wendet Filter (Kernel) auf das Bild an, um **Merkmale zu erkennen**.

```mermaid
flowchart LR
    subgraph Input["Input (5×5)"]
        IM["1 0 1 0 1<br/>0 1 0 1 0<br/>1 0 1 0 1<br/>0 1 0 1 0<br/>1 0 1 0 1"]
    end
    
    subgraph Kernel["Filter (3×3)"]
        K["1 0 1<br/>0 1 0<br/>1 0 1"]
    end
    
    subgraph Output["Feature Map (3×3)"]
        O["Ergebnis der<br/>Faltungsoperation"]
    end
    
    Input --> Kernel --> Output
```

**Funktionsweise:**
1. Der Kernel "gleitet" über das Eingabebild
2. An jeder Position: elementweise Multiplikation und Summation
3. Das Ergebnis ist eine Feature Map

```python
from tensorflow.keras.layers import Conv2D

# Convolutional Layer mit 32 Filtern (3×3)
conv_layer = Conv2D(
    filters=32,           # Anzahl der Filter
    kernel_size=(3, 3),   # Größe des Filters
    activation='relu',    # Aktivierungsfunktion
    padding='same'        # Ausgabegröße = Eingabegröße
)
```


#### Pooling Layer

Das Pooling reduziert die **räumliche** Größe der Feature Maps und extrahiert dominante Merkmale.

| Pooling-Typ | Beschreibung | Verwendung |
|-------------|--------------|------------|
| **Max Pooling** | Nimmt den maximalen Wert | Standard, wirkt entrauschend |
| **Average Pooling** | Berechnet den Durchschnitt | Glättung der Features |
| **Global Average** | Ein Wert pro Feature Map | Vor dem Output Layer |

```python
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D

# Max Pooling (2×2 Fenster)
max_pool = MaxPooling2D(pool_size=(2, 2))

# Average Pooling
avg_pool = AveragePooling2D(pool_size=(2, 2))
```

#### Flatten Layer

Wandelt die mehrdimensionalen Feature Maps in einen eindimensionalen Vektor um.

```python
from tensorflow.keras.layers import Flatten

# Beispiel: (7, 7, 64) → (3136,)
flatten = Flatten()
```

#### Dense Layer

Vollständig verbundene Schichten für die finale Klassifizierung.

```python
from tensorflow.keras.layers import Dense

# Hidden Layer
dense1 = Dense(128, activation='relu')

# Output Layer (10 Klassen)
output = Dense(10, activation='softmax')
```


### CNN-Parameter verstehen

| Parameter | Beschreibung | Typische Werte |
|-----------|--------------|----------------|
| `filters` | Anzahl der Kernel/Filter | 32, 64, 128, 256 |
| `kernel_size` | Größe des Filters | (3, 3), (5, 5) |
| `strides` | Schrittweite des Filters | (1, 1), (2, 2) |
| `padding` | Randbehandlung | 'valid', 'same' |
| `activation` | Aktivierungsfunktion | 'relu' |

> **Best Practice**
>
> Beginne mit wenigen Filtern (32) und erhöhe die Anzahl in tieferen Schichten. Die räumliche Auflösung nimmt ab, während die Anzahl der Feature Maps zunimmt.

---

## Sequenzmodellierung

Sequenzmodellierung befasst sich mit Daten, bei denen die Reihenfolge wichtig ist. Dies umfasst zwei Hauptanwendungsbereiche:

1. **Natural Language Processing (NLP)** - Textverarbeitung
2. **Zeitreihenanalyse** - Temporale Daten

### Natural Language Processing (NLP)

NLP ist der Bereich, der sich mit der Verarbeitung und dem Verständnis menschlicher Sprache befasst.

```mermaid
flowchart LR
    subgraph Tasks["NLP-Aufgaben"]
        A["Spracherkennung<br/>(Speech-to-Text)"]
        B["Tagging<br/>(Wortarten)"]
        C["Named Entity<br/>Recognition"]
        D["Sentiment<br/>Analysis"]
        E["Textgenerierung<br/>(Text-to-Speech)"]
    end
```

**Typische NLP-Aufgaben:**

| Aufgabe | Beschreibung | Beispiel |
|---------|--------------|----------|
| **Spracherkennung** | Audio → Text | Diktiersoftware |
| **POS-Tagging** | Wortarten bestimmen | "Der [DET] Hund [NOUN] läuft [VERB]" |
| **NER** | Entitäten erkennen | "Berlin [LOC]", "Siemens [ORG]" |
| **Sentiment Analysis** | Stimmung erkennen | Positiv, Negativ, Neutral |
| **Textgenerierung** | Text erzeugen | ChatGPT, Claude |

### Text-Preprocessing

Das Preprocessing für NLP unterscheidet sich von der normalen Datenbereinigung:

```python
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(texts, max_words=10000, max_len=100):
    """
    Bereitet Texte für neuronale Netze vor.
    
    Parameters:
    -----------
    texts : list
        Liste von Texten
    max_words : int
        Maximale Anzahl der Wörter im Vokabular
    max_len : int
        Maximale Sequenzlänge
        
    Returns:
    --------
    tuple
        (padded_sequences, tokenizer)
    """
    # Normalisierung
    cleaned_texts = []
    for text in texts:
        # Kleinschreibung
        text = text.lower()
        # Sonderzeichen entfernen
        text = re.sub(r'[^\w\s]', '', text)
        # Mehrfache Leerzeichen entfernen
        text = re.sub(r'\s+', ' ', text).strip()
        cleaned_texts.append(text)
    
    # Tokenisierung
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(cleaned_texts)
    
    # Text zu Sequenzen
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    
    # Padding (gleiche Länge)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    return padded, tokenizer

# Beispiel
texts = ["Das ist ein Test.", "Machine Learning ist spannend!"]
sequences, tokenizer = preprocess_text(texts)
print(f"Vokabulargröße: {len(tokenizer.word_index)}")
```

### Zeitreihenanalyse

Zeitreihendaten sind Datenpunkte, die in zeitlicher Reihenfolge indiziert sind.

**Wichtige Eigenschaften:**

```mermaid
flowchart TD
    subgraph Eigenschaften["Zeitreihen-Eigenschaften"]
        T["Trend<br/>Langfristige Richtung"]
        S["Saisonalität<br/>Wiederkehrende Muster"]
        Z["Zyklische Effekte<br/>Unregelmäßige Schwankungen"]
        R["Rauschen<br/>Zufällige Variation"]
    end
    
    T --> |"beeinflusst"| Prognose
    S --> |"beeinflusst"| Prognose
    Z --> |"beeinflusst"| Prognose
    R --> |"erschwert"| Prognose
```

| Eigenschaft | Beschreibung | Beispiel |
|-------------|--------------|----------|
| **Trend** | Systematischer Anstieg/Abfall | Wirtschaftswachstum |
| **Saisonalität** | Regelmäßige Muster | Weihnachtsgeschäft |
| **Zyklische Effekte** | Unregelmäßige Schwankungen | Konjunkturzyklen |

---

## Recurrent Neural Networks (RNN)

Recurrent Neural Networks sind eine Architektur für die Verarbeitung von sequenziellen Daten. Sie haben eingebaute Rückkopplungsschleifen, die es ermöglichen, Informationen aus vorherigen Zeitschritten zu nutzen.

### Grundkonzept

```mermaid
flowchart LR
    subgraph Standard["Standard NN"]
        I1["Input"] --> H1["Hidden"] --> O1["Output"]
    end
    
    subgraph RNN["Recurrent NN"]
        I2["Input t"] --> H2["Hidden t"]
        H2 --> O2["Output t"]
        H2 --> |"Gedächtnis"| H2
    end
```

**Kernidee:** Ein RNN nutzt nicht nur die aktuelle Eingabe, sondern auch das "Gedächtnis" aus vorherigen Schritten.

### RNN aufgefaltet

Wenn man ein RNN über die Zeit "auffaltet", wird die Rückkopplungsschleife sichtbar:

```mermaid
flowchart LR
    subgraph t0["t=0"]
        X0["x₀"] --> H0["h₀"]
        H0 --> Y0["y₀"]
    end
    
    subgraph t1["t=1"]
        X1["x₁"] --> H1["h₁"]
        H1 --> Y1["y₁"]
    end
    
    subgraph t2["t=2"]
        X2["x₂"] --> H2["h₂"]
        H2 --> Y2["y₂"]
    end
    
    %% Bestehende Logik
    H0 --> |"State"| H1
    H1 --> |"State"| H2

    %% Erzwingen der Ausrichtung (falls nötig)
    %% t0 --- t1 --- t2 (Verbindung zwischen Subgraphen direkt)
    %% oder unsichtbar zwischen Nodes:
    X0 ~~~ X1 ~~~ X2
```


### Problem: Vanishing Gradients

RNNs haben Schwierigkeiten, langfristige Abhängigkeiten zu lernen. Bei langen Sequenzen werden die Gradienten während des Backpropagation entweder sehr klein (vanishing) oder sehr groß (exploding).

```mermaid
flowchart LR
    subgraph Problem["Vanishing Gradient Problem"]
        A["Wichtige Info<br/>am Anfang"] 
        B["..."]
        C["..."]
        D["..."]
        E["Vorhersage<br/>am Ende"]
    end
    
    A --> |"Gradient wird<br/>immer kleiner"| B --> C --> D --> E
```

**Lösung:** Long Short-Term Memory (LSTM)

---

## Long Short-Term Memory (LSTM)

LSTMs sind eine spezielle RNN-Architektur, die das Problem der verschwindenden Gradienten löst. Sie können Informationen über lange Zeiträume hinweg effektiv speichern und verarbeiten.

### LSTM-Zelle Architektur

Eine LSTM-Zelle enthält drei "Gates" (Tore), die den Informationsfluss steuern:

```mermaid
flowchart TD
    subgraph LSTM["LSTM-Zelle"]
        subgraph Gates["Gates"]
            F["Forget Gate<br/>Was vergessen?"]
            I["Input Gate<br/>Was speichern?"]
            O["Output Gate<br/>Was ausgeben?"]
        end
        
        C["Cell State<br/>Langzeitgedächtnis"]
        H["Hidden State<br/>Kurzzeitgedächtnis"]
    end
    
    Input["Input x_t"] --> F
    Input --> I
    Input --> O
    
    PrevH["Hidden h_(t-1)"] --> F
    PrevH --> I
    PrevH --> O
    
    F --> |"steuert"| C
    I --> |"aktualisiert"| C
    C --> |"beeinflusst"| H
    O --> |"steuert"| H
    
    H --> Output["Output y_t"]
```

### Die drei Gates erklärt

| Gate | Funktion | Analogie |
|------|----------|----------|
| **Forget Gate** | Entscheidet, welche Informationen aus dem Zellzustand gelöscht werden | "Was kann ich vergessen?" |
| **Input Gate** | Entscheidet, welche neuen Informationen gespeichert werden | "Was ist wichtig zu merken?" |
| **Output Gate** | Entscheidet, welche Informationen als Output verwendet werden | "Was ist jetzt relevant?" |



### LSTM vs. SimpleRNN

| Aspekt | SimpleRNN | LSTM |
|--------|-----------|------|
| **Architektur** | Einfache Rückkopplung | Drei Gates + Cell State |
| **Langzeitgedächtnis** | Schwach | Stark |
| **Trainingszeit** | Schneller | Langsamer |
| **Parameter** | Weniger | Mehr |
| **Anwendung** | Kurze Sequenzen | Lange Sequenzen |

> **Best Practice**
>
> Verwende LSTMs für die meisten Sequenzprobleme. Für sehr lange Sequenzen oder wenn Kontextverständnis in beide Richtungen wichtig ist, nutze bidirektionale LSTMs.

---

## AutoEncoder

Ein AutoEncoder ist ein spezieller Typ von neuronalem Netzwerk, das Daten auf ihre wichtigsten Merkmale komprimiert und dann wieder rekonstruiert.

### Architektur

```mermaid
flowchart LR
    subgraph Encoder["Encoder"]
        I["Input<br/>(784 dim)"]
        E1["Dense<br/>(256)"]
        E2["Dense<br/>(128)"]
        L["Latent Space<br/>(32 dim)"]
    end
    
    subgraph Decoder["Decoder"]
        D1["Dense<br/>(128)"]
        D2["Dense<br/>(256)"]
        O["Output<br/>(784 dim)"]
    end
    
    I --> E1 --> E2 --> L
    L --> D1 --> D2 --> O
    
    I -.-> |"Ziel: Output ≈ Input"| O
```

**Komponenten:**

| Komponente | Funktion | Beschreibung |
|------------|----------|--------------|
| **Encoder** | Komprimierung | Reduziert Eingabe auf wesentliche Merkmale |
| **Latent Space** | Repräsentation | Komprimierte Darstellung der Daten |
| **Decoder** | Rekonstruktion | Erzeugt Ausgabe aus komprimierter Darstellung |

### Anwendungsbereiche

1. **Dimensionsreduktion**: Alternative zu PCA
2. **Anomalieerkennung**: Normale Daten rekonstruieren besser
3. **Rauschunterdrückung**: Denoising AutoEncoder
4. **Generative Modelle**: Variational AutoEncoder (VAE)

### Anomalieerkennung mit AutoEncoder

AutoEncoder eignen sich hervorragend zur Anomalieerkennung: Anomalien werden schlechter rekonstruiert als normale Daten.

### Denoising AutoEncoder

Ein Denoising AutoEncoder lernt, verrauschte Eingaben zu bereinigen:


---

## Zusammenfassung

```mermaid
flowchart TD
    subgraph SNN["Spezielle Neuronale Netze"]
        CNN["CNN<br/>Bildverarbeitung"]
        RNN["RNN/LSTM<br/>Sequenzen"]
        AE["AutoEncoder<br/>Komprimierung"]
    end
    
    subgraph Tasks["Aufgaben"]
        T1["Bildklassifizierung"]
        T2["Objekterkennung"]
        T3["NLP"]
        T4["Zeitreihen"]
        T5["Anomalieerkennung"]
        T6["Dimensionsreduktion"]
    end
    
    CNN --> T1
    CNN --> T2
    RNN --> T3
    RNN --> T4
    AE --> T5
    AE --> T6
```

| Architektur | Stärke | Typische Anwendung |
|-------------|--------|-------------------|
| **CNN** | Lokale Muster in 2D-Daten | Bilder, Videos |
| **RNN** | Sequenzielle Abhängigkeiten | Kurze Sequenzen |
| **LSTM** | Langzeit-Abhängigkeiten | Text, Zeitreihen |
| **AutoEncoder** | Komprimierung & Rekonstruktion | Anomalien, Rauschen |

> **Ausblick**
>
> Moderne Architekturen wie **Transformer** (BERT, GPT) haben in vielen NLP-Aufgaben LSTMs abgelöst. Für Computer Vision sind **Vision Transformer (ViT)** und fortgeschrittene CNNs wie **ResNet** oder **EfficientNet** Stand der Technik.

---


**Version:** 1.0     
**Stand:** Januar 2026     
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.     