# Generative KI Kurs

Ein umfassender deutschsprachiger Kurs zu praktischen Anwendungen von Generative KI Technologien.

# 1 📚 Kursübersicht

Dieser Kurs bietet eine praxisorientierte Einführung in moderne GenAI-Technologien mit Fokus auf OpenAI GPT-Modelle, Hugging Face und LangChain. Alle Materialien sind in deutscher Sprache verfasst und für die Ausführung in Google Colab optimiert.

# 2 ✨ Zielgruppe
- **Entwickler:innen** mit guten Python-Grundkenntnissen
- **IT-Fachkräfte**, die KI-Technologien in bestehende Projekte integrieren möchten
- **Technikbegeisterte Quereinsteiger:innen** mit Programmiererfahrung

# 3 🎯 Lernziele

- Verstehen der Grundlagen von Generative AI
- Praktische Anwendung von GPT-Modellen und OpenAI APIs
- Beherrschung des LangChain-Frameworks
- Entwicklung von RAG-Systemen (Retrieval Augmented Generation)
- Multimodale KI-Anwendungen (Text, Bild, Audio, Video)
- Einsatz lokaler und Open-Source-Modelle (Ollama)
- Fine-Tuning-Techniken und Modelloptimierung
- Entwicklung von KI-Agenten und Gradio-Benutzeroberflächen



# 4 📁 Repository-Struktur

```
GenAI/
├── README.md                     # Diese Datei
├── 01 ipynb/                   # 📚 Jupyter Notebooks (Hauptkursmaterialien)
│   ├── M00_Kurs_Intro.ipynb       # Kurseinführung und Überblick
│   ├── M01_GenAI_Intro.ipynb      # Einführung in Generative AI
│   ├── M02_Modellsteuerung_und_optimierung.ipynb
│   ├── M03_Codieren_mit_GenAI.ipynb
│   ├── M04_LangChain101.ipynb
│   ├── M05_LLM_Transformer.ipynb
│   ├── M06_Chat_Memory.ipynb
│   ├── M07_OutputParser.ipynb
│   ├── M08_Retrieval_Augmented_Generation.ipynb
│   ├── M09_Multimodal_Bild.ipynb
│   ├── M10_Agenten.ipynb
│   ├── M11_Gradio.ipynb
│   ├── M12_Lokale_Open_Source_Modelle.ipynb
│   ├── M13_SQL_RAG.ipynb
│   ├── M14_Multimodal_RAG.ipynb
│   ├── M15_Multimodal_Audio.ipynb
│   ├── M16_Multimodal_Video.ipynb
│   ├── M17_MCP_Model_Context_Protocol.ipynb
│   └── M18_Fine_Tuning.ipynb
├── 02 data/                    # 📊 Trainingsdaten und Beispieldateien
│   ├── biografien_1.txt           # Beispielbiografien für RAG
│   ├── biografien_2.md
│   ├── biografien_3.pdf
│   ├── biografien_4.docx
│   ├── customers.db               # SQLite Datenbank für SQL RAG
│   ├── northwind.db              # Beispiel-Datenbank
│   ├── mein_buch.pdf             # Beispiel-PDF für Textverarbeitung
│   ├── apfel.jpg                 # Beispielbilder für Bildverarbeitung
│   ├── people.jpg
│   ├── *_training_final.jsonl    # Fine-Tuning Datensätze
│   ├── *_testset_final.jsonl
│   └── *.mp3, *.mp4, *.wav       # Audio/Video-Beispiele
├── 03 doc/                     # 📖 Dokumentation und Ressourcen
│   └── GenAI_all_in_one.pdf       # Kompakte Kursübersicht
└── 04 model/                   # 🤖 Modellverzeichnis (derzeit leer)
```


# 5 📋 Kursstruktur

## 5.1 Basismodule (1-12) - Obligatorisch
| Modul | Thema              | Beschreibung                                             |
| ----- | ------------------ | -------------------------------------------------------- |
| M01   | GenAI Intro        | Überblick Generative AI, OpenAI, Hugging Face, LangChain |
| M02   | Modellsteuerung    | Prompting, Context Engineering, RAG, Fine-Tuning         |
| M03   | Codieren mit GenAI | Prompting für Code, Debugging, Grenzen                   |
| M04   | LangChain 101      | Architektur, Kernkonzepte, Best Practices                |
| M05   | LLM & Transformer  | Foundation Models, Transformer-Architektur               |
| M06   | Chat & Memory      | Kurz-/Langzeit-Memory, Externes Memory                   |
| M07   | Output Parser      | Strukturierte Ausgaben, JSON, Custom Parser              |
| M08   | RAG                | ChromaDB, Embeddings, Q&A über Dokumente                 |
| M09   | Multimodal Bild    | Bildgenerierung, In-/Outpainting, Klassifizierung        |
| M10   | Agents             | KI-Agenten, Architekturen, Multi-Agenten-Systeme         |
| M11   | Gradio             | UI-Entwicklung für KI-Anwendungen                        |
| M12   | Lokale Modelle     | Ollama, Open Source vs. Closed Source                    |

## 5.2 Erweiterungsmodule (13-23) - Fakultativ

| Modul | Thema                        | Beschreibung                                       |
| ----- | ---------------------------- | -------------------------------------------------- |
| M13   | SQL RAG                      | Integration von LLMs mit Datenbanken               |
| M14   | Multimodal RAG               | Text und Bilder kombiniert verarbeiten             |
| M15   | Multimodal Video             | Video-zu-Text, Video-Analyse, Objekterkennung      |
| M16   | Multimodal Audio             | Speech-to-Text, Text-to-Speech, Audio-Pipeline     |
| M17   | MCP - Model Context Protocol | Standardisiertes Protokoll für Tool-Einsatz        |
| M18   | Fine-Tuning                  | Parameter Efficient Fine-Tuning, Modellevaluierung |
| M19   | Modellauswahl und Evaluation | Systematische Modellauswahl und Bewertung          |
| M20   | Advanced Prompt Engineering  | Fortgeschrittene Prompt-Strategien und -Techniken  |
| M21   | Context Engineering          | Strategien für effektives Context Management       |
| M22   | EU AI Act / Ethik            | Rechtliche Compliance und ethische KI-Entwicklung  |
| M23   | KI-Challenge                 | Praktische Integration aller Kursmodule            |

# 6 🛠️ Technisches Setup

- Browser mit Internet-Zugang
- Goggle-Account
- Installation Google Colab


# 7 🌟 Entwicklungsumgebung
- **Plattform**: Google Colab 
- **Sprache**: Python 3.11+
- **Vorwissen**: Solide Python-Grundkenntnisse erforderlich

# 8 🔑 Erforderliche API-Schlüssel
- **Open-AI-Account:** : Für Zugang ChatGPT
- **OpenAI API Key**: Für GPT-Modelle (kostenpflichtig, bis zu 10 EUR für den gesamten Kurs)
- **Hugging Face Token**: Für Community-Modelle (kostenlos)


# 9 🔧 Verwendete Technologien

- **OpenAI**: GPT-4o-mini, Text-Embedding-3-small, DALL-E
- **LangChain**: Prompts, Chains, Parser, Runnables, Agents, ChromaDB Integration
- **Hugging Face**: Transformers, Community-Modelle, Tokenizer
- **ChromaDB**: Vektor-Datenbank für RAG-Systeme und Embeddings
- **Ollama**: Lokale Modellausführung (Llama, Mistral, etc.)
- **Gradio**: Benutzeroberflächen-Entwicklung für KI-Apps
- **Weitere**: MarkItDown, Unstructured, PyPDF, SQLite, ...

# 10 💡 Hinweise für Lernende

- Jedes Notebook ist eigenständig und kann unabhängig ausgeführt werden
- Umgebungssetup erfolgt automatisch über das `genai_lib` Utility-Paket
- Modifikation der Beispiele wird ausdrücklich als Lernübung empfohlen
- Fortschreitende Komplexität innerhalb jeder Modulreihe
- Praktische Übungen am Ende jedes Moduls
- Community-Support über GitHub Issues


# 11 ⚖️ Lizenz

Dieses Projekt steht unter der **MIT-Lizenz** (siehe `LICENSE`-Datei).

**MIT License - Copyright (c) 2024 Ralf**

Die Kursmaterialien können frei verwendet, modifiziert und weiterverbreitet werden.



