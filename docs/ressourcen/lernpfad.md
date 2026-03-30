---
layout: default
title: Lernpfad
parent: Ressourcen
nav_order: 1
description: "Empfohlene Reihenfolge durch die produktiven ML_Intro-Notebooks"
has_toc: true
---

# Lernpfad
{: .no_toc }

> **Welche Reihenfolge ergibt für den Kurs den klarsten Lernfortschritt?** <br>
> Der Lernpfad ordnet die produktiven Notebooks aus `ML_Intro/01_notebook` zu einem nachvollziehbaren Kursverlauf.

---

# Inhaltsverzeichnis
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Ziel des Lernpfads

Der Kurs folgt der klassischen ML-Logik: Problem verstehen, Daten vorbereiten, Modell trainieren, Ergebnis bewerten und das Ergebnis in eine nutzbare Form überführen. Genau diese Abfolge bildet der Lernpfad ab. Er dient nicht als vollständiges Verzeichnis aller Notebooks, sondern als empfohlene Reihenfolge für ein konsistentes Lernerlebnis.

## Kernpfad

Der Kernpfad deckt die zentrale Lernlinie des Kurses ab:

1. `00_general/b000_launch.ipynb`
2. `00_general/b020_pandas_basics.ipynb`
3. `00_general/b040_datasets.ipynb`
4. `01_supervised/b110_sl_dt_titanic.ipynb`
5. `01_supervised/b120_sl_lr_mpg.ipynb`
6. `02_unsupervised/b200_ul_kmeans_dbscan_location.ipynb`
7. `02_unsupervised/b220_ul_dbscan_nid.ipynb`
8. `02_unsupervised/b240_ul_pca_special.ipynb`
9. `03_network/b310_nn_mlp_cancer.ipynb`
10. `03_network/b320_nn_keras_cancer.ipynb`
11. `04_ensemble/b400_sl_rf_diamonds_inverse.ipynb`
12. `04_ensemble/b410_xg_cancer.ipynb`
13. `04_ensemble/b430_stacking_titanic.ipynb`
14. `05_tuning/b510_cv_dt_titanic.ipynb`
15. `05_tuning/b520_bootstrapping_dt_titanic.ipynb`
16. `05_tuning/b530_gridsearch_nn_mlp_cancer.ipynb`
17. `05_tuning/b570_roc_auc_threshold.ipynb`
18. `06_workflow/b610_pipeline_dt_diamonds.ipynb`
19. `09_diverse/b910_data_app_gradio_diamonds.ipynb`
20. `09_diverse/b940_save_load_dt_diamonds.ipynb`
21. `09_diverse/b950_save_load_rf_diamonds.ipynb`

Diese Reihenfolge bildet den Grundbogen des Kurses ab: Einstieg, erste Modelle, unsupervised Verfahren, neuronale Netze, Ensembles, Bewertung, Workflow und Transfer.

## Vertiefung nach Themenfeld

### Regression und tabellarische Daten

Für stärkere Regressionsthemen und Modellvergleiche eignen sich zusätzlich:

- `03_network/b330_nn_mlp_diamonds.ipynb`
- `03_network/b340_nn_keras_diamonds.ipynb`
- `04_ensemble/b420_xg_diamonds.ipynb`
- `05_tuning/b516_cv_rf_diamonds.ipynb`
- `05_tuning/b550_randomizedsearch_rf_diamonds.ipynb`

### Unsupervised und Mustererkennung

Für zusätzliche Mustererkennung und Assoziationsanalyse:

- `02_unsupervised/b230_ul_apriori_food.ipynb`

### Deep Learning und Spezialthemen

Für Vision, NLP, Zeitreihen und Autoencoder:

- `07_special/b710_vision_keras_mnist.ipynb`
- `07_special/b715_vision_yolo.ipynb`
- `07_special/b720_nlp_keras_spam.ipynb`
- `07_special/b730_ts_keras_wetter.ipynb`
- `07_special/b740_ts_chronos_wetter.ipynb`
- `07_special/b750_autoencoder_nid.ipynb`

### XAI, Anwendungen und KI-Integration

Für erklärbare KI, Anwendungen und neuere KI-Bezüge:

- `09_diverse/b900_xai_frameworks_titanic.ipynb`
- `09_diverse/b930_save_load_keras_sinus_multi.ipynb`
- `09_diverse/b960_analyse_credit_data_with_gemini_chat.ipynb`
- `09_diverse/b970_analyse_traffic_mit_gemini_chat.ipynb`
- `09_diverse/b980_ai_integration.ipynb`

## Kompakte Empfehlungen

### Für Einsteiger

Sinnvoll ist ein reduzierter Pfad mit Datenbasis, ersten Modellen und Workflow:

1. `b000_launch`
2. `b020_pandas_basics`
3. `b040_datasets`
4. `b110_sl_dt_titanic`
5. `b120_sl_lr_mpg`
6. `b200_ul_kmeans_dbscan_location`
7. `b310_nn_mlp_cancer`
8. `b400_sl_rf_diamonds_inverse`
9. `b510_cv_dt_titanic`
10. `b610_pipeline_dt_diamonds`

### Für schnelle Praxisorientierung

Wer zügig vom Datensatz zur anwendungsnahen Umsetzung kommen will, kann auf diesen verkürzten Pfad gehen:

1. `b040_datasets`
2. `b110_sl_dt_titanic`
3. `b400_sl_rf_diamonds_inverse`
4. `b510_cv_dt_titanic`
5. `b610_pipeline_dt_diamonds`
6. `b910_data_app_gradio_diamonds`

### Für fortgeschrittene Teilnehmende

Für stärkere Vergleichs-, Tuning- und Spezialthemen:

1. `b430_stacking_titanic`
2. `b530_gridsearch_nn_mlp_cancer`
3. `b550_randomizedsearch_rf_diamonds`
4. `b740_ts_chronos_wetter`
5. `b900_xai_frameworks_titanic`
6. `b980_ai_integration`

## Mindestpfad

Wenn ein stark reduzierter Kernpfad benötigt wird, reichen diese acht Notebooks:

1. `b020_pandas_basics`
2. `b040_datasets`
3. `b110_sl_dt_titanic`
4. `b120_sl_lr_mpg`
5. `b200_ul_kmeans_dbscan_location`
6. `b400_sl_rf_diamonds_inverse`
7. `b510_cv_dt_titanic`
8. `b610_pipeline_dt_diamonds`

Damit sind Datenzugang, supervised und unsupervised Verfahren, klassische Modelle, Bewertung und Workflow im Kern abgedeckt.

## Einordnung zu den übrigen Ressourcen

Der Lernpfad beantwortet die Reihenfolgefrage. Die Seite [Interaktive Visualisierung](./interaktive-visualisierung.html) ergänzt den Kurs dort, wo Begriffe visuell klarer werden. Die Seite [Links](./link_sammlung.html) dient als externe Vertiefung für Videos, Dokumentationen und Werkzeuge.

---

**Version:** 1.0<br>
**Stand:** März 2026<br>
**Kurs:** Machine Learning. Verstehen. Anwenden. Gestalten.
