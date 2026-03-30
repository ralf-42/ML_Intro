# Lernpfad ML_Intro

> Empfohlene Reihenfolge für den produktiven Notebook-Bestand in `ML_Intro/01_notebook`.

Dieser Lernpfad trennt zwischen:

- **Kernpfad**: für ein konsistentes Grundverständnis des ML-Workflows
- **Vertiefung**: für Spezialisierung, Vergleich und Transfer

`_misc` ist nicht Teil des Lernpfads.

---

## 1. Ziel des Lernpfads

Der Kurs folgt der Logik:

1. Problem verstehen
2. Daten vorbereiten
3. Modell trainieren
4. Ergebnis bewerten
5. Modell in Anwendung oder Transfer überführen

Die Reihenfolge der Notebooks sollte diese Logik sichtbar machen. Nicht jedes Notebook ist Voraussetzung für jedes andere. Der Kernpfad bildet daher nur die wichtigste Linie ab.

---

## 2. Kernpfad

### Phase 1: Einstieg und Werkzeugbasis

1. [b000_launch.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/00_general/b000_launch.ipynb)
2. [b020_pandas_basics.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/00_general/b020_pandas_basics.ipynb)
3. [b040_datasets.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/00_general/b040_datasets.ipynb)

Ziel:
- Arbeitsumgebung und Datenlogik verstehen
- Tabellenstrukturen lesen, filtern und vorbereiten
- typische ML-Datensätze und Aufgabentypen einordnen

### Phase 2: Erste supervised Modelle

4. [b110_sl_dt_titanic.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/01_supervised/b110_sl_dt_titanic.ipynb)
5. [b120_sl_lr_mpg.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/01_supervised/b120_sl_lr_mpg.ipynb)

Ziel:
- Klassifikation und Regression unterscheiden
- erste Modellierungsentscheidungen nachvollziehen
- zentrale Begriffe wie Split, Features, Zielvariable und Metrik verankern

### Phase 3: Unsupervised Learning

6. [b200_ul_kmeans_dbscan_location.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/02_unsupervised/b200_ul_kmeans_dbscan_location.ipynb)
7. [b220_ul_dbscan_nid.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/02_unsupervised/b220_ul_dbscan_nid.ipynb)
8. [b240_ul_pca_special.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/02_unsupervised/b240_ul_pca_special.ipynb)

Ziel:
- Lernen ohne Labels einordnen
- Clustering, Anomalieerkennung und Dimensionsreduktion unterscheiden
- Grenzen von Interpretierbarkeit und Bewertung verstehen

### Phase 4: Neuronale Netze

9. [b310_nn_mlp_cancer.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/03_network/b310_nn_mlp_cancer.ipynb)
10. [b320_nn_keras_cancer.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/03_network/b320_nn_keras_cancer.ipynb)

Ziel:
- Unterschied zwischen klassischem MLP und Keras-Workflow verstehen
- Trainingslogik, Loss, Epochen und Modellkomplexität einordnen

### Phase 5: Ensembles und stärkere Modelle

11. [b400_sl_rf_diamonds_inverse.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/04_ensemble/b400_sl_rf_diamonds_inverse.ipynb)
12. [b410_xg_cancer.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/04_ensemble/b410_xg_cancer.ipynb)
13. [b430_stacking_titanic.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/04_ensemble/b430_stacking_titanic.ipynb)

Ziel:
- Ensemble-Idee verstehen
- Random Forest, Boosting und Stacking gegeneinander abgrenzen
- Leistungsgewinn gegen Komplexität abwägen

### Phase 6: Bewertung und Tuning

14. [b510_cv_dt_titanic.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/05_tuning/b510_cv_dt_titanic.ipynb)
15. [b520_bootstrapping_dt_titanic.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/05_tuning/b520_bootstrapping_dt_titanic.ipynb)
16. [b530_gridsearch_nn_mlp_cancer.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/05_tuning/b530_gridsearch_nn_mlp_cancer.ipynb)
17. [b570_roc_auc_threshold.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/05_tuning/b570_roc_auc_threshold.ipynb)

Ziel:
- robuste Bewertung von punktueller Modellleistung unterscheiden
- Hyperparameter-Tuning methodisch einordnen
- Thresholds und Metriken als Geschäftsentscheidung verstehen

### Phase 7: Workflow und Transfer

18. [b610_pipeline_dt_diamonds.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/06_workflow/b610_pipeline_dt_diamonds.ipynb)
19. [b910_data_app_gradio_diamonds.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/09_diverse/b910_data_app_gradio_diamonds.ipynb)
20. [b940_save_load_dt_diamonds.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/09_diverse/b940_save_load_dt_diamonds.ipynb)
21. [b950_save_load_rf_diamonds.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/09_diverse/b950_save_load_rf_diamonds.ipynb)

Ziel:
- vom Einzelmodell zum reproduzierbaren Workflow übergehen
- Modelle speichern, laden und in kleine Anwendungen überführen

---

## 3. Vertiefung nach Themenfeld

### Regression und strukturierte Tabellendaten

- [b120_sl_lr_mpg.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/01_supervised/b120_sl_lr_mpg.ipynb)
- [b330_nn_mlp_diamonds.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/03_network/b330_nn_mlp_diamonds.ipynb)
- [b340_nn_keras_diamonds.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/03_network/b340_nn_keras_diamonds.ipynb)
- [b420_xg_diamonds.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/04_ensemble/b420_xg_diamonds.ipynb)
- [b516_cv_rf_diamonds.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/05_tuning/b516_cv_rf_diamonds.ipynb)
- [b550_randomizedsearch_rf_diamonds.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/05_tuning/b550_randomizedsearch_rf_diamonds.ipynb)

### Klassifikation und Imbalance

- [b110_sl_dt_titanic.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/01_supervised/b110_sl_dt_titanic.ipynb)
- [b410_xg_cancer.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/04_ensemble/b410_xg_cancer.ipynb)
- [b570_roc_auc_threshold.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/05_tuning/b570_roc_auc_threshold.ipynb)

### Unsupervised und Mustererkennung

- [b200_ul_kmeans_dbscan_location.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/02_unsupervised/b200_ul_kmeans_dbscan_location.ipynb)
- [b220_ul_dbscan_nid.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/02_unsupervised/b220_ul_dbscan_nid.ipynb)
- [b230_ul_apriori_food.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/02_unsupervised/b230_ul_apriori_food.ipynb)
- [b240_ul_pca_special.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/02_unsupervised/b240_ul_pca_special.ipynb)

### Deep Learning und Spezialthemen

- [b710_vision_keras_mnist.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/07_special/b710_vision_keras_mnist.ipynb)
- [b715_vision_yolo.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/07_special/b715_vision_yolo.ipynb)
- [b720_nlp_keras_spam.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/07_special/b720_nlp_keras_spam.ipynb)
- [b730_ts_keras_wetter.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/07_special/b730_ts_keras_wetter.ipynb)
- [b740_ts_chronos_wetter.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/07_special/b740_ts_chronos_wetter.ipynb)
- [b750_autoencoder_nid.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/07_special/b750_autoencoder_nid.ipynb)

### Erklärbarkeit, Anwendungen und KI-Integration

- [b900_xai_frameworks_titanic.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/09_diverse/b900_xai_frameworks_titanic.ipynb)
- [b910_data_app_gradio_diamonds.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/09_diverse/b910_data_app_gradio_diamonds.ipynb)
- [b930_save_load_keras_sinus_multi.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/09_diverse/b930_save_load_keras_sinus_multi.ipynb)
- [b960_analyse_credit_data_with_gemini_chat.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/09_diverse/b960_analyse_credit_data_with_gemini_chat.ipynb)
- [b970_analyse_traffic_mit_gemini_chat.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/09_diverse/b970_analyse_traffic_mit_gemini_chat.ipynb)
- [b980_ai_integration.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/09_diverse/b980_ai_integration.ipynb)

---

## 4. Kompakte Empfehlungen nach Zielgruppe

### Für Einsteiger

Empfohlene Reihenfolge:

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

Empfohlene Reihenfolge:

1. `b040_datasets`
2. `b110_sl_dt_titanic`
3. `b400_sl_rf_diamonds_inverse`
4. `b510_cv_dt_titanic`
5. `b610_pipeline_dt_diamonds`
6. `b910_data_app_gradio_diamonds`

### Für fortgeschrittene Teilnehmende

Empfohlene Reihenfolge:

1. `b430_stacking_titanic`
2. `b530_gridsearch_nn_mlp_cancer`
3. `b550_randomizedsearch_rf_diamonds`
4. `b740_ts_chronos_wetter`
5. `b900_xai_frameworks_titanic`
6. `b980_ai_integration`

---

## 5. Mindestpfad für Kurskonsistenz

Wenn ein reduzierter Kernpfad benötigt wird, reichen diese 8 Notebooks:

1. [b020_pandas_basics.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/00_general/b020_pandas_basics.ipynb)
2. [b040_datasets.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/00_general/b040_datasets.ipynb)
3. [b110_sl_dt_titanic.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/01_supervised/b110_sl_dt_titanic.ipynb)
4. [b120_sl_lr_mpg.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/01_supervised/b120_sl_lr_mpg.ipynb)
5. [b200_ul_kmeans_dbscan_location.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/02_unsupervised/b200_ul_kmeans_dbscan_location.ipynb)
6. [b400_sl_rf_diamonds_inverse.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/04_ensemble/b400_sl_rf_diamonds_inverse.ipynb)
7. [b510_cv_dt_titanic.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/05_tuning/b510_cv_dt_titanic.ipynb)
8. [b610_pipeline_dt_diamonds.ipynb](/C:/Users/ralfb/OneDrive/Desktop/Kurse/ML_Intro/01_notebook/06_workflow/b610_pipeline_dt_diamonds.ipynb)

Damit ist der Grundbogen des Kurses abgedeckt:

- Daten
- supervised / unsupervised
- klassische Modelle
- Bewertung
- Workflow

---

## 6. Hinweise zur Weiterentwicklung

Wenn der Kurs künftig stärker vereinheitlicht wird, sollte der Lernpfad sichtbar in drei Ebenen gegliedert werden:

- **Pflichtstoff**
- **empfohlene Vertiefung**
- **optionale Spezialthemen**

Sinnvoll wäre außerdem, den Kernpfad direkt in den Docs zu spiegeln, damit Notebook-Reihenfolge und Dokumentationsstruktur besser zusammenlaufen.

---

**Version:** 1.0  
**Stand:** März 2026  
**Projekt:** ML_Intro
