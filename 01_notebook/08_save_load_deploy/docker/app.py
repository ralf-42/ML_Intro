"""Gradio-App für die Diamantenpreis-Schätzung.

Die App ist aus dem Notebook `b840_data_app_gradio_pipeline_diamonds.ipynb`
abgeleitet und lädt eine gespeicherte scikit-learn-Pipeline.
"""

from pathlib import Path

import gradio as gr
import joblib
from pandas import DataFrame


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diamonds_pipeline.joblib"


def load_model():
    """Lädt die gespeicherte Pipeline aus dem App-Verzeichnis."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelldatei nicht gefunden: {MODEL_PATH}. "
            "Die Datei diamonds_pipeline.joblib muss neben app.py liegen."
        )
    return joblib.load(MODEL_PATH)


model = load_model()

cut_seq = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_seq = ["J", "I", "H", "G", "F", "E", "D"]
clarity_seq = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


def predict_diamonds(
    carat: float,
    cut: str,
    color: str,
    clarity: str,
    depth: float,
    table: float,
) -> str:
    """Schätzt den Diamantenpreis mit der geladenen Pipeline."""
    feature_vector = DataFrame(
        [[carat, cut, color, clarity, depth, table]],
        columns=["carat", "cut", "color", "clarity", "depth", "table"],
    )
    prediction = model.predict(feature_vector)
    return f"{prediction[0]:,.2f} $"


def create_interface() -> gr.Interface:
    """Erzeugt die Gradio-Oberfläche."""
    return gr.Interface(
        fn=predict_diamonds,
        inputs=[
            gr.Slider(label="Carat", minimum=0.2, maximum=5.0, step=0.1, value=0.7),
            gr.Radio(label="Cut", choices=cut_seq, value="Very Good"),
            gr.Dropdown(label="Color", choices=color_seq, value="F"),
            gr.Dropdown(label="Clarity", choices=clarity_seq, value="SI1"),
            gr.Slider(
                label="Total Depth Percentage",
                minimum=43.0,
                maximum=79.0,
                step=0.5,
                value=63.5,
            ),
            gr.Slider(label="Table", minimum=43.0, maximum=95.0, step=0.5, value=56.0),
        ],
        outputs=gr.Textbox(label="Schätzpreis"),
        title="Schätzpreis für Diamanten",
        description=(
            '<p style="text-align: center";>'
            "Demo, liefert möglicherweise ungenaue oder falsche Informationen "
            "über Preise </br><b>- Alle Angaben ohne Gewähr - </b></p>"
        ),
        allow_flagging="never",
    )


interface = create_interface()


if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
