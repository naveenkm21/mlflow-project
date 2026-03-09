import joblib
import gradio as gr
from sklearn.datasets import load_diabetes


MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)
feature_names = load_diabetes().feature_names


def predict(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):
    values = [[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]]
    pred = model.predict(values)[0]
    return f"Predicted diabetes progression score: {pred:.3f}"


inputs = [
    gr.Number(label=feature_names[0], value=0.03),
    gr.Number(label=feature_names[1], value=0.05),
    gr.Number(label=feature_names[2], value=0.06),
    gr.Number(label=feature_names[3], value=0.02),
    gr.Number(label=feature_names[4], value=-0.04),
    gr.Number(label=feature_names[5], value=-0.03),
    gr.Number(label=feature_names[6], value=-0.04),
    gr.Number(label=feature_names[7], value=-0.01),
    gr.Number(label=feature_names[8], value=0.02),
    gr.Number(label=feature_names[9], value=-0.02),
]

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction"),
    title="Diabetes Progression Predictor",
    description="RandomForestRegressor trained on sklearn diabetes dataset.",
)


if __name__ == "__main__":
    app.launch(share=True)
