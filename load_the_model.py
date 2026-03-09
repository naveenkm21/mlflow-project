import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes

RUN_ID = "42613daa65704dfb9e9518e724404401"

# Use the local tracking folder used by mlflow_demo.py.
mlflow.set_tracking_uri("file:./mlruns")

# With sklearn autolog, the model artifact path is typically "model".
model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")

# Diabetes model expects 10 input features.
sample = load_diabetes().data[:1]
prediction = model.predict(sample)

print("Prediction:", prediction)