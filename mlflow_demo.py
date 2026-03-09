import mlflow
import joblib
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Keep runs in the local project folder so `mlflow ui` picks them up.
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Demo_Local_Run")
mlflow.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    db.data,
    db.target,
    test_size=0.2,
    random_state=42,
)

with mlflow.start_run(run_name="rf_diabetes") as run:
    # Create and train model.
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        max_features=3,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    # Autolog records most details, but metrics are logged explicitly for clarity.
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Export a portable model file for local serving or Hugging Face Spaces.
    joblib.dump(rf, "model.pkl")
    mlflow.log_artifact("model.pkl")

    print(f"Run ID: {run.info.run_id}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print("Saved model: model.pkl")