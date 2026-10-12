from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path(__file__).parent / "iris_model.joblib"
model = joblib.load(MODEL_PATH)

FLOWER_NAMES = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica",
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None
    values = {}

    if request.method == "POST":
        fields = {
            "sepal_length": "Sepal length (cm)",
            "sepal_width": "Sepal width (cm)",
            "petal_length": "Petal length (cm)",
            "petal_width": "Petal width (cm)",
        }
        try:
            measurements = []
            for key, label in fields.items():
                raw = request.form.get(key, "").strip()
                values[key] = raw
                number = float(raw)
                if number <= 0 or number > 10:
                    raise ValueError(f"{label} must be greater than 0 and no more than 10 cm.")
                measurements.append(number)

            sample = np.array([measurements])
            predicted_class = int(model.predict(sample)[0])
            prediction = FLOWER_NAMES[predicted_class]

            if hasattr(model, "predict_proba"):
                confidence = round(float(np.max(model.predict_proba(sample))) * 100, 1)

        except ValueError as exc:
            error = str(exc) or "Please enter valid numeric measurements."
        except Exception:
            error = "Something went wrong. Check the measurements and try again."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error,
        values=values,
    )

if __name__ == "__main__":
    app.run(debug=True)