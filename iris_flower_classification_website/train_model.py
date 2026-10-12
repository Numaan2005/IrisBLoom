"""Train and save the Iris flower classification model."""
import joblib
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.30, random_state=42, stratify=iris.target
)

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")

joblib.dump(model, "iris_model.joblib")
print("Saved model to iris_model.joblib")
