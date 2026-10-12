# Iris Flower Classification Website

A responsive Flask website for the Iris flower classification project. It predicts Iris-setosa, Iris-versicolor, or Iris-virginica from four measurements using a StandardScaler + K-Nearest Neighbors (KNN) model.

## Run on Windows

1. Install Python 3.10 or newer.
2. Extract this ZIP and open the extracted folder in VS Code or Command Prompt.
3. Create a virtual environment (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

4. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

5. Train the model and create `iris_model.joblib`:

   ```powershell
   python train_model.py
   ```

6. Start the website:

   ```powershell
   python app.py
   ```

7. Open the local address shown in the terminal, usually http://127.0.0.1:5000

## Notes

- The model is trained on scikit-learn's built-in Iris dataset; the website does not need to download a CSV.
- This is a local educational demo, not a deployed public website.
- If you want to use the exact model from your notebook, we can adapt `train_model.py` to load your existing notebook dataset and settings.
