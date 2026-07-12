# Credit Card Fraud Detection: A Cost-Based Approach

Fraud detection is a deceptively hard problem. Fraud is extremely rare (under 0.2% of transactions), which means a model can be 99.8% accurate while catching almost no fraud at all. Accuracy is the wrong lens.

This project works through the problem the way a risk team would: not "how accurate is the model?" but **"what does each type of mistake cost the business, and where should we set the decision boundary?"**

## Links

- **Live interactive dashboard**: https://fraud-detection--dashboard.streamlit.app


## What's inside

- **`notebooks/fraud_detection.ipynb`** — the full analysis: exploring the class imbalance, exposing why accuracy misleads, handling imbalance, comparing logistic regression against XGBoost, and choosing a model and decision threshold by business cost rather than accuracy.
- **`app.py`** — an interactive Streamlit dashboard where you set the cost of a missed fraud vs a false alarm and watch the cost-optimal decision update live, score real transactions, view feature importance, and read a recommendation that adapts to your cost assumptions.

## The approach

1. Explore the data and confront the class imbalance
2. Build a baseline model and show why accuracy is misleading on rare events
3. Handle the imbalance with class weighting, and watch it overcorrect into false alarms
4. Train a stronger model (XGBoost) that balances the tradeoff
5. Assign real dollar costs to errors and choose the model on cost, not accuracy
6. Tune the decision threshold to the cost-optimal point

The key finding: the highest-recall model is not the best one. The model that minimises real business cost is, and the decision threshold is a lever that should be tuned deliberately with the cost of each error in mind.

## Tech stack

Python, pandas, scikit-learn, XGBoost, matplotlib, and Streamlit.

## Running it locally

The raw dataset is not included in this repo because it is large (about 150 MB).

1. Download the **Credit Card Fraud Detection** dataset from Kaggle (ULB / Machine Learning Group) and place `creditcard.csv` in a `data/` folder at the project root.
2. Create a virtual environment and install the dependencies:
   ```
   python -m venv venv
   venv\Scripts\Activate.ps1        # on Windows
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost jupyter streamlit joblib
   ```
3. Run the notebook to reproduce the analysis (and regenerate the model artifacts), or launch the dashboard directly:
   ```
   streamlit run app.py
   ```

## A note on the data

The features `V1`-`V28` are anonymised (PCA-transformed) to protect real cardholders' privacy, so they can't be interpreted individually. This is realistic: real fraud data is almost always de-identified. Only `Time` and `Amount` are in their original form.

## Possible future improvements

- Serve the model behind a small API rather than loading it directly in the app
- Add a lightweight database to log flagged transactions for review
- Experiment with additional resampling strategies (SMOTE, undersampling) and compare on cost