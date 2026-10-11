# AI4I Predictive Maintenance

An interactive machine-learning application for predicting machine
failures using the **AI4I 2020 Predictive Maintenance** dataset. The
application includes data exploration, preprocessing, comparison of five
classifiers, evaluation, individual predictions, and a FastAPI endpoint.

## Features

-   Dashboard with dataset overview, shape, missing-value count, data
    types, and preview.
-   Dataset Explorer with descriptive statistics, class distributions,
    numeric histograms, correlation heatmap, and positive-class rates by
    feature.
-   Configurable binary target, explanatory features, train/test split,
    and random seed.
-   Five classifiers: Logistic Regression, Decision Tree, K-Nearest
    Neighbors (KNN), Random Forest, and XGBoost.
-   Numerical preprocessing: median imputation and standardization.
-   Categorical preprocessing: most-frequent imputation and one-hot
    encoding.
-   Optional `RandomizedSearchCV` tuning with stratified
    cross-validation and average precision scoring.
-   Evaluation with Accuracy, Precision, Recall, F1-score, ROC-AUC,
    PR-AUC, ROC and precision-recall curves, confusion matrix, and
    decision threshold.
-   FastAPI prediction endpoint returning a predicted class and, where
    supported, estimated failure probability.

## Dataset and leakage prevention

The default local file is `ai4i2020_cleaning_source.csv`, placed beside
`app.py`. The application also offers UCI download and CSV upload
options.

For the AI4I `Machine failure` target, the recommended predictors are:

-   `Type`
-   `Air temperature [K]`
-   `Process temperature [K]`
-   `Rotational speed [rpm]`
-   `Torque [Nm]`
-   `Tool wear [min]`

Do not use `UDI` or `Product ID` as predictive features. Exclude `TWF`,
`HDF`, `PWF`, `OSF`, and `RNF` when predicting `Machine failure`,
because they describe failure modes and may cause target leakage. The
interface warns if these columns are manually selected; verify the final
feature list before reporting results.

## Project structure

``` text
AI4I_Predictive_Maintenance/
├── app.py
├── api.py
├── ai4i2020_cleaning_source.csv
├── requirements.txt
├── api_model.joblib   # generated after qualifying XGBoost training
└── README.md
```

The Streamlit app saves `api_model.joblib` when XGBoost is successfully
trained with target `Machine failure` and positive class `1`.

## Installation

Use a compatible Python version and install the packages listed in
`requirements.txt`. The project needs at least: `streamlit`, `pandas`,
`numpy`, `plotly`, `scikit-learn`, `xgboost`, `joblib`, `requests`,
`fastapi`, and `uvicorn`. The `ucimlrepo` package is also needed for the
UCI download option.

Create and activate a virtual environment in the project directory.

**Windows (PowerShell):**

``` powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux:**

``` bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If any package listed above is missing from `requirements.txt`, add it
before installation.

## Run Streamlit

In the project directory:

``` bash
streamlit run app.py
```

The default source expects `ai4i2020_cleaning_source.csv` beside
`app.py`. Alternatively, select **Download from UCI** or **Upload CSV**
in the sidebar.

## Run FastAPI

Open a **second terminal** in the same project directory and activate
the same virtual environment:

``` bash
python -m uvicorn api:app --reload
```

By default, the API runs at `http://127.0.0.1:8000`.

-   API root: <http://127.0.0.1:8000/>
-   Health check: <http://127.0.0.1:8000/health>
-   Interactive documentation: <http://127.0.0.1:8000/docs>

Before the first API prediction, train XGBoost in **Train & Compare**
with target `Machine failure` and positive class `1`. The model is saved
to `api_model.joblib`. Restart the API after replacing the saved model
so it loads the latest artifact.

### Example prediction request

The API expects the original input feature names required by the trained
pipeline, including units:

``` json
{
  "features": {
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 100
  }
}
```

A prediction is a model estimate, not a guarantee of future machine
behavior.

## Evaluation notes

-   Use a stratified train/test split so both classes are represented.
-   Failures are the minority class, so accuracy alone can be
    misleading. Review recall, precision, F1, PR-AUC, and the confusion
    matrix together.
-   Select the decision threshold using validation data, not the final
    test set. Report the feature list, split, random seed, threshold,
    model parameters, and library versions for reproducibility.
-   Test the local CSV route and UCI download route in a fresh
    environment, and verify `/health`, `/docs`, and `/predict`.

## Limitations and submission checklist

-   Module 5 Explainable AI (SHAP/LIME and global/local explanations) is
    not implemented in this version.
-   The API URL in the app is configured for local development. A
    deployed frontend and API need an appropriate reachable API URL;
    `127.0.0.1` is not automatically a remote server address.
-   The public dataset may not represent every real machine or operating
    environment. Validate on recent operational data before production
    use.
-   [ ] Include `app.py`, `api.py`, `requirements.txt`, README,
    notebook, and one-page PDF in the repository.
-   [ ] Add screenshots of the dashboard, model comparison, evaluation
    plots, and successful API prediction.
-   [ ] Exclude virtual environments, caches, secrets, and unnecessary
    generated files.
-   [ ] Document how to obtain the dataset and regenerate
    `api_model.joblib`.
