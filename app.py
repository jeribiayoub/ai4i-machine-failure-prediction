import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier

# =========================================================
# 1. CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AI4I Predictive Maintenance",
    page_icon="",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ai4i2020_cleaning_source.csv"
API_MODEL_PATH = BASE_DIR / "api_model.joblib"

MODEL_NAMES = [
    "Logistic Regression", "Decision Tree", "KNN",
    "Random Forest", "XGBoost"
]
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]
FAILURE_LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


# =========================================================
# 2. DATA LOADING
# =========================================================
@st.cache_data
def load_local_dataset(path):
    return pd.read_csv(path)


@st.cache_data
def download_uci_dataset():
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=601)
    features = dataset.data.features
    targets = dataset.data.targets
    result = pd.concat([features, targets], axis=1)
    return result.loc[:, ~result.columns.duplicated()]


st.sidebar.title("AI4I Maintenance")
st.sidebar.caption("Interactive Classification Application")

data_source = st.sidebar.radio(
    "Dataset source", ["Bundled CSV", "Download from UCI", "Upload CSV"]
)
uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

try:
    if data_source == "Bundled CSV":
        if not DATA_PATH.exists():
            st.error(
                f"CSV not found: {DATA_PATH}. Place the CSV beside app.py "
                "or select another dataset source."
            )
            st.stop()
        df = load_local_dataset(str(DATA_PATH))
    elif data_source == "Download from UCI":
        with st.spinner("Downloading AI4I dataset from UCI..."):
            df = download_uci_dataset()
    else:
        if uploaded_file is None:
            st.info("Upload a CSV file to continue.")
            st.stop()
        df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Dataset loading failed: {error}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("The dataset must contain rows and at least two columns.")
    st.stop()

data_signature = int(pd.util.hash_pandas_object(df.head(50), index=True).sum())


# =========================================================
# 3. HELPERS
# =========================================================
def make_preprocessor(X_train):
    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_classifier(name, p, positive_weight, seed):
    if name == "Logistic Regression":
        return LogisticRegression(
            C=p["lr_c"], max_iter=2000, class_weight="balanced",
            random_state=seed
        )
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=p["tree_depth"],
            min_samples_split=p["tree_min_split"],
            class_weight="balanced", random_state=seed
        )
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=p["knn_neighbors"])
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=p["rf_estimators"], max_depth=p["rf_depth"],
            class_weight="balanced", random_state=seed, n_jobs=-1
        )
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=p["xgb_estimators"], max_depth=p["xgb_depth"],
            learning_rate=p["xgb_learning_rate"],
            scale_pos_weight=positive_weight, eval_metric="logloss",
            random_state=seed, n_jobs=1
        )
    raise ValueError(f"Unknown model: {name}")


def get_search_space(name, positive_weight):
    spaces = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0]
        },
        "Decision Tree": {
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10]
        },
        "KNN": {"classifier__n_neighbors": [3, 5, 7, 11, 15]},
        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5]
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.03, 0.1, 0.2],
            "classifier__scale_pos_weight": [1.0, 5.0, 10.0, float(positive_weight)]
        }
    }
    return spaces.get(name, {})


def positive_probabilities(model, X):
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Encoded positive class 1 is missing from the model.")
    return probs[:, classes.index(1)]


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities)
    }


def get_metrics_table(results, y_true, threshold):
    rows = []
    for name, result in results.items():
        rows.append({
            "Model": name,
            **calculate_metrics(y_true, result["probabilities"], threshold)
        })
    return pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)


# =========================================================
# 4. NAVIGATION
# =========================================================
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset Explorer",
        "Train & Compare",
        "Evaluation & Prediction",
        "Explainable AI",
        "Live Prediction",
    ],
)
st.sidebar.divider()


# =========================================================
# 5. DASHBOARD
# =========================================================
if page == "Dashboard":
    st.title("Predictive Maintenance Dashboard")
    st.write(
        "Analyze machine data, train classification models, and compare "
        "their ability to detect machine failures."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Rows", f"{len(df):,}")
    c2.metric("Dataset Columns", len(df.columns))
    c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset Structure")
    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique Values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    st.dataframe(info, use_container_width=True)
    st.subheader("Available Classification Models")
    st.write(", ".join(MODEL_NAMES))
    st.info("Go to Train & Compare to train and configure models.")


# =========================================================
# 6. DATASET EXPLORER
# =========================================================
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    tab1, tab2, tab3 = st.tabs([
        "Preview & Statistics", "Distributions & Correlations",
        "Positive-Class Rate"
    ])

    with tab1:
        row_count = st.slider(
            "Rows to display", 1, min(100, len(df)), min(10, len(df))
        )
        st.dataframe(df.head(row_count), use_container_width=True)
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)
        missing = pd.DataFrame({
            "Column": df.columns,
            "Missing": [int(df[c].isna().sum()) for c in df.columns]
        }).sort_values("Missing", ascending=False)
        st.subheader("Missing Values")
        st.dataframe(missing, use_container_width=True)

        target_dist = st.selectbox(
            "Target for class distribution",
            df.columns.tolist(),
            index=df.columns.get_loc("Machine failure")
            if "Machine failure" in df.columns else 0
        )
        counts = (
            df[target_dist].value_counts(dropna=False)
            .rename_axis("Class").reset_index(name="Count")
        )
        st.plotly_chart(
            px.bar(counts, x="Class", y="Count", text="Count",
                   title=f"Class Distribution: {target_dist}"),
            use_container_width=True
        )

    with tab2:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric:
            selected = st.selectbox("Feature distribution", numeric)
            st.plotly_chart(
                px.histogram(df, x=selected, title=f"Distribution of {selected}"),
                use_container_width=True
            )
            st.subheader("Correlation Heatmap")
            st.plotly_chart(
                px.imshow(
                    df[numeric].corr(), text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    title="Numerical Feature Correlations"
                ),
                use_container_width=True
            )
        else:
            st.info("No numerical columns are available.")

    with tab3:
        rate_target = st.selectbox(
            "Target variable", df.columns.tolist(), key="rate_target"
        )
        classes = sorted(df[rate_target].dropna().unique().tolist(), key=str)
        if len(classes) == 2:
            positive = st.selectbox(
                "Positive class", classes,
                index=classes.index(1) if 1 in classes else 1
            )
            feature_options = [c for c in df.columns if c != rate_target]
            feature = st.selectbox("Feature to analyze", feature_options)
            rate_df = df[[rate_target, feature]].dropna().copy()
            rate_df["_positive"] = (rate_df[rate_target] == positive).astype(int)
            if pd.api.types.is_numeric_dtype(rate_df[feature]):
                try:
                    rate_df["_group"] = pd.qcut(
                        rate_df[feature], q=10, duplicates="drop"
                    ).astype(str)
                except ValueError:
                    rate_df["_group"] = rate_df[feature].astype(str)
            else:
                rate_df["_group"] = rate_df[feature].astype(str)
            summary = (
                rate_df.groupby("_group", observed=True)["_positive"].mean()
                .mul(100).reset_index(name="Positive Rate (%)")
                .rename(columns={"_group": "Feature Value"})
            )
            st.plotly_chart(
                px.bar(summary, x="Feature Value", y="Positive Rate (%)",
                       text_auto=".2f", title="Positive-Class Rate by Feature"),
                use_container_width=True
            )
        else:
            st.info("Choose a binary target to calculate a positive-class rate.")


# =========================================================
# 7. TRAIN AND COMPARE
# =========================================================
elif page == "Train & Compare":
    st.title("Train & Compare Models")
    st.write("Choose a target and features, configure the split, and train models.")

    target_options = df.columns.tolist()
    target = st.selectbox(
        "Target variable", target_options,
        index=target_options.index("Machine failure")
        if "Machine failure" in target_options else 0
    )
    target_classes = sorted(df[target].dropna().unique().tolist(), key=str)
    if len(target_classes) != 2:
        st.error("Binary classification only: choose a target with exactly two classes.")
        st.stop()

    positive_class = st.selectbox(
        "Positive class (class of interest)", target_classes,
        index=target_classes.index(1) if 1 in target_classes else 1
    )

    candidate_features = [c for c in df.columns if c != target]
    excluded = set(IDENTIFIER_COLUMNS)
    if target == "Machine failure":
        excluded.update(FAILURE_LEAKAGE_COLUMNS)
    defaults = [c for c in candidate_features if c not in excluded]
    if not defaults:
        defaults = candidate_features
    selected_features = st.multiselect(
        "Explanatory features", candidate_features, default=defaults
    )

    if target == "Machine failure":
        leakage = [c for c in selected_features if c in FAILURE_LEAKAGE_COLUMNS]
        if leakage:
            st.warning(
                f"Potential target leakage: {leakage}. These columns describe "
                "failure modes and should normally be excluded."
            )

    st.subheader("Train/Test Split")
    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider("Test set proportion", 0.10, 0.40, 0.20, 0.05)
    with c2:
        seed = int(st.number_input(
            "Random seed", min_value=0, max_value=99999, value=42, step=1
        ))

    st.subheader("Models to Train")
    selected_models = st.multiselect(
        "Select one or more models", MODEL_NAMES, default=MODEL_NAMES
    )

    st.subheader("Hyperparameters")
    with st.expander("Configure model parameters", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            lr_c = st.select_slider(
                "Logistic Regression: C", [0.01, 0.1, 1.0, 10.0], value=1.0
            )
            tree_depth_opt = st.selectbox(
                "Decision Tree: max_depth", ["None", 5, 10, 20], index=2
            )
            tree_min_split = st.select_slider(
                "Decision Tree: min_samples_split", [2, 5, 10], value=2
            )
            knn_neighbors = st.slider("KNN: n_neighbors", 3, 25, 5, 2)
        with c2:
            rf_estimators = st.slider("Random Forest: n_estimators", 100, 500, 200, 50)
            rf_depth_opt = st.selectbox(
                "Random Forest: max_depth", ["None", 5, 10, 20], index=0
            )
            xgb_estimators = st.slider("XGBoost: n_estimators", 100, 500, 200, 50)
            xgb_depth = st.slider("XGBoost: max_depth", 2, 10, 5)
            xgb_lr = st.select_slider(
                "XGBoost: learning_rate", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
                value=0.1
            )

    tune = st.checkbox("Optimize hyperparameters with RandomizedSearchCV", value=False)
    search_iterations = st.slider(
        "Random search iterations per model", 2, 10, 3, disabled=not tune
    )
    st.caption(
        "Tuning uses 3-fold stratified cross-validation and average precision "
        "(PR-AUC) as its optimization metric."
    )

    if st.button("Train Selected Models", type="primary", use_container_width=True):
        if not selected_features:
            st.error("Select at least one explanatory feature.")
        elif not selected_models:
            st.error("Select at least one model.")
        else:
            data = df[selected_features + [target]].dropna(subset=[target]).copy()
            X = data[selected_features]
            y = (data[target] == positive_class).astype(int)
            counts = y.value_counts()
            if len(counts) != 2 or counts.min() < 2:
                st.error("Both classes need at least two observations.")
                st.stop()

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=seed, stratify=y
                )
            except Exception as error:
                st.error(f"Train/test split failed: {error}")
                st.stop()

            positive_weight = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
            params = {
                "lr_c": lr_c,
                "tree_depth": None if tree_depth_opt == "None" else int(tree_depth_opt),
                "tree_min_split": tree_min_split,
                "knn_neighbors": knn_neighbors,
                "rf_estimators": rf_estimators,
                "rf_depth": None if rf_depth_opt == "None" else int(rf_depth_opt),
                "xgb_estimators": xgb_estimators,
                "xgb_depth": xgb_depth,
                "xgb_learning_rate": xgb_lr
            }

            results, errors = {}, {}
            progress, status = st.progress(0.0), st.empty()

            for i, name in enumerate(selected_models):
                status.write(f"Training {name} ({i + 1}/{len(selected_models)})...")
                try:
                    pipe = Pipeline([
                        ("preprocessor", make_preprocessor(X_train)),
                        ("classifier", make_classifier(name, params, positive_weight, seed))
                    ])
                    best_params = {}
                    if tune:
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                        search = RandomizedSearchCV(
                            pipe,
                            param_distributions=get_search_space(name, positive_weight),
                            n_iter=search_iterations,
                            scoring="average_precision",
                            cv=cv,
                            random_state=seed,
                            n_jobs=-1,
                            refit=True,
                            error_score="raise"
                        )
                        search.fit(X_train, y_train)
                        pipe = search.best_estimator_
                        best_params = search.best_params_
                    else:
                        pipe.fit(X_train, y_train)

                    results[name] = {
                        "pipeline": pipe,
                        "probabilities": positive_probabilities(pipe, X_test),
                        "best_params": best_params
                    }
                except Exception as error:
                    errors[name] = str(error)
                progress.progress((i + 1) / len(selected_models))

            status.empty()
            progress.empty()

            if not results:
                st.error("No models trained successfully.")
                for name, message in errors.items():
                    st.error(f"{name}: {message}")
            else:
                st.session_state["model_results"] = results
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test
                st.session_state["target_name"] = target
                st.session_state["positive_class"] = str(positive_class)
                st.session_state["data_signature"] = data_signature
                st.session_state["selected_features"] = selected_features
                
                st.session_state["X_train"] = X_train

                # Remove explanations from the previous training run.
                st.session_state.pop("xai_global_result", None)
                st.session_state.pop("xai_local_result", None)
                st.session_state.pop("xai_local_index", None)
                st.session_state.pop("xai_permutation_result", None)
                

                if target == "Machine failure" and positive_class == 1 and "XGBoost" in results:
                    joblib.dump(results["XGBoost"]["pipeline"], API_MODEL_PATH)
                    st.success(f"XGBoost pipeline saved for FastAPI: {API_MODEL_PATH.name}")
                else:
                    st.info(
                        "To save the API model, train XGBoost with target "
                        "'Machine failure' and positive class 1."
                    )

                st.success(
                    f"Successfully trained {len(results)} model(s). "
                    "Open Evaluation & Prediction to compare them."
                )
                if errors:
                    st.warning("Some models failed:")
                    for name, message in errors.items():
                        st.error(f"{name}: {message}")
                for name, result in results.items():
                    if result["best_params"]:
                        st.write(f"**{name} best parameters:** {result['best_params']}")

    if (
        "model_results" in st.session_state
        and st.session_state.get("data_signature") == data_signature
    ):
        st.info("Trained models are available in Evaluation & Prediction.")


# =========================================================
# 8. EVALUATION AND PREDICTION
# =========================================================
elif page == "Evaluation & Prediction":
    st.title("Evaluation & Prediction")
    results = st.session_state.get("model_results")
    if not results:
        st.info("No trained models are available in this session. Go to Train & Compare first.")
        st.stop()
    if st.session_state.get("data_signature") != data_signature:
        st.warning("The dataset changed since training. Train the models again.")
        st.stop()

    y_test = st.session_state["y_test"]
    X_test = st.session_state["X_test"]
    st.write(
        f"Target: **{st.session_state['target_name']}** | "
        f"Positive class: **{st.session_state['positive_class']}** | "
        f"Test observations: **{len(y_test)}**"
    )

    threshold = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01)
    st.subheader("Model Comparison")
    metrics_df = get_metrics_table(results, y_test, threshold)
    st.dataframe(metrics_df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
        "F1": "{:.4f}", "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"
    }), use_container_width=True)

    st.success(f"Highest PR-AUC among trained models: {metrics_df.iloc[0]['Model']}")
    st.caption(
        "For imbalanced failure data, PR-AUC is especially useful. "
        "The highest PR-AUC model may differ from the highest-F1 model."
    )

    st.subheader("Metric Comparison")
    chosen_metrics = st.multiselect(
        "Metrics to visualize",
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        default=["Precision", "Recall", "F1", "PR-AUC"]
    )
    if chosen_metrics:
        plot_df = metrics_df.melt(
            id_vars="Model", value_vars=chosen_metrics,
            var_name="Metric", value_name="Score"
        )
        fig = px.bar(
            plot_df, x="Model", y="Score", color="Metric",
            barmode="group", title="Performance Comparison"
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("ROC Curves")
    roc_fig = go.Figure()
    for name, result in results.items():
        probs = result["probabilities"]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        roc_fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"
        ))
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
        line={"dash": "dash"}
    ))
    roc_fig.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        xaxis_range=[0, 1], yaxis_range=[0, 1], title="ROC Curves"
    )
    st.plotly_chart(roc_fig, use_container_width=True)

    st.subheader("Precision-Recall Curves")
    pr_fig = go.Figure()
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result["probabilities"])
        pr_fig.add_trace(go.Scatter(
            x=recall, y=precision, mode="lines", name=name
        ))
    pr_fig.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis_range=[0, 1], yaxis_range=[0, 1],
        title="Precision-Recall Curves"
    )
    st.plotly_chart(pr_fig, use_container_width=True)

    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Model to inspect", list(results.keys()))
    selected_probs = results[selected_model]["probabilities"]
    predictions = (selected_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    st.plotly_chart(px.imshow(
        cm,
        x=["Predicted Negative", "Predicted Positive"],
        y=["Actual Negative", "Actual Positive"],
        text_auto=True, color_continuous_scale="Blues",
        title=f"Confusion Matrix: {selected_model}"
    ), use_container_width=True)

    current = calculate_metrics(y_test, selected_probs, threshold)
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{current['Precision']:.3f}")
    c2.metric("Recall", f"{current['Recall']:.3f}")
    c3.metric("F1 Score", f"{current['F1']:.3f}")

    st.subheader("Test an Individual Observation")
    sample_index = st.slider("Select test observation", 0, len(X_test) - 1, 0)
    sample = X_test.iloc[[sample_index]]
    st.write("Input features")
    st.dataframe(sample, use_container_width=True)
    actual = int(y_test.iloc[sample_index])
    st.write(
        "Actual class:",
        st.session_state["positive_class"] if actual == 1 else "Negative class"
    )

    prediction_rows = []
    for name, result in results.items():
        probability = float(positive_probabilities(result["pipeline"], sample)[0])
        predicted = int(probability >= threshold)
        prediction_rows.append({
            "Model": name,
            "Positive-Class Probability": probability,
            "Predicted Class": (
                st.session_state["positive_class"] if predicted == 1
                else "Negative class"
            )
        })
    pred_df = pd.DataFrame(prediction_rows)
    st.dataframe(pred_df.style.format({
        "Positive-Class Probability": "{:.2%}"
    }), use_container_width=True)
    st.caption(
        "The threshold changes the predicted class, precision, recall, F1, "
        "and confusion matrix. ROC and PR curves use model probabilities."
    )


# =========================================================
# 9. LIVE PREDICTION VIA FASTAPI
# =========================================================


# =========================================================
# EXPLAINABLE AI
# =========================================================
elif page == "Explainable AI":
    import matplotlib.pyplot as plt

    from xai import (
        get_permutation_importance,
        calculate_shap,
        plot_shap_summary,
        plot_shap_dependence,
        plot_shap_waterfall,
        get_lime_explanation,
    )

    st.title("Explainable AI — Machine Failure Prediction")
    st.write(
        "Understand which features influence predictions globally "
        "and why the model predicts failure for an individual machine."
    )

    results = st.session_state.get("model_results")
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_test = st.session_state.get("y_test")

    if not results or X_train is None or X_test is None or y_test is None:
        st.warning(
            "No trained models are available. Open Train & Compare, "
            "train at least one model, then return here."
        )
        st.stop()

    if st.session_state.get("data_signature") != data_signature:
        st.warning("The dataset changed. Please retrain your models first.")
        st.stop()

    if (
        st.session_state.get("target_name") != "Machine failure"
        or st.session_state.get("positive_class") != "1"
    ):
        st.warning(
            "This XAI page is configured for the AI4I target "
            "'Machine failure', with class 1 meaning failure. "
            "Train with that target and positive class to continue."
        )
        st.stop()

    if "XGBoost" not in results:
        st.warning(
            "Train XGBoost as well to use the SHAP and LIME explanations. "
            "Permutation importance below can still compare the models "
            "you have trained."
        )

    tab_global, tab_local = st.tabs(
        ["Global explanation", "Individual machine explanation"]
    )

    # -----------------------------------------------------
    # GLOBAL EXPLANATION
    # -----------------------------------------------------
    with tab_global:
        st.header("1. Permutation importance")
        st.write(
            "This measures how much the model's average precision "
            "changes when one feature is shuffled. Larger positive "
            "values indicate a more useful feature for this evaluation."
        )

        available_models = list(results.keys())
        selected_models_xai = st.multiselect(
            "Models to compare",
            available_models,
            default=available_models,
            key="xai_permutation_models",
        )

        if st.button("Calculate permutation importance"):
            if not selected_models_xai:
                st.error("Select at least one model.")
            else:
                rows = []
                progress = st.progress(0.0)

                for i, model_name in enumerate(selected_models_xai):
                    importance_df = get_permutation_importance(
                        results[model_name]["pipeline"],
                        X_test,
                        y_test,
                        repeats=5,
                    )
                    importance_df["Model"] = model_name
                    rows.append(importance_df)
                    progress.progress((i + 1) / len(selected_models_xai))

                progress.empty()
                st.session_state["xai_permutation_result"] = pd.concat(
                    rows, ignore_index=True
                )

        permutation_df = st.session_state.get("xai_permutation_result")

        if permutation_df is not None:
            shown = permutation_df[
                permutation_df["Model"].isin(selected_models_xai)
            ]
            if not shown.empty:
                st.dataframe(
                    shown.sort_values(
                        ["Model", "Importance"], ascending=[True, False]
                    ).round(4),
                    use_container_width=True,
                )

                fig = px.bar(
                    shown,
                    x="Importance",
                    y="Feature",
                    color="Model",
                    barmode="group",
                    orientation="h",
                    title="Permutation Importance (Average Precision)",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.header("2. SHAP global explanation — XGBoost")

        if "XGBoost" in results:
            st.write(
                "SHAP estimates how each transformed feature contributes "
                "to the model output. The plots below use up to 300 "
                "test observations."
            )

            if st.button("Generate global SHAP plots"):
                try:
                    shap_sample = X_test.sample(
                        n=min(300, len(X_test)), random_state=42
                    )
                    with st.spinner("Calculating SHAP values..."):
                        shap_result = calculate_shap(
                            results["XGBoost"]["pipeline"],
                            X_train,
                            shap_sample,
                        )
                    st.session_state["xai_global_result"] = shap_result
                    st.success("Global SHAP values calculated.")
                except Exception as error:
                    st.error(f"SHAP calculation failed: {error}")

            global_result = st.session_state.get("xai_global_result")

            if global_result is not None:
                st.subheader("SHAP summary — feature impact")
                try:
                    fig = plot_shap_summary(global_result, plot_type="dot")
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)

                    fig = plot_shap_summary(global_result, plot_type="bar")
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)

                    st.subheader("SHAP dependence plot")
                    feature_names = global_result["feature_names"]
                    chosen_feature = st.selectbox(
                        "Feature to inspect",
                        feature_names,
                        key="xai_dependence_feature",
                    )
                    fig = plot_shap_dependence(
                        global_result, chosen_feature
                    )
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)

                except Exception as error:
                    st.error(f"Could not display a SHAP plot: {error}")

    # -----------------------------------------------------
    # LOCAL / INDIVIDUAL EXPLANATION
    # -----------------------------------------------------
    with tab_local:
        st.header("3. Explain one machine")

        if "XGBoost" not in results:
            st.info("Train XGBoost to enable the SHAP and LIME local explanations.")
        else:
            sample_index = st.slider(
                "Select a machine from the test set",
                min_value=0,
                max_value=len(X_test) - 1,
                value=0,
                key="xai_sample_index",
            )

            threshold = st.slider(
                "Failure classification threshold",
                min_value=0.01,
                max_value=0.99,
                value=0.50,
                step=0.01,
                key="xai_threshold",
            )

            sample = X_test.iloc[[sample_index]]
            pipeline = results["XGBoost"]["pipeline"]

            st.subheader("Machine measurements")
            st.dataframe(sample, use_container_width=True)

            probability = float(
                positive_probabilities(pipeline, sample)[0]
            )
            prediction = int(probability >= threshold)
            actual = int(y_test.iloc[sample_index])

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted failure probability", f"{probability:.2%}")
            col2.metric(
                "Prediction",
                "Failure" if prediction == 1 else "No failure",
            )
            col3.metric(
                "Actual test label",
                "Failure" if actual == 1 else "No failure",
            )

            st.caption(
                "SHAP and LIME explain the model's behavior; they do not "
                "prove that a feature physically caused a machine failure."
            )

            if st.button("Explain selected machine with SHAP and LIME"):
                try:
                    with st.spinner("Calculating local explanations..."):
                        local_shap = calculate_shap(
                            pipeline,
                            X_train,
                            sample,
                            max_background=100,
                        )
                        local_lime = get_lime_explanation(
                            pipeline,
                            X_train,
                            sample,
                            num_features=8,
                        )

                    st.session_state["xai_local_result"] = {
                        "shap": local_shap,
                        "lime": local_lime,
                    }
                    st.session_state["xai_local_index"] = sample_index

                except Exception as error:
                    st.error(f"Local explanation failed: {error}")

            local_result = st.session_state.get("xai_local_result")
            local_index = st.session_state.get("xai_local_index")

            if local_result is not None and local_index == sample_index:
                st.subheader("SHAP waterfall")
                try:
                    fig = plot_shap_waterfall(local_result["shap"])
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)

                    st.subheader("LIME explanation")
                    lime_figure = local_result["lime"].as_pyplot_figure(
                        label=1
                    )
                    st.pyplot(lime_figure, clear_figure=True)
                    plt.close(lime_figure)

                    st.subheader("Top SHAP contributions")
                    local_shap = local_result["shap"]
                    contribution_values = local_shap["values"][0]
                    names = local_shap["feature_names"]

                    contributions = pd.DataFrame({
                        "Feature": names,
                        "SHAP contribution": contribution_values,
                    })
                    contributions["Direction"] = np.where(
                        contributions["SHAP contribution"] > 0,
                        "Pushes model output toward failure",
                        "Pushes model output away from failure",
                    )
                    contributions["Absolute contribution"] = (
                        contributions["SHAP contribution"].abs()
                    )
                    contributions = contributions.sort_values(
                        "Absolute contribution", ascending=False
                    )

                    st.dataframe(
                        contributions[
                            [
                                "Feature",
                                "SHAP contribution",
                                "Direction",
                            ]
                        ].head(8).round(4),
                        use_container_width=True,
                    )

                    top_features = contributions.head(3)["Feature"].tolist()
                    st.write(
                        "**Main features influencing this model output:** "
                        + ", ".join(top_features)
                        + ". Check the plots to see the direction and size "
                        "of each contribution."
                    )

                    st.info(
                        "SHAP and LIME use different approximation methods, "
                        "so their feature rankings may differ. SHAP values "
                        "here explain the model output, not a causal effect."
                    )

                except Exception as error:
                    st.error(f"Could not display local explanations: {error}")
            elif local_result is not None:
                st.info(
                    "The selected machine changed. Click the explanation "
                    "button again to explain this machine."
                )


elif page == "Live Prediction":
    st.title("Live Prediction via FastAPI")
    st.write(
        "Enter machine operating measurements to predict whether a machine "
        "failure is likely."
    )
    st.caption(
        "This page uses the saved XGBoost pipeline through FastAPI. "
        "You do not need to retrain models in this Streamlit session."
    )

    if not API_MODEL_PATH.exists():
        st.warning(
            "The saved API model was not found. Go to Train & Compare, "
            "choose target 'Machine failure', positive class 1, select XGBoost, "
            "and train once."
        )

    st.divider()
    with st.form("api_prediction_form"):
        machine_type = st.selectbox("Machine type", ["L", "M", "H"], index=1)
        c1, c2 = st.columns(2)
        with c1:
            air_temp = st.number_input("Air temperature (K)", value=300.0)
            process_temp = st.number_input("Process temperature (K)", value=310.0)
            rotational_speed = st.number_input(
                "Rotational speed (rpm)", min_value=0, value=1500
            )
        with c2:
            torque = st.number_input("Torque (Nm)", min_value=0.0, value=40.0)
            tool_wear = st.number_input("Tool wear (min)", min_value=0, value=100)
        submit = st.form_submit_button(
            "Predict", type="primary", use_container_width=True
        )

    if submit:
        if not API_MODEL_PATH.exists():
            st.error("Cannot predict: api_model.joblib does not exist. Train XGBoost first.")
        else:

            payload = {
                "features": {
                    "Type": machine_type,
                    "Air temperature [K]": air_temp,
                    "Process temperature [K]": process_temp,
                    "Rotational speed [rpm]": rotational_speed,
                    "Torque [Nm]": torque,
                    "Tool wear [min]": tool_wear
                }
            }
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json=payload,
                    timeout=10
                )
                if response.ok:
                    result = response.json()
                    st.subheader("Prediction Result")
                    if result["prediction"] == 1:
                        st.error("Prediction: Machine failure")
                    else:
                        st.success("Prediction: No machine failure")
                    probability = result.get("probability_failure")
                    if probability is not None:
                        st.metric("Estimated failure probability", f"{probability:.2%}")
                        st.progress(max(0.0, min(1.0, probability)))
                    st.caption(
                        "This is a model estimate, not a guarantee that a failure "
                        "will or will not occur."
                    )
                else:
                    st.error(f"API error ({response.status_code}): {response.text}")
            except requests.RequestException as error:
                st.error(
                    "Could not connect to FastAPI. Start it in another terminal "
                    "with: python -m uvicorn api:app --reload"
                )
                st.caption(f"Technical details: {error}")


# =========================================================
# 10. FOOTER
# =========================================================
st.sidebar.divider()
st.sidebar.caption("AI4I 2020 Predictive Maintenance")
st.sidebar.caption("Supervised Binary Classification")
