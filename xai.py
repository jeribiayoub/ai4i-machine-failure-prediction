
# xai.py
"""Explainability utilities for the AI4I predictive maintenance project."""

import numpy as np
import pandas as pd
import shap

from scipy import sparse
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance


def to_dense(data):
    """Convert transformed sparse matrices to NumPy arrays when necessary."""
    if sparse.issparse(data):
        return data.toarray()
    return np.asarray(data)


def get_transformed_data(pipeline, X):
    """Apply the already-fitted preprocessing step."""
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    return to_dense(transformed), feature_names


def get_permutation_importance(pipeline, X_test, y_test, repeats=5):
    """
    Calculate permutation importance on held-out data.
    Average precision is useful for the imbalanced failure target.
    """
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=42,
        n_jobs=1,
    )

    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": result.importances_mean,
        "Std": result.importances_std,
    })

    return importance.sort_values("Importance", ascending=False)



def calculate_shap(pipeline, X_background, X_explain, max_background=100):
    """
    Explain the predicted probability of Machine failure = 1
    for a fitted tree-based classifier inside the pipeline.
    """
    if X_background.empty:
        raise ValueError("X_background must contain at least one row.")
    if X_explain.empty:
        raise ValueError("X_explain must contain at least one row.")
    if max_background < 1:
        raise ValueError("max_background must be at least 1.")

    classifier = pipeline.named_steps["classifier"]

    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError(
            f"Positive class 1 is missing. Model classes: {classes}"
        )
    positive_index = classes.index(1)

    background = X_background.sample(
        n=min(max_background, len(X_background)),
        random_state=42,
    )

    background_transformed, feature_names = get_transformed_data(
        pipeline, background
    )
    explain_transformed, explain_feature_names = get_transformed_data(
        pipeline, X_explain
    )

    if feature_names != explain_feature_names:
        raise ValueError("Background and explanation features do not match.")

    # Use the training background and explain probability, not raw margin.
    explainer = shap.TreeExplainer(
        classifier,
        data=background_transformed,
        feature_perturbation="interventional",
        model_output="probability",
    )
    explanation = explainer(explain_transformed)

    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    # Handle SHAP versions that return values for multiple classes.
    if values.ndim == 3:
        if values.shape[-1] <= positive_index:
            raise ValueError(
                f"Cannot select positive class from SHAP shape {values.shape}."
            )
        values = values[:, :, positive_index]
    elif values.ndim != 2:
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}")

    # Normalize expected-value shape for the waterfall plot.
    if base_values.ndim == 0:
        base_values = np.repeat(base_values, len(X_explain))
    elif base_values.ndim == 2:
        if base_values.shape[1] > positive_index:
            base_values = base_values[:, positive_index]
        else:
            base_values = base_values[:, 0]

    base_values = np.asarray(base_values).reshape(-1)

    if len(base_values) == 1 and len(X_explain) > 1:
        base_values = np.repeat(base_values, len(X_explain))

    if values.shape != (len(X_explain), len(feature_names)):
        raise ValueError(
            "SHAP output dimensions do not match the explained rows/features: "
            f"{values.shape}"
        )
    if len(base_values) != len(X_explain):
        raise ValueError(
            "SHAP base values do not match the number of explained rows."
        )

    return {
        "values": values,
        "base_values": base_values,
        "data": explain_transformed,
        "feature_names": feature_names,
        "explainer": explainer,
        "output_scale": "probability",
        "positive_class": 1,
    }



def plot_shap_summary(shap_result, plot_type="dot"):
    """Draw a global SHAP summary plot."""
    import matplotlib.pyplot as plt

    plt.figure()
    shap.summary_plot(
        shap_result["values"],
        shap_result["data"],
        feature_names=shap_result["feature_names"],
        plot_type=plot_type,
        show=False,
    )
    fig = plt.gcf()
    return fig


def plot_shap_dependence(shap_result, feature_name):
    """Draw a SHAP dependence plot for one transformed feature."""
    import matplotlib.pyplot as plt

    plt.figure()
    shap.dependence_plot(
        feature_name,
        shap_result["values"],
        shap_result["data"],
        feature_names=shap_result["feature_names"],
        show=False,
    )
    return plt.gcf()


def plot_shap_waterfall(shap_result, row_index=0):
    """Draw a local SHAP waterfall plot."""
    import matplotlib.pyplot as plt

    local_explanation = shap.Explanation(
        values=shap_result["values"][row_index],
        base_values=float(shap_result["base_values"][row_index]),
        data=shap_result["data"][row_index],
        feature_names=shap_result["feature_names"],
    )

    plt.figure()
    shap.plots.waterfall(local_explanation, show=False, max_display=12)
    return plt.gcf()


def get_lime_explanation(pipeline, X_background, X_instance, num_features=8):
    """
    Generate a LIME explanation using transformed features and the
    classifier inside the fitted pipeline.
    """
    classifier = pipeline.named_steps["classifier"]

    background_transformed, feature_names = get_transformed_data(
        pipeline, X_background
    )
    instance_transformed, _ = get_transformed_data(
        pipeline, X_instance
    )

    lime_explainer = LimeTabularExplainer(
        training_data=background_transformed,
        feature_names=feature_names,
        class_names=["No failure", "Failure"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    explanation = lime_explainer.explain_instance(
        data_row=instance_transformed[0],
        predict_fn=classifier.predict_proba,
        num_features=min(num_features, len(feature_names)),
        labels=(1,),
    )

    return explanation
