import json
import time
from html import escape
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from supervised_algorithm.k_nn import k_nn
from supervised_algorithm.linear_regression import linear_regression
from supervised_algorithm.naives_bayes import fit_naive_bayes, predict_naive_bayes
from supervised_algorithm.rosenblatt_perecptron import (
    fit_perceptron,
    predict_perceptron,
)


#Application parameters
model_names = [
    "Rosenblatt Perceptron",
    "Gaussian Naive Bayes",
    "K-Nearest Neighbors",
    "Linear Regression",
]

model_descriptions = {
    "Rosenblatt Perceptron": "Binary linear classifier updated after each misclassified example.",
    "Gaussian Naive Bayes": "Probabilistic classifier based on Gaussian feature distributions.",
    "K-Nearest Neighbors": "Classifier that predicts from the labels of the closest points.",
    "Linear Regression": "Regression model trained with batch gradient descent.",
}

known_dataset_names = [
    "iris", "wine", "breast-w", "diabetes", "heart-statlog", "vehicle",
    "phoneme", "ecoli", "glass", "ionosphere", "segment", "sonar",
    "spambase", "waveform-5000", "wdbc", "yeast", "boston", "house_16H",
    "cpu_act", "elevators", "kin8nm",
]

classification_dataset_names = {
    "iris",
    "wine",
    "breast-w",
    "heart-statlog",
    "vehicle",
    "phoneme",
    "ecoli",
    "glass",
    "ionosphere",
    "segment",
    "sonar",
    "spambase",
    "waveform-5000",
    "wdbc",
    "yeast",
}

max_training_rows = 5000


st.set_page_config(page_title="ML LAB // RETRO", page_icon="▣", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');
    .stApp { background: #101820; color: #f6f1d1; font-family: 'VT323', monospace; font-size: 20px; }
    h1, h2, h3 { color: #ffcc66 !important; font-family: 'Press Start 2P', monospace !important; text-shadow: 3px 3px #7b2d26; }
    .hero { border-left: 6px solid #71d99e; background: linear-gradient(135deg, #182632, #203544); padding: 24px 28px; margin: 12px 0 24px; box-shadow: 8px 8px #7b2d26; }
    .hero-kicker, .small-label { color: #71d99e; letter-spacing: 3px; font-size: 17px; }
    .hero-copy { color: #d8d2b5; font-size: 24px; margin-top: 12px; }
    .panel { border: 1px solid #456273; background: #14212b; padding: 18px; margin: 10px 0 18px; }
    .empty-state { border: 1px dashed #456273; color: #a9b6ae; background: #14212b; padding: 14px 18px; margin: 12px 0; font-size: 19px; }
    .prediction-result { position: relative; overflow: hidden; border: 2px solid #ffcc66; background: #182632; padding: 18px 22px; margin: 18px 0; box-shadow: 6px 6px #7b2d26; animation: prediction-enter 0.45s ease-out both; }
    .prediction-result::after { content: ""; position: absolute; top: 0; left: -35%; width: 28%; height: 100%; background: rgba(255, 204, 102, 0.16); transform: skewX(-18deg); animation: prediction-scan 1.4s ease-out 0.15s both; pointer-events: none; }
    .prediction-label { color: #71d99e; font-size: 17px; letter-spacing: 3px; }
    .prediction-value { color: #ffcc66; font-family: 'Press Start 2P', monospace; font-size: 22px; margin-top: 10px; animation: prediction-pulse 1.2s ease-out 0.45s both; }
    .comparison-heading { color: #71d99e; border-left: 5px solid #ffcc66; background: #182632; padding: 12px 16px; margin: 20px 0 12px; font-family: 'Press Start 2P', monospace; font-size: 14px; letter-spacing: 1px; box-shadow: 4px 4px #7b2d26; }
    @keyframes prediction-enter { from { opacity: 0; transform: translateY(10px) scale(0.98); } to { opacity: 1; transform: translateY(0) scale(1); } }
    @keyframes prediction-scan { from { left: -35%; } to { left: 125%; } }
    @keyframes prediction-pulse { 0%, 100% { text-shadow: 0 0 0 transparent; } 50% { text-shadow: 0 0 12px #ffcc66; } }
    [data-testid="stSidebar"] { background: #182632; border-right: 3px solid #ffcc66; }
    [data-testid="stMetric"] { background: #182632; border: 2px solid #71d99e; padding: 12px; }
    .stButton > button { background: #7b2d26; color: #f6f1d1; border: 2px solid #ffcc66; border-radius: 0; font-family: 'VT323', monospace; font-size: 22px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def load_openml_catalog():
    #Load active OpenML dataset metadata
    url = "https://www.openml.org/api/v1/json/data/list/status/active/limit/10000"
    with urlopen(url, timeout=30) as response:
        content = json.loads(response.read().decode("utf-8"))

    catalog = []
    for dataset in content["data"]["dataset"]:
        qualities = {
            item["name"]: float(item["value"])
            for item in dataset.get("quality", [])
        }
        if qualities.get("NumberOfSymbolicFeatures", 0) != 0:
            continue

        number_of_classes = qualities.get("NumberOfClasses")
        dataset_type = (
            "classification"
            if (
                (number_of_classes is not None and number_of_classes > 0)
                or dataset["name"] in classification_dataset_names
            )
            else "regression"
        )
        catalog.append(
            {
                "did": int(dataset["did"]),
                "name": dataset["name"],
                "type": dataset_type,
                "instances": int(qualities.get("NumberOfInstances", 0)),
                "features": int(qualities.get("NumberOfFeatures", 0)),
                "classes": int(qualities.get("NumberOfClasses", 0)),
            }
        )
    return catalog


@st.cache_data
def load_openml_dataset(dataset_id):
    #Extract features and labels
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    X_frame = data.data
    target = data.target

    if X_frame is None:
        raise ValueError("Features are not available.")

    #Some OpenML datasets do not expose a target through sklearn
    if target is None:
        full_frame = data.frame
        if full_frame is None or full_frame.shape[1] < 2:
            raise ValueError("Target is not available.")
        target = full_frame.iloc[:, -1]
        X_frame = full_frame.iloc[:, :-1]

    numeric_features = X_frame.select_dtypes(include="number").copy()
    if numeric_features.shape[1] == 0:
        raise ValueError("No numerical features are available.")

    X = numeric_features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    T = np.asarray(target)

    #Convert numeric targets represented as strings
    numeric_target = pd.to_numeric(pd.Series(T), errors="coerce")
    if numeric_target.notna().mean() >= 0.95:
        T = numeric_target.to_numpy(dtype=float)

    valid_rows = np.isfinite(X).all(axis=1) & pd.notna(T)
    if T.dtype.kind in "biufc":
        valid_rows &= np.isfinite(T)

    X = X[valid_rows]
    T = T[valid_rows]
    if len(X) < 10:
        raise ValueError("The dataset has too few valid rows.")
    if len(np.unique(T)) < 2:
        raise ValueError("The target has only one usable value.")

    return X, T, list(numeric_features.columns), getattr(target, "name", None) or "target"


def get_compatible_datasets(catalog, model_name):
    #Select datasets compatible by task and OpenML metadata
    if model_name == "Linear Regression":
        datasets = [
            item
            for item in catalog
            if item["type"] == "regression"
            and item["name"] not in classification_dataset_names
        ]
    else:
        datasets = [item for item in catalog if item["type"] == "classification"]
        if model_name == "Rosenblatt Perceptron":
            datasets = [item for item in datasets if item["classes"] == 2]

    #Put known datasets first, then larger datasets
    datasets.sort(
        key=lambda item: (
            known_dataset_names.index(item["name"])
            if item["name"] in known_dataset_names
            else len(known_dataset_names),
            -item["instances"],
        )
    )
    return datasets[:50]


def prepare_training_data(X, T):
    #Keep the interface responsive on very large OpenML datasets
    if len(X) <= max_training_rows:
        return X, T
    random_state = np.random.RandomState(42)
    rows = random_state.choice(len(X), max_training_rows, replace=False)
    return X[rows], T[rows]


def split_dataset(X, T):
    #Stratified split is only possible when every class has at least two samples
    unique_classes, class_counts = np.unique(T, return_counts=True)
    stratify = T if len(unique_classes) > 1 and np.min(class_counts) >= 2 else None
    return train_test_split(
        X,
        T,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )


def fit_model(model_name, dataset_id, k):
    start_time = time.perf_counter()
    X, T, feature_names, target_name = load_openml_dataset(dataset_id)
    X, T = prepare_training_data(X, T)

    if model_name == "Linear Regression":
        T = pd.to_numeric(pd.Series(T), errors="coerce").to_numpy(dtype=float)
        valid_rows = np.isfinite(T)
        X, T = X[valid_rows], T[valid_rows]
        if len(X) < 10:
            raise ValueError("The regression dataset has too few numeric rows.")
        X_train, X_test, T_train, T_test = split_dataset(X, T)
        theta, _, _, epochs, feature_mean, feature_std = linear_regression(
            X_train, T_train.reshape(-1, 1)
        )
        X_test = (X_test - feature_mean) / feature_std
        X_test = np.hstack((X_test, np.ones((len(X_test), 1))))
        predictions = (X_test @ theta).ravel()
        metrics = {
            "MSE": mean_squared_error(T_test, predictions),
            "R2": r2_score(T_test, predictions),
            "Epochs": epochs,
        }
        model = (theta, feature_mean, feature_std)
    else:
        if model_name == "Rosenblatt Perceptron" and len(np.unique(T)) != 2:
            raise ValueError("The Perceptron requires two classes.")

        X_train, X_test, T_train, T_test = split_dataset(X, T)
        if model_name == "Rosenblatt Perceptron":
            W, classes, epochs = fit_perceptron(X_train, T_train)
            predictions = np.array(
                [predict_perceptron(W, classes, row) for row in X_test]
            )
            model = (W, classes)
            metrics = {"Accuracy": accuracy_score(T_test, predictions), "Epochs": epochs}
        elif model_name == "Gaussian Naive Bayes":
            classes, means, vars, priors = fit_naive_bayes(X_train, T_train)
            predictions = np.array(
                [predict_naive_bayes(classes, means, vars, priors, row) for row in X_test]
            )
            model = (classes, means, vars, priors)
            metrics = {"Accuracy": accuracy_score(T_test, predictions)}
        else:
            predictions = np.array([k_nn(X_train, T_train, row, k) for row in X_test])
            model = (X_train, T_train, k)
            metrics = {"Accuracy": accuracy_score(T_test, predictions)}

    return (
        model,
        metrics,
        time.perf_counter() - start_time,
        feature_names,
        target_name,
    )


@st.cache_resource
def get_model_cache():
    #Keep fitted models available across Streamlit reruns and user sessions
    return {}


def train_model(model_name, dataset_id, k):
    #Return the actual cache status instead of guessing from elapsed time
    cache_key = get_cache_key(model_name, dataset_id, k)
    model_cache = get_model_cache()
    if cache_key in model_cache:
        return model_cache[cache_key], True

    result = fit_model(model_name, dataset_id, k)
    model_cache[cache_key] = result
    return result, False


def predict_with_model(model_name, model, user_features):
    #Predict the user's feature vector with the loaded model
    if model_name == "Rosenblatt Perceptron":
        return predict_perceptron(model[0], model[1], user_features)
    if model_name == "Gaussian Naive Bayes":
        return predict_naive_bayes(*model, user_features)
    if model_name == "K-Nearest Neighbors":
        return k_nn(model[0], model[1], user_features, model[2])

    theta, feature_mean, feature_std = model
    user_features = (user_features - feature_mean) / feature_std
    return float(np.append(user_features, 1.0) @ theta.ravel())


def get_cache_key(model_name, dataset_id, k):
    #Identify one trained model configuration
    return model_name, dataset_id, k


def openml_dataset_url(dataset_id):
    return f"https://www.openml.org/d/{int(dataset_id)}"


def style_comparison_table(dataframe):
    #Give the comparison table the same visual language as the application
    def status_style(value):
        if value == "CACHE HIT":
            return "color: #71d99e; font-weight: bold;"
        if value == "UNAVAILABLE":
            return "color: #ff8a80; font-weight: bold;"
        return "color: #ffcc66; font-weight: bold;"

    def metric_style(value):
        return "color: #f6f1d1; font-weight: bold;"

    styles = dataframe.style.set_properties(
        **{
            "background-color": "#14212b",
            "color": "#d8d2b5",
            "border-color": "#456273",
            "font-family": "VT323, monospace",
            "font-size": "18px",
        }
    ).set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#7b2d26"),
                    ("color", "#ffcc66"),
                    ("font-family", "Press Start 2P, monospace"),
                    ("font-size", "12px"),
                    ("letter-spacing", "1px"),
                    ("border", "2px solid #ffcc66"),
                    ("padding", "12px"),
                    ("text-align", "left"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #456273"),
                    ("padding", "10px 12px"),
                ],
            },
            {
                "selector": "tbody tr:hover td",
                "props": [
                    ("background-color", "#203544"),
                    ("color", "#ffcc66"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border", "2px solid #71d99e"),
                    ("box-shadow", "6px 6px #7b2d26"),
                ],
            },
        ]
    )
    if "Status" in dataframe:
        styles = styles.map(status_style, subset=["Status"])
    for column in dataframe.columns:
        if column not in {"Model", "Status", "Details"}:
            styles = styles.map(metric_style, subset=[column])
    return styles


METRIC_HELP = {
    "Accuracy": (
        "Share of correct predictions on the test set. "
        "For example, 0.90 means 90% of predictions were correct."
    ),
    "MSE": (
        "Mean Squared Error: average squared difference between the "
        "predicted and real target values. Lower is better."
    ),
    "R2": (
        "R² score: how much of the target variation is explained by the "
        "model. A score closer to 1 is generally better."
    ),
    "Epochs": (
        "Number of complete passes through the training data used by the "
        "gradient-descent model."
    ),
    "Training time": (
        "Time spent fitting the model for its initial training run."
    ),
    "Load time": (
        "Time needed to return the model for this request. A very short "
        "time usually means it was loaded from the cache."
    ),
}


try:
    catalog = load_openml_catalog()
except Exception:
    catalog = []

st.markdown(
    '<div class="hero"><div class="hero-kicker">ML LAB / OPENML WORKBENCH</div>'
    '<h1>MACHINE LEARNING LAB</h1>'
    '<div class="hero-copy">Train, inspect, reuse and compare models '
    "on real datasets.</div></div>",
    unsafe_allow_html=True,
)

mode = st.radio(
    "Workspace",
    ["▶ MODEL LAB", "⇄ COMPARE MODELS"],
    horizontal=True,
    key="workspace_mode",
)

if mode == "▶ MODEL LAB":
    selected_model = st.selectbox("Choose a model", model_names, key="lab_model")
    st.markdown(
        f'<div class="panel"><span class="small-label">SELECTED MODEL</span><br>'
        f"{model_descriptions[selected_model]}</div>",
        unsafe_allow_html=True,
    )
    datasets = get_compatible_datasets(catalog, selected_model)
    search = st.text_input("Search compatible OpenML datasets", key="lab_search")
    if search:
        datasets = [item for item in datasets if search.lower() in item["name"].lower()]

    if datasets:
        labels = {
            f'{item["name"]} [{item["features"]} features, {item["instances"]} rows]': item
            for item in datasets
        }
        label = st.selectbox(
            f"Choose a dataset ({len(datasets)} available)",
            list(labels),
            key="lab_dataset",
        )
        dataset = labels[label]
        st.markdown(
            f'[VIEW DATASET DETAILS ON OPENML ↗]({openml_dataset_url(dataset["did"])})'
        )
        k = (
            st.number_input("K neighbours", min_value=1, value=3, step=1)
            if selected_model == "K-Nearest Neighbors"
            else 3
        )
        model_key = get_cache_key(selected_model, dataset["did"], k)
        if st.session_state.get("lab_key") != model_key:
            st.session_state.pop("lab_result", None)
            st.session_state.pop("prediction_result", None)
            st.session_state["lab_key"] = model_key

        if st.button("▶ FIT / LOAD MODEL", use_container_width=True):
            try:
                with st.spinner("Loading dataset and fitting model..."):
                    load_start = time.perf_counter()
                    result, cached = train_model(
                        selected_model, dataset["did"], k
                    )
                    load_duration = time.perf_counter() - load_start
                    st.session_state["lab_result"] = (*result, load_duration, cached)
            except Exception:
                st.markdown(
                    '<div class="empty-state">This dataset cannot be used with '
                    "this model. Try another compatible dataset.</div>",
                    unsafe_allow_html=True,
                )

        if "lab_result" in st.session_state:
            (
                model,
                metrics,
                duration,
                feature_names,
                target_name,
                load_duration,
                cached,
            ) = st.session_state["lab_result"]
            st.markdown("### LOADED MODEL // METRICS")
            st.caption(f"Status: {'CACHE HIT' if cached else 'NEW TRAINING'}")
            columns = st.columns(len(metrics) + 2)
            for column, (name, value) in zip(columns, metrics.items()):
                column.metric(
                    name,
                    f"{value:.4f}" if isinstance(value, float) else value,
                    help=METRIC_HELP.get(name),
                )
            columns[-2].metric(
                "Training time",
                f"{duration:.4f} s",
                help=METRIC_HELP["Training time"],
            )
            columns[-1].metric(
                "Load time",
                f"{load_duration:.5f} s",
                help=METRIC_HELP["Load time"],
            )

            st.markdown("### YOUR FEATURE VECTOR")
            st.caption(f"Target: {target_name}")
            values = []
            feature_columns = st.columns(min(4, len(feature_names)))
            for index, feature_name in enumerate(feature_names):
                with feature_columns[index % len(feature_columns)]:
                    values.append(st.number_input(feature_name, value=0.0, key=f"lab_feature_{index}"))
            if st.button("PREDICT MY FEATURES", use_container_width=True):
                prediction = predict_with_model(selected_model, model, np.array(values))
                st.session_state["prediction_result"] = (target_name, prediction)
            if "prediction_result" in st.session_state:
                prediction_target, prediction = st.session_state["prediction_result"]
                if isinstance(prediction, (float, np.floating)):
                    prediction_text = f"{prediction:.4f}"
                else:
                    prediction_text = str(prediction)
                st.markdown(
                    f'<div class="prediction-result">'
                    f'<div class="prediction-label">PREDICTION READY // '
                    f'{escape(str(prediction_target))}</div>'
                    f'<div class="prediction-value">{escape(prediction_text)}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.markdown('<div class="empty-state">No compatible dataset found.</div>', unsafe_allow_html=True)

else:
    st.markdown("### COMPARE MODELS ON THE SAME DATASET")
    st.markdown(
        '<div class="panel"><span class="small-label">WORKFLOW</span><br>'
        "Choose the models first, then choose a dataset shared by all of them.</div>",
        unsafe_allow_html=True,
    )
    models_to_compare = st.multiselect(
        "Models to compare",
        model_names,
        default=["Gaussian Naive Bayes", "K-Nearest Neighbors"],
        key="compare_models",
    )
    if len(models_to_compare) >= 2:
        compatible = [
            {(item["did"]): item for item in get_compatible_datasets(catalog, name)}
            for name in models_to_compare
        ]
        common_ids = set(compatible[0])
        for items in compatible[1:]:
            common_ids &= set(items)
        datasets = sorted(
            [compatible[0][dataset_id] for dataset_id in common_ids],
            key=lambda item: item["name"],
        )
        search = st.text_input("Search common OpenML datasets", key="compare_search")
        if search:
            datasets = [item for item in datasets if search.lower() in item["name"].lower()]

        if datasets:
            labels = {
                f'{item["name"]} [{item["features"]} features, {item["instances"]} rows]': item
                for item in datasets
            }
            label = st.selectbox("Choose a dataset", list(labels), key="compare_dataset")
            dataset = labels[label]
            st.markdown(
                f'[VIEW DATASET DETAILS ON OPENML ↗]({openml_dataset_url(dataset["did"])})'
            )
            if "K-Nearest Neighbors" in models_to_compare:
                k = st.number_input("K neighbours", min_value=1, value=3, step=1)
            else:
                k = 3
            if st.button("⇄ RUN COMPARISON", use_container_width=True):
                with st.spinner("Fitting selected models..."):
                    progress = st.progress(0)
                    rows = []
                    for index, name in enumerate(models_to_compare):
                        try:
                            load_start = time.perf_counter()
                            cache_key = get_cache_key(name, dataset["did"], k)
                            result, cached = train_model(
                                name, dataset["did"], k
                            )
                            _, metrics, duration, _, _ = result
                            load_duration = time.perf_counter() - load_start
                            rows.append(
                                {
                                    "Model": name,
                                    **metrics,
                                    "Status": "CACHE HIT" if cached else "NEW TRAINING",
                                    "Training time (s)": round(duration, 4),
                                    "Load time (s)": round(load_duration, 5),
                                    "Details": "Ready",
                                }
                            )
                        except Exception:
                            rows.append(
                                {
                                    "Model": name,
                                    "Status": "UNAVAILABLE",
                                    "Details": (
                                        "No usable numerical features or target "
                                        "were found in this dataset."
                                    ),
                                }
                            )
                        finally:
                            progress.progress((index + 1) / len(models_to_compare))
                    comparison_data = pd.DataFrame(rows)
                    st.markdown(
                        '<div class="comparison-heading">'
                        "COMPARISON RESULTS // SAME DATASET</div>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        style_comparison_table(comparison_data.fillna("—")),
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.markdown('<div class="empty-state">No common dataset found.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">Select at least two models.</div>', unsafe_allow_html=True)
