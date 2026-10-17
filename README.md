# Machine Learning Lab

Machine Learning Lab is an interactive Streamlit application for exploring
machine learning algorithms on real-world datasets from
[OpenML](https://www.openml.org/).

The project combines from-scratch algorithm implementations with a web
interface where users can train models, inspect their results, make
predictions and compare different approaches on the same dataset.

## Features

### Model Lab

The Model Lab allows users to:

- choose a machine learning model;
- search through compatible OpenML datasets;
- open the selected dataset directly on OpenML;
- train the model and view its metrics;
- see training and cache loading times;
- enter custom feature values using the dataset's real feature names;
- generate a prediction from the trained model;
- reuse trained models through Streamlit's cache.

### Model comparison

The comparison workspace allows users to:

- select several models;
- find datasets compatible with all selected models;
- train the models on the same dataset;
- use the same train/test split for a fair comparison;
- compare metrics, training time, loading time and cache status;
- identify models that cannot process a dataset.

The `K neighbours` setting is displayed only when K-Nearest Neighbors is
selected.

## Available models

The current version includes:

- **Rosenblatt Perceptron** — binary linear classification;
- **Gaussian Naive Bayes** — probabilistic classification based on Gaussian
  feature distributions;
- **K-Nearest Neighbors** — classification based on the closest training
  examples;
- **Linear Regression** — regression trained with batch gradient descent.

The algorithms are implemented in the `supervised_algorithm/` directory. The
project intentionally keeps the implementations simple and readable instead
of relying on a complex object-oriented architecture.

## Metrics

Classification models use:

- **Accuracy** — the proportion of correct predictions on the test set;
- **Epochs** — the number of training passes when the algorithm uses epochs.

The regression model uses:

- **MSE** — the mean squared error between predictions and real values;
- **R²** — the proportion of target variation explained by the model;
- **Epochs** — the number of gradient descent iterations.

Each metric includes an explanation directly in the interface.

## OpenML datasets

The application loads the active OpenML catalog and filters datasets according
to the selected model and task.

Dataset metadata is loaded first. The actual dataset is downloaded only when it
is required for training or comparison. Runtime validation is also performed
because OpenML metadata does not always guarantee that a dataset contains usable
features and a valid target.

To keep the application responsive, datasets larger than 5,000 rows are
sampled before training using a reproducible random seed.

Every selected dataset includes a direct link to its OpenML page, where users
can find the complete description, features, target, data quality and other
metadata.

## Caching

Trained models are stored using a Streamlit resource cache. The cache key
depends on:

```text
model + dataset + hyperparameters
```

For K-Nearest Neighbors, the value of `k` is included in the key.

The interface displays two statuses:

- `NEW TRAINING` — the model configuration was not found in the cache;
- `CACHE HIT` — an existing trained model was reused.

Training time and cache loading time are displayed separately. The cache is
stored in server memory, so it can be cleared when the application restarts or
when Streamlit's cache is manually cleared.

## User interface

The application uses a retro terminal-inspired design with:

- dark colors;
- pixel-style headings;
- terminal typography;
- animated prediction results;
- styled comparison tables;
- visual training and cache statuses.

## Technology stack

- **Python** — application and algorithm implementations;
- **Streamlit** — interactive web interface;
- **NumPy** — numerical computations;
- **Pandas** — data manipulation and tables;
- **Scikit-learn** — OpenML access, train/test splitting and evaluation
  metrics;
- **Matplotlib** — visualisation support;
- **OpenML** — dataset catalog and data source.

Scikit-learn is used for data access, splitting and metrics. The main
algorithm logic used by the application is implemented in the project's own
Python files.

## Project structure

```text
.
├── app.py
├── supervised_algorithm/
│   ├── k_nn.py
│   ├── linear_regression.py
│   ├── naives_bayes.py
│   ├── rosenblatt_perecptron.py
│   └── visualize.py
├── requirements.txt
└── README.md
```

Each algorithm script contains the implementation used by the application and
can also be used independently from the command line where supported.

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

The application is usually available at:

```text
http://localhost:8501
```

An internet connection is required to retrieve the OpenML catalog and
download datasets.

## Future development

Machine Learning Lab is designed to evolve over time. Planned improvements
include:

- additional supervised learning algorithms;
- unsupervised learning models such as K-Means and hierarchical clustering;
- dimensionality reduction methods such as PCA;
- improved dataset validation;
- confusion matrices and more detailed classification metrics;
- decision boundary and prediction visualisations;
- hyperparameter controls;
- experiment history and result export;
- automated tests for the algorithms;
- persistent model storage;
- online deployment.

New models will be added progressively as their implementations, metrics and
compatibility rules are developed and tested.
