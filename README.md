# Machine Learning Lab
Interactive Streamlit application for exploring machine learning models on
real-world datasets from [OpenML](https://www.openml.org/).
The application allows users to train models, make predictions with custom
features, view metrics and compare several models on the same dataset.
**Live application:** [machine-learning-lab-wb.streamlit.app](https://machine-learning-lab-wb.streamlit.app/)
## Features
- Browse compatible OpenML datasets.
- Open dataset details directly on OpenML.
- Train and reuse models with caching.
- Enter custom feature values and generate predictions.
- View metrics, training time and loading time.
- Compare several models on a shared dataset.
- Retro terminal-inspired interface.
Models are cached temporarily in memory. Reusing the same model, dataset and
parameters avoids training the model again and makes subsequent loads faster.
## Implemented models
- Rosenblatt Perceptron
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Linear Regression
The implementations are located in `supervised_algorithm/`.
## Tech stack
- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learns
- Matplotlib
- OpenML
## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```
The application is available locally at `http://localhost:8501`.
An internet connection is required to retrieve datasets from OpenML.
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
## Future development
This project is designed to evolve over time. Future additions may include:
- more supervised learning models;
- unsupervised learning models;
- dimensionality reduction;
- additional visualizations and metrics;
- hyperparameter controls;
- automated tests;