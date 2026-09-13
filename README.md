# Disaster Tweet Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](requirements.txt)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](notebooks/fakenews_disaster_detection.ipynb)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Real%20or%20Not%3F-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/c/nlp-getting-started)

This project builds a reproducible natural-language-processing baseline for the [Real or Not? Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) task. The goal is to classify whether a tweet describes a real disaster (`target = 1`) or uses disaster-related language without reporting an actual event (`target = 0`).

## Approach

The notebook compares two compact, interpretable text-classification pipelines:

- TF-IDF features using unigrams and bigrams, with English stop words removed
- Multinomial Naive Bayes
- Logistic Regression

Models are compared on a stratified validation split using accuracy, balanced accuracy, class-specific precision and recall, macro F1, weighted F1, ROC-AUC, and average precision. The strongest validation model is then retrained on the complete labeled dataset and used to create `submission.csv`.

## Repository Layout

```text
data/
	train.csv                         Labeled training data
	test.csv                          Unlabeled test data
notebooks/
	fakenews_disaster_detection.ipynb End-to-end analysis
requirements.txt                    Python dependencies
```

## Reproduce the Analysis

Use Python 3.9 or newer:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

Open `notebooks/fakenews_disaster_detection.ipynb` in Jupyter or VS Code and run all cells from top to bottom. The notebook expects the data files in `data/` and writes the generated predictions to `submission.csv` at the project root.

## Data

The included CSV files are from Kaggle's introductory NLP competition. The dataset contains tweet text and optional metadata fields such as keyword and location. This baseline intentionally uses the tweet text only, making the experiment easy to reproduce and providing a clear reference point for future feature engineering.

## Next Steps

Potential extensions include cleaning URLs and user mentions, testing character-level features, incorporating `keyword` and `location`, tuning decision thresholds, and evaluating cross-validation stability before submitting to Kaggle.

## License

This project is licensed under the [MIT License](LICENSE).
