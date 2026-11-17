# Spam Detection Model

A supervised machine-learning project that classifies emails as spam or legitimate email using logistic regression. The model is trained on precomputed word-frequency features from `spam_data.csv`, evaluated on a held-out test set, and summarized with standard binary-classification metrics and a confusion matrix.

## Project overview

The project demonstrates a complete baseline classification workflow:

1. Load and inspect the email dataset with pandas.
2. Separate the numeric word-frequency features from the target label.
3. Exclude the non-predictive email identifier.
4. Split the data into training and test sets.
5. Train a logistic regression classifier.
6. Evaluate the classifier using accuracy, precision, recall, F1 score, and a confusion matrix.

Logistic regression is a useful baseline for this dataset because it is efficient, interpretable, and well suited to binary classification with many numeric features.

## Dataset

The included `spam_data.csv` file contains:

| Property | Value |
| --- | ---: |
| Rows | 5,172 |
| Columns | 3,002 |
| Numeric word features | 3,000 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Legitimate emails (`Prediction = 0`) | 3,672 |
| Spam emails (`Prediction = 1`) | 1,500 |

Important columns:

- `Email No.` is a text identifier such as `Email 1`. It is excluded from training because it does not describe the content of an email and cannot be passed directly to logistic regression.
- The remaining 3,000 columns contain numeric word-occurrence features.
- `Prediction` is the binary target: `0` represents a legitimate email and `1` represents spam.

The dataset is moderately imbalanced, with legitimate emails representing roughly 71% of the records. For that reason, accuracy should be considered together with precision, recall, F1 score, and the confusion matrix.

## Repository structure

```text
Spam_Detection_Model/
|-- app.py                              # Interactive Streamlit showcase
|-- spam_detect.py                      # Model training and evaluation
|-- spam_detection_pipeline.joblib      # Saved trained classifier
|-- spam_data.csv                       # Word features and target labels
|-- requirements.txt                    # Python dependencies
|-- README.md                           # Project documentation
`-- LICENSE                             # MIT License
```

## Requirements

- Python 3.9 or later
- pandas
- scikit-learn
- seaborn
- matplotlib

Install the dependencies with:

```bash
python -m pip install pandas scikit-learn seaborn matplotlib
```

Using a virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Then install the dependencies using the earlier `pip` command.

## Running the project

Clone or download the repository, open a terminal in its root directory, and run:

```bash
python spam_detect.py
```

The CSV file must remain in the same directory as the script because the program loads it using the relative path `spam_data.csv`.

### Streamlit demo

Install the dependencies and launch the interactive UI from the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app accepts pasted email text, shows the model's spam probability, lets you adjust
the classification threshold, and explains the most influential recognized words. It
loads the saved `spam_detection_pipeline.joblib` model and uses the dataset header to
recreate the same 3,000 word-count features used during training.

During execution, the script:

- prints descriptive statistics for the numeric columns;
- prints the number of missing values in every column;
- reserves 20% of the data for testing;
- trains the model on the remaining 80%;
- prints accuracy, precision, recall, and F1 score; and
- opens a window containing the confusion-matrix heatmap.

The split uses `random_state=42`, making the train/test partition reproducible when the same dataset and library versions are used.

## Model workflow

### Feature and target selection

The identifier and target columns are removed from the feature matrix:

```python
x = data.drop(columns=["Email No.", "Prediction"])
y = data["Prediction"]
```

This is necessary because scikit-learn's logistic regression implementation expects numeric input. Keeping `Email No.` would cause an error such as:

```text
ValueError: could not convert string to float: 'Email 3165'
```

### Train/test split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```

The test set is not used during model fitting. It provides an estimate of how the trained model performs on unseen rows from the same dataset.

### Training

```python
model = LogisticRegression()
model.fit(x_train, y_train)
```

Each learned coefficient represents the relationship between a word feature and the model's estimated probability that an email is spam.

## Evaluation metrics

- **Accuracy** is the proportion of all test emails classified correctly.
- **Precision** is the proportion of emails predicted as spam that are actually spam. High precision means fewer legitimate emails are incorrectly blocked.
- **Recall** is the proportion of actual spam emails detected by the model. High recall means less spam is missed.
- **F1 score** is the harmonic mean of precision and recall and is useful when both types of error matter.
- **Confusion matrix** reports true negatives, false positives, false negatives, and true positives.

For spam filtering, precision and recall often matter more than accuracy alone. A false positive may hide a legitimate email, while a false negative allows spam into the inbox. The preferred balance depends on how the model will be used.

## Current limitations

- The training script retrains the model each time it runs instead of reusing the saved artifact.
- It operates on precomputed word features and does not accept raw email text directly.
- The evaluation uses one train/test split rather than cross-validation.
- The split is not currently stratified, so class proportions are not explicitly preserved in each subset.
- No feature scaling, regularization tuning, threshold tuning, or comparison with alternative classifiers is performed.
- Results describe performance on this dataset and do not guarantee equivalent performance on emails from other sources or time periods.

## Possible improvements

- Use `stratify=y` in `train_test_split` to preserve the class distribution.
- Add cross-validation for a more stable performance estimate.
- Tune logistic regression hyperparameters such as `C`, `solver`, and `max_iter`.
- Add a text-processing pipeline with `TfidfVectorizer` so the model can train on and classify raw email text.
- Save the fitted pipeline with `joblib` for reuse without retraining.
- Add ROC-AUC and precision-recall curves.
- Inspect influential features to better understand model decisions.
- Add automated tests and a dependency file such as `requirements.txt`.

## Troubleshooting

### `FileNotFoundError: spam_data.csv`

Run the script from the repository root and confirm that `spam_data.csv` is beside `spam_detect.py`.

### `ValueError: could not convert string to float`

Confirm that both `Email No.` and `Prediction` are excluded from `x`. All model features must be numeric.

### Logistic regression convergence warning

If scikit-learn reports that the model did not converge, increase the iteration limit, for example:

```python
model = LogisticRegression(max_iter=1000)
```

Feature scaling or a different solver may also help.

### Confusion matrix does not appear

The plot requires a graphical environment. In a headless environment, save it instead of displaying it:

```python
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
```

## License

This project is available under the MIT License. See `LICENSE` for the complete terms.
