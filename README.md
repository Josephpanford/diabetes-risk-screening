# Can Routine Measurements Flag Diabetes Risk Before a Lab Test?

A machine learning capstone project that tests whether **routine, non-laboratory measurements** (age, BMI, blood pressure, etc.) can screen for diabetes risk, and how much laboratory values (glucose, insulin) add on top.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for clinical decisions.



## Research question

Can routine measurements flag diabetes risk **before** a lab test is done?

## Dataset

- **Pima Indians Diabetes Database** (Kaggle): https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- 768 patients, 8 predictors + `Outcome` (1 = diabetic), about 35% positive
- Women aged 21+ of Pima heritage only
- The data file is **not** included in this repo. Download it from Kaggle and place it where the notebook expects it (see *How to run*).

| Model | Features |
|---|---|
| **Model A (routine)** | Pregnancies, BloodPressure, SkinThickness, BMI, DiabetesPedigreeFunction, Age |
| **Model B (routine + lab, benchmark)** | Model A features + Glucose, Insulin |

Model A answers the research question. Model B only shows how much the lab values add.

## Method

| Step | Detail |
|---|---|
| Cleaning | Impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI treated as missing |
| Split | 80/20 stratified train/test (614 / 154 patients); test set untouched until final evaluation |
| Preprocessing | Median imputation (and scaling for Logistic Regression) inside sklearn `Pipeline`s to prevent data leakage |
| Models compared | Logistic Regression, Random Forest, XGBoost |
| Validation | 5-fold x 5-repeat stratified cross-validation on the training set |
| Model selection | Highest mean ROC-AUC (Model A features) |
| Imbalance | Class weights / `scale_pos_weight` |
| Threshold | Chosen on out-of-fold training predictions to reach 85% recall (screening goal), then applied once to the test set |
| Metrics | Recall, specificity, precision, F1, ROC-AUC, PR-AUC, Brier score, confusion matrix, calibration curve |
| Interpretation | Permutation importance for Model A |

## Results

### Model comparison (cross-validation, Model A features, mean ± SD)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.754 ± 0.028** | 0.605 ± 0.050 | 0.662 ± 0.068 | 0.526 ± 0.041 | 0.583 ± 0.034 |
| Random Forest | 0.742 ± 0.043 | 0.571 ± 0.056 | 0.629 ± 0.077 | 0.544 ± 0.043 | 0.580 ± 0.040 |
| XGBoost | 0.732 ± 0.047 | 0.557 ± 0.063 | 0.658 ± 0.075 | 0.529 ± 0.047 | 0.584 ± 0.042 |

The three models are within about one standard deviation of each other, so Logistic Regression was chosen as the best-performing and most interpretable option. (CV recall uses the default 0.5 threshold.)

### Held-out test set (Logistic Regression, tuned threshold)

| Metric | Model A (routine) | Model B (routine + lab) |
|---|---|---|
| Threshold | 0.397 | 0.359 |
| Recall (sensitivity) | 0.796 | 0.870 |
| Specificity | 0.540 | 0.650 |
| Precision | 0.483 | 0.573 |
| F1 | 0.601 | 0.691 |
| ROC-AUC | 0.729 | 0.813 |
| PR-AUC | 0.554 | 0.673 |
| Brier score | 0.211 | 0.181 |
| TP / FP / FN / TN | 43 / 46 / 11 / 54 | 47 / 35 / 7 / 65 |

### Feature importance (Model A, drop in ROC-AUC when shuffled)

BMI (0.066) > Age (0.031) > Pregnancies (0.027) > DiabetesPedigreeFunction (0.014) > SkinThickness (about 0) ≈ BloodPressure (about 0)

## Key findings

- Routine measurements carry real signal (test ROC-AUC about 0.73), so they can serve as a **pre-screen**.
- Lab values add a clear improvement (ROC-AUC +0.08), so routine screening does not replace testing.
- At the screening threshold, Model A flags many healthy people (46 of 100 false alarms), so it is a triage tool, not a diagnostic one.
- The threshold tuned to 85% recall on training data gave about 80% recall on the test set, which shows the uncertainty of a small test set.
- BMI and Age are the most influential routine measurements.

## Limitations

- Small dataset (768 patients) and small test set (54 diabetic cases), so test metrics are noisy.
- Single population (Pima women, 21+); results may not generalize to other groups or to men.
- Only six routine features, and some have many missing values (SkinThickness about 30%).
- No external validation dataset.
- Not clinically validated.

## How to run

**On Kaggle (recommended):**
1. Create a notebook and click **Add Input**, then attach the Pima Indians Diabetes dataset.
2. Upload `diabetes-prediction.ipynb` and run all cells. Cell 2 finds the CSV automatically.

**Locally:**
```bash
pip install -r requirements.txt
```
Download `diabetes.csv` from Kaggle and change the folder searched in Cell 2 (`/kaggle/input`) to the folder containing the file, then run the notebook in Jupyter.

## Repository contents

| File | Description |
|---|---|
| `diabetes-prediction.ipynb` | Full analysis notebook |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from version control |

## Acknowledgements

Dataset: National Institute of Diabetes and Digestive and Kidney Diseases (via UCI / Kaggle).
