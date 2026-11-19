# Heart Disease Risk Prediction from Clinical Measurements

**DS5002 Data Science Tools and Techniques — Term Project (Fall 2026), FAST-NUCES Lahore**
Instructor: Dr. Asma Ahmad · TA: Burhan Hadi Butt

**Group 15**

| Member | Student ID |
|---|---|
| Raza Ul Mustafa | 26L-8030 |
| Ahmad Ali | 26L-8015 |
| Gohar Maqsood | 26L-8033 |

---

## 1. Project overview

Heart disease is the leading cause of death worldwide. This project builds an end-to-end machine learning pipeline that predicts whether a patient has heart disease from **11 clinical and exercise-test measurements** (age, sex, chest pain type, blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise angina, ST depression and ST slope).

**Research questions**

| | Question |
|---|---|
| RQ1 | Can a tuned classifier reach **recall ≥ 0.90** on the disease class with precision ≥ 0.80 on unseen patients? |
| RQ2 | How much does performance drop when the model is tested on a **hospital it has not seen** (leave-one-hospital-out)? |
| RQ3 | Is "Cholesterol = 0" a **shortcut** for the Swiss hospital rather than a medical signal? |
| RQ4 | Which features drive the predictions (SHAP), and are they the same in every hospital? |

**Project parts**

| Part | Notebook | Content | Status |
|---|---|---|---|
| Part 1 | `notebooks/01_data_cleaning_eda.ipynb` | loading, cleaning, preprocessing, feature engineering and selection, EDA, insights | done |
| Part 2 | `notebooks/02_modeling_pipeline.ipynb` | train/validation/test design, 8 models, tuning, optimization, evaluation, comparison, robustness, final pipeline | done |
| Demo | `app/streamlit_app.py` | web app that uses the saved final pipeline | done |

**Part 1** (notebook `01_data_cleaning_eda.ipynb`) was prepared by **Raza Ul Mustafa (26L-8030)**, **Part 2** (notebook `02_modeling_pipeline.ipynb`) by **Gohar Maqsood (26L-8033)**. See `CONTRIBUTORS.md` for the work split.

## 2. Dataset

- **Name:** Heart Disease (UCI Machine Learning Repository)
- **Source:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Creators:** Andras Janosi, William Steinbrunn, Matthias Pfisterer, Robert Detrano (1988)
- **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0)
- **Files used** (in `data/raw/`, unchanged): `processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data`
- **Size:** 920 records from 4 hospitals → **918 unique patients** after removing 2 duplicates (508 with disease, 410 normal)
- **Target:** `HeartDisease` = 1 if the UCI diagnosis `num` > 0, else 0
- **Related file:** the Kaggle "Heart Failure Prediction" dataset (fedesoriano, 2021) contains the same 918 patients but without the hospital label, which is why we use the original UCI files.

Citation:
> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

## 3. Repository structure

```
heart-disease-risk/
├── data/
│   ├── raw/                 # 4 original UCI files (not modified)
│   └── processed/           # heart_clean.csv, split_indices.csv (created by notebook 01)
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb      # Part 1
│   └── 02_modeling_pipeline.ipynb      # Part 2
├── app/
│   └── streamlit_app.py     # demo app (uses models/final_pipeline.joblib)
├── models/
│   └── final_pipeline.joblib            # preprocessing + model + threshold (created by notebook 02)
├── reports/
│   ├── figures/             # fig_* from notebook 01, p2_* from notebook 02
│   └── p2_*.csv             # Part 2 result tables (CV, tuning, test, leave-one-hospital-out)
├── CONTRIBUTORS.md
├── requirements.txt
└── README.md
```

## 4. Required libraries

Python **3.11 or newer** (tested with Python 3.13). All versions are pinned in `requirements.txt`:

| Library | Used for |
|---|---|
| pandas, numpy | data loading and cleaning |
| scipy | statistical tests (chi-square, point-biserial correlation) |
| scikit-learn | imputation, encoding, scaling, train/test split, mutual information, models |
| matplotlib, seaborn | visualizations |
| xgboost, shap | models and explanations (Part 2) |
| streamlit, joblib | demo app and saved model (Part 2) |
| jupyter, ipykernel | running the notebooks |

## 5. Setup

**Option A — with uv (fast)**

```bash
git clone https://github.com/Razamirxa/heart-disease-risk.git
cd heart-disease-risk
uv venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
uv pip install -r requirements.txt
```

**Option B — with pip**

```bash
git clone https://github.com/Razamirxa/heart-disease-risk.git
cd heart-disease-risk
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

No download is needed: the raw data files are already in `data/raw/`.

## 6. How to run the notebooks

1. Activate the environment (step 5).
2. Start Jupyter from the project folder:
   ```bash
   jupyter notebook
   ```
   (or open the notebook in VS Code and select the `.venv` interpreter)
3. Open `notebooks/01_data_cleaning_eda.ipynb` and choose **Kernel → Restart & Run All**.
   - The notebook reads `data/raw/`, and writes `data/processed/heart_clean.csv`, `data/processed/split_indices.csv` and the figures in `reports/figures/`.
   - It runs in under a minute on a laptop. All random steps use `random_state = 42`, so the results are the same on every run.
4. Run `notebooks/02_modeling_pipeline.ipynb` **after** notebook 01, because it uses the saved split.
   - It takes about 1.5 minutes (most of the time is the Random Forest grid search).
   - It writes `models/final_pipeline.joblib`, the result tables `reports/p2_*.csv` and the figures `reports/figures/p2_*.png`.

To run the notebooks from the command line instead:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_data_cleaning_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_modeling_pipeline.ipynb
```

## 7. How to run the demo app

With the environment active, from the project folder:

```bash
streamlit run app/streamlit_app.py
```

The app opens in the browser (`http://localhost:8501`). Enter a patient's 11 measurements (blood pressure, cholesterol, fasting sugar and ST slope can be marked "not measured"). The app shows the predicted probability of heart disease, the decision at the threshold 0.35 ("refer for further tests" or "low risk"), and which inputs pushed the prediction up or down. It is a student project for learning, not a medical tool.

## 8. Main Part 1 findings

- 918 patients, 55% disease; disease rate differs strongly by hospital (Hungary 36%, Cleveland 46%, VA 74%, Switzerland 93%).
- 172 cholesterol values were 0 (all Swiss patients, 49 VA patients): hidden missing values, recoded as missing. Missing cholesterol is linked to an 81% disease rate because of the Swiss hospital, which is a shortcut risk.
- Strongest features: ChestPainType, ExerciseAngina, MaxHR, Oldpeak, ST_Slope and Sex.
- Imputation, encoding and scaling are fitted on the training set only (80/20 split stratified by hospital and diagnosis, 734 / 184 patients).

## 9. Part 2 results

**Method:** 5-fold cross-validation on the 734 training patients (stratified by hospital and diagnosis) for model comparison, tuning and the choice of threshold; the 184 test patients are used once at the end. All preprocessing is inside one scikit-learn `Pipeline`, so nothing leaks from validation or test data.

**Model comparison (mean CV ROC-AUC)**

| Model | Default settings | After tuning |
|---|---|---|
| **Logistic Regression** | **0.886 ± 0.005** | **0.887** (L1, C = 1) |
| SVM (RBF) | 0.871 | 0.884 |
| Random Forest | 0.873 | 0.882 |
| XGBoost | 0.854 | 0.879 |
| Naive Bayes | 0.877 | — |
| kNN | 0.839 | — |
| Decision Tree | 0.714 | — |
| Majority-class baseline | 0.500 | — |

**Final model:** Logistic Regression (L1, C = 1) with decision threshold **0.35**, chosen on the training folds so that recall ≥ 0.90. It had the best and most stable CV score and is the simplest to explain.

**Test set (184 patients, used once)**

| Recall | Precision | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|
| **0.931** (95% CI 0.878–0.980) | 0.779 (0.703–0.849) | 0.848 | 0.815 | **0.890** (0.839–0.935) |

Confusion matrix: 95 of 102 sick patients found (7 missed), 27 false alarms among 82 healthy patients.

**Answers to the research questions**

| RQ | Result |
|---|---|
| RQ1 recall ≥ 0.90 with precision ≥ 0.80 | Partly: recall 0.931 met, precision 0.779 just below 0.80 |
| RQ2 unseen hospital (train on 3, test on the 4th) | ROC-AUC Cleveland 0.857, Hungary 0.882 (small drop from 0.887); Switzerland 0.726, VA 0.737 (large drop at very different hospitals) |
| RQ3 missing cholesterol as a shortcut | Yes: it raises mixed-hospital CV AUC (0.878 → 0.887) but makes no difference at a new hospital (−0.004 to +0.003) |
| RQ4 main drivers (SHAP) | Asymptomatic chest pain, sex, exercise angina, Oldpeak and MaxHR in every hospital; missing cholesterol mainly in Switzerland |

**Reproducibility:** all random steps use `random_state = 42`; running the two notebooks in order reproduces the numbers above (the "seconds" column of the tuning table changes from run to run).
