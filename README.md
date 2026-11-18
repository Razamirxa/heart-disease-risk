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
| Part 2 | `notebooks/02_modeling_pipeline.ipynb` | split, models, tuning, evaluation, comparison, final pipeline, app | in progress |

**Part 1** (notebook `01_data_cleaning_eda.ipynb` and this README) was prepared by **Raza Ul Mustafa (26L-8030)**.

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
├── app/                     # Streamlit app (Part 2)
├── models/                  # saved final pipeline (Part 2)
├── reports/figures/         # figures created by the notebooks
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

To run a notebook from the command line instead:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_data_cleaning_eda.ipynb
```

## 7. Main Part 1 findings

- 918 patients, 55% disease; disease rate differs strongly by hospital (Hungary 36%, Cleveland 46%, VA 74%, Switzerland 93%).
- 172 cholesterol values were 0 (all Swiss patients, 49 VA patients): hidden missing values, recoded as missing. Missing cholesterol is linked to an 81% disease rate because of the Swiss hospital, which is a shortcut risk.
- Strongest features: ChestPainType, ExerciseAngina, MaxHR, Oldpeak, ST_Slope and Sex.
- Imputation, encoding and scaling are fitted on the training set only (80/20 split stratified by hospital and diagnosis, 734 / 184 patients).
