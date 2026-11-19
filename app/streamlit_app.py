"""Heart Disease Risk Screening — demo app (DS5002 Term Project, Group 15).

Run from the repository folder:
    streamlit run app/streamlit_app.py

The app loads the final pipeline saved by notebooks/02_modeling_pipeline.ipynb
(models/final_pipeline.joblib): preprocessing + Logistic Regression + the decision threshold.
It is a student project for learning, not a medical device.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "final_pipeline.joblib"

# Readable names for the model's columns (used in the explanation chart)
NICE_NAMES = {
    "num__Age": "Age", "num__RestingBP": "Resting blood pressure", "num__Cholesterol": "Cholesterol",
    "num__MaxHR": "Maximum heart rate", "num__Oldpeak": "ST depression (Oldpeak)",
    "bin__Sex_M": "Sex: male", "bin__ExerciseAngina_Y": "Exercise-induced angina",
    "bin__FastingBS_1.0": "Fasting blood sugar > 120",
    "nom__ChestPainType_ASY": "Chest pain: asymptomatic", "nom__ChestPainType_ATA": "Chest pain: atypical angina",
    "nom__ChestPainType_NAP": "Chest pain: non-anginal", "nom__ChestPainType_TA": "Chest pain: typical angina",
    "nom__RestingECG_LVH": "Resting ECG: LVH", "nom__RestingECG_Normal": "Resting ECG: normal",
    "nom__RestingECG_ST": "Resting ECG: ST-T abnormality", "ord__ST_Slope": "ST slope (Up < Flat < Down)",
    "flag__Cholesterol_missing": "Cholesterol not measured", "flag__ST_Slope_missing": "ST slope not recorded",
}

# Two ready-made patients for the demo
EXAMPLES = {
    "Higher-risk example": dict(age=58, sex="Male", cp="ASY", bp=140, bp_known=True, chol=260, chol_known=True,
                                fbs="No", ecg="Normal", maxhr=120, angina="Yes", oldpeak=1.8, slope="Flat"),
    "Lower-risk example": dict(age=42, sex="Female", cp="ATA", bp=120, bp_known=True, chol=210, chol_known=True,
                               fbs="No", ecg="Normal", maxhr=170, angina="No", oldpeak=0.0, slope="Up"),
}


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def set_example(name):
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value


def make_patient_row(v):
    # one-row DataFrame in exactly the format used during training; NaN = "not measured"
    row = {
        "Age": float(v["age"]),
        "RestingBP": float(v["bp"]) if v["bp_known"] else np.nan,
        "Cholesterol": float(v["chol"]) if v["chol_known"] else np.nan,
        "MaxHR": float(v["maxhr"]),
        "Oldpeak": float(v["oldpeak"]),
        "Sex": "M" if v["sex"] == "Male" else "F",
        "ExerciseAngina": "Y" if v["angina"] == "Yes" else "N",
        "FastingBS": {"Yes": 1.0, "No": 0.0}.get(v["fbs"], np.nan),
        "ChestPainType": v["cp"],
        "RestingECG": v["ecg"],
        "ST_Slope": v["slope"] if v["slope"] != "Not recorded" else np.nan,
        "Cholesterol_missing": 0 if v["chol_known"] else 1,
        "ST_Slope_missing": 1 if v["slope"] == "Not recorded" else 0,
    }
    df = pd.DataFrame([row])
    cat = ["Sex", "ExerciseAngina", "ChestPainType", "RestingECG", "ST_Slope"]
    df[cat] = df[cat].astype(object)
    return df


def explain(bundle, patient):
    # contribution of each column to the log-odds = coefficient x (scaled / encoded value)
    prep = bundle["pipeline"].named_steps["prep"]
    model = bundle["pipeline"].named_steps["model"]
    z = prep.transform(patient[bundle["features"]])[0]
    contrib = pd.Series(model.coef_[0] * z, index=prep.get_feature_names_out())
    contrib = contrib[contrib != 0].rename(index=NICE_NAMES)
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="Heart Disease Risk Screening", page_icon="🫀", layout="wide")
st.title("🫀 Heart Disease Risk Screening")
st.caption("DS5002 Term Project · Group 15 · Raza Ul Mustafa, Ahmad Ali, Gohar Maqsood · "
           "For learning only, not for medical decisions.")

bundle = load_bundle()
threshold = bundle["threshold"]

with st.sidebar:
    st.header("About the model")
    st.write(f"**{bundle['model_name']}** (L1, C = {bundle['settings']['C']}) inside one scikit-learn pipeline: "
             "imputation → scaling / encoding → model.")
    tm = bundle["test_metrics"]
    st.write(f"Decision threshold: **{threshold:.2f}** (chosen for recall ≥ 0.90)")
    st.write("**Test set (184 patients)**")
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{tm['recall']:.3f}")
    c2.metric("Precision", f"{tm['precision']:.3f}")
    c1.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    c2.metric("Accuracy", f"{tm['accuracy']:.3f}")
    st.info("Trained on 734 patients from 4 hospitals (UCI Heart Disease data). At a very different "
            "hospital the accuracy can be lower. Missing cholesterol acts partly as a hospital shortcut.")
    st.write("**Try an example:**")
    for name in EXAMPLES:
        st.button(name, on_click=set_example, args=(name,), width="stretch")

for key, value in EXAMPLES["Higher-risk example"].items():   # default values on first load
    st.session_state.setdefault(key, value)

st.subheader("Patient measurements")
col1, col2, col3 = st.columns(3)
with col1:
    st.number_input("Age (years)", 20, 100, step=1, key="age")
    st.radio("Sex", ["Male", "Female"], horizontal=True, key="sex")
    st.selectbox("Chest pain type", ["ASY", "ATA", "NAP", "TA"], key="cp",
                 format_func=lambda c: {"ASY": "ASY – asymptomatic", "ATA": "ATA – atypical angina",
                                        "NAP": "NAP – non-anginal pain", "TA": "TA – typical angina"}[c])
    st.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes", "Not known"], key="fbs")
with col2:
    st.checkbox("Resting blood pressure measured", key="bp_known")
    st.number_input("Resting blood pressure (mm Hg)", 80, 220, step=1, key="bp",
                    disabled=not st.session_state["bp_known"])
    st.checkbox("Cholesterol measured", key="chol_known")
    st.number_input("Serum cholesterol (mg/dl)", 80, 650, step=1, key="chol",
                    disabled=not st.session_state["chol_known"])
    st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], key="ecg",
                 format_func=lambda e: {"Normal": "Normal", "ST": "ST-T wave abnormality",
                                        "LVH": "Left ventricular hypertrophy"}[e])
with col3:
    st.number_input("Maximum heart rate in the exercise test", 60, 210, step=1, key="maxhr")
    st.radio("Chest pain during exercise (angina)", ["No", "Yes"], horizontal=True, key="angina")
    st.number_input("ST depression, Oldpeak (mm)", -3.0, 7.0, step=0.1, format="%.1f", key="oldpeak")
    st.selectbox("Slope of the peak exercise ST segment", ["Up", "Flat", "Down", "Not recorded"], key="slope")

values = {k: st.session_state[k] for k in EXAMPLES["Higher-risk example"]}
patient = make_patient_row(values)
risk = float(bundle["pipeline"].predict_proba(patient[bundle["features"]])[0, 1])

st.divider()
left, right = st.columns([1, 1.3])
with left:
    st.subheader("Result")
    st.metric("Predicted probability of heart disease", f"{risk:.1%}")
    st.progress(min(max(risk, 0.0), 1.0))
    if risk >= threshold:
        st.error(f"**Refer for further tests.** The probability is above the screening threshold ({threshold:.0%}).")
    else:
        st.success(f"**Low risk.** The probability is below the screening threshold ({threshold:.0%}).")
    st.caption("The threshold is low on purpose: missing a sick patient is worse than an extra test.")
with right:
    st.subheader("What pushed this prediction")
    contrib = explain(bundle, patient)
    chart = pd.DataFrame({"effect on risk (log-odds)": contrib.iloc[::-1]})
    st.bar_chart(chart, horizontal=True)
    st.caption("Positive bars raise the risk, negative bars lower it (coefficient × scaled value).")

with st.expander("Input as the model sees it"):
    st.dataframe(patient, hide_index=True)
