# =====================================================
# HEART DISEASE PREDICTION - STREAMLIT UI
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #C0392B;
        text-align: center;
        font-weight: bold;
        padding: 1rem 0;
    }
    .result-box-yes {
        background: #FADBD8;
        border: 2px solid #E74C3C;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #C0392B;
    }
    .result-box-no {
        background: #D5F5E3;
        border: 2px solid #27AE60;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E8449;
    }
    .info-box {
        background: #EBF5FB;
        border-left: 4px solid #2E86C1;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('heart_disease_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── HEADER ────────────────────────────────────────────
st.markdown(
    '<p class="main-header">❤️ Heart Disease Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
    <b>About:</b> This AI system predicts the likelihood of heart disease
    based on your health parameters. Fill in the details below and click
    <b>Predict</b> to get your result instantly.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── INPUT FORM ────────────────────────────────────────
st.subheader("📋 Enter Your Health Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Info**")
    age = st.slider("Age", 20, 80, 45,
                    help="Your current age in years")
    sex = st.radio("Gender",
                   options=[1, 0],
                   format_func=lambda x: "Male" if x == 1 else "Female")
    cp  = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-Anginal Pain",
            3: "Asymptomatic"
        }[x],
        help="Type of chest pain experienced"
    )

with col2:
    st.markdown("**Blood Parameters**")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)",
                         80, 200, 120,
                         help="Resting blood pressure")
    chol     = st.slider("Cholesterol (mg/dl)",
                         100, 600, 200,
                         help="Serum cholesterol level")
    fbs      = st.radio(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    thalach  = st.slider("Max Heart Rate Achieved",
                         60, 220, 150,
                         help="Maximum heart rate achieved")

with col3:
    st.markdown("**ECG & Other**")
    restecg = st.selectbox(
        "Resting ECG Result",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Abnormality",
            2: "Left Ventricular Hypertrophy"
        }[x]
    )
    exang   = st.radio(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    oldpeak = st.slider("ST Depression (Oldpeak)",
                        0.0, 6.0, 1.0, step=0.1)
    slope   = st.selectbox(
        "ST Slope",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }[x]
    )
    ca   = st.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thal",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Normal",
            1: "Fixed Defect",
            2: "Reversible Defect",
            3: "Unknown"
        }[x]
    )

st.markdown("---")

# ── PREDICT BUTTON ────────────────────────────────────
col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
with col_b2:
    predict_btn = st.button(
        "🔍 Predict Now",
        use_container_width=True,
        type="primary"
    )

# ── RESULT ────────────────────────────────────────────
if predict_btn:
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak, slope, ca, thal
    ]])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box-yes">
                ⚠️ Heart Disease Detected<br>
                <small>Probability: {probability[1]*100:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
            st.error("Please consult a doctor immediately.")
        else:
            st.markdown(f"""
            <div class="result-box-no">
                ✅ No Heart Disease Detected<br>
                <small>Probability: {probability[0]*100:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
            st.success("Your heart appears healthy! Keep maintaining a healthy lifestyle.")

    st.markdown("---")

    # Probability bars
    st.subheader("📈 Confidence Level")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric("No Disease Probability", f"{probability[0]*100:.1f}%")
        st.progress(float(probability[0]))
    with col_p2:
        st.metric("Disease Probability", f"{probability[1]*100:.1f}%")
        st.progress(float(probability[1]))

    # Input summary table
    st.markdown("---")
    st.subheader("📋 Your Input Summary")
    summary = pd.DataFrame({
        'Parameter': [
            'Age', 'Gender', 'Chest Pain', 'Blood Pressure',
            'Cholesterol', 'Blood Sugar', 'ECG', 'Max Heart Rate',
            'Exercise Angina', 'ST Depression', 'ST Slope',
            'Major Vessels', 'Thal'
        ],
        'Value': [
            age,
            'Male' if sex == 1 else 'Female',
            {0:'Typical',1:'Atypical',2:'Non-Anginal',3:'Asymptomatic'}[cp],
            f"{trestbps} mm Hg",
            f"{chol} mg/dl",
            'Yes' if fbs == 1 else 'No',
            {0:'Normal',1:'ST-T Abnormal',2:'LV Hypertrophy'}[restecg],
            thalach,
            'Yes' if exang == 1 else 'No',
            oldpeak,
            {0:'Upsloping',1:'Flat',2:'Downsloping'}[slope],
            ca,
            {0:'Normal',1:'Fixed',2:'Reversible',3:'Unknown'}[thal]
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.85rem;'>
    ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only.
    Always consult a qualified medical professional for health advice.<br>
    Built with ❤️ using Machine Learning & Streamlit
</div>
""", unsafe_allow_html=True)