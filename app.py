import streamlit as st
import joblib
import numpy as np

# Set page config
st.set_page_config(
    page_title="Heart Health Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .predict-box {
        padding: 2rem;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    h1 {
        color: #1e1e1e;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model and scaler
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

if model and scaler:
    st.title("❤️ Heart Disease Prediction System")
    st.markdown("<p style='text-align: center; color: #666;'>Advanced machine learning tool for early heart disease detection</p>", unsafe_allow_html=True)
    st.divider()

    # Input section organized in columns
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Patient Info")
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4], help="1: Typical Angina, 2: Atypical Angina, 3: Non-anginal Pain, 4: Asymptomatic")

        with col2:
            st.subheader("🩺 Vitals")
            bp = st.slider("Resting BP (mm Hg)", 80, 200, 120)
            cholesterol = st.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "True" if x == 1 else "False")
            max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)

        with col3:
            st.subheader("🧪 Tests")
            ekg = st.selectbox("Resting EKG Results", [0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
            exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            st_depression = st.slider("ST Depression (Relative to rest)", 0.0, 6.0, 1.0)
            slope_st = st.selectbox("Slope of Peak Exercise ST", [1, 2, 3])
            vessels_fluro = st.selectbox("Major Vessels Colored by Fluroscopy", [0, 1, 2, 3])
            thallium = st.selectbox("Thallium Scan Results", [3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}[x])

    st.divider()

    # Prediction Button
    if st.button("Analyze Health Risk"):
        # Prepare input
        input_data = np.array([[age, sex, chest_pain, bp, cholesterol, fbs, ekg, max_hr, exercise_angina, st_depression, slope_st, vessels_fluro, thallium]])
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0]
        risk_prob = probability[1]

        # Results Display
        st.subheader("Results")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prediction[0] == 1:
                st.error("### HIGH RISK")
                st.write("The model suggests a high risk of heart disease.")
            else:
                st.success("### LOW RISK")
                st.write("The model suggests a low risk of heart disease.")

        with res_col2:
            st.write(f"Confidence Level: **{max(probability)*100:.1f}%**")
            # Standard progress bar
            st.progress(risk_prob)
            st.info(f"The calculated risk probability is {risk_prob*100:.1f}%. Please consult with a medical professional for a formal diagnosis.")
else:
    st.error("System is currently unavailable. Please ensure 'heart_disease_model.pkl' and 'scaler.pkl' are in the application directory.")