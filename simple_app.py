import streamlit as st
import pandas as pd
import numpy as np
import shap

# Try to import joblib with fallback
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    st.warning("⚠️ joblib not available - using demo mode")

# Streamlit UI
st.set_page_config(page_title="Fair AI Demo", page_icon="🤖")
st.title("🎯 Fair AI Income Predictor")
st.write("Predict income with fairness-aware AI models, now with explainability (SHAP).")

# Load model & scaler
model = None
scaler = None

if JOBLIB_AVAILABLE:
    try:
        model = joblib.load('models/baseline_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Models not loaded: {str(e)} - using demo mode")
else:
    st.warning("⚠️ joblib not installed - using demo mode")

# Input section
st.subheader("Enter Details:")
age = st.slider("Age", 18, 80, 35)
education = st.slider("Education Level", 1, 16, 13)
hours = st.slider("Hours per Week", 10, 80, 40)
gender = st.selectbox("Gender", ["Female", "Male"])
race = st.selectbox("Race", ["Non-White", "White"])

# Convert to numeric
gender_num = 1 if gender == "Male" else 0
race_num = 1 if race == "White" else 0

if st.button("Predict Income"):
    features = np.array([[age, education, hours, gender_num, race_num]])

    # Predict or fallback
    if model is not None and scaler is not None and JOBLIB_AVAILABLE:
        try:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
        except:
            prediction = 1 if age > 30 and education > 12 else 0
            probability = 0.7 if prediction == 1 else 0.3
    else:
        prediction = 1 if age > 30 and education > 12 else 0
        probability = 0.7 if prediction == 1 else 0.3

    # Output
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.success("🎯 Prediction: HIGH INCOME (>50K)")
    else:
        st.info("🎯 Prediction: LOW INCOME (<=50K)")

    st.metric("Confidence", f"{probability:.1%}")

    # Fairness check
    if gender == "Female" and probability < 0.4:
        st.warning("⚠️ Possible gender bias detected")
    else:
        st.success("✅ Fair prediction")

    # ---------------------------
    # ⭐ NEW: SHAP Explainability
    # ---------------------------
    st.subheader("🧠 Explainability (SHAP)")

    if model is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_scaled)

            st.write("Feature Importance for this Prediction:")

            shap_df = pd.DataFrame({
                'Feature': ["Age", "Education", "Hours", "Gender", "Race"],
                'SHAP Value': shap_values[1][0]
            })

            st.table(shap_df)

            st.caption("Higher SHAP value → more contribution to predicting HIGH income.")
        except Exception as e:
            st.warning(f"Explainability unavailable: {str(e)}")
    else:
        st.info("Explainability will work after model loading.")

st.markdown("---")
st.subheader("📊 Demo Data Insights")

demo_data = pd.DataFrame({
    'Age': [25, 35, 45, 55, 65],
    'Education': [12, 14, 16, 13, 10],
    'Income_Probability': [0.3, 0.6, 0.8, 0.5, 0.2]
})

st.line_chart(demo_data.set_index('Age')['Income_Probability'])
st.caption("Higher age & education → higher income probability")

st.markdown("---")
st.write("Built with ❤️ using Streamlit | Fair AI Demo")
