import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open('student_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🎓 Student Performance Predictor")
st.markdown("AI & ML Powered Performance Predictor")
st.markdown("Predict whether a student is likely to **Pass or Fail** based on their background.")
st.divider()

with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This model was trained on the **Students Performance in Exams**
    dataset using a **Random Forest Classifier**.

    **Features used:**
    - Gender
    - Race/Ethnicity
    - Parental Education
    - Lunch Type
    - Test Preparation
    """)
    st.warning("⚠️ Note: Model accuracy is limited by class "
               "imbalance (97% Pass vs 3% Fail in training data).")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox(
        "Race/Ethnicity",
        ["Group A", "Group B", "Group C", "Group D", "Group E"]
    )
    parent_edu = st.selectbox(
        "Parental Level of Education",
        ["Some High School", "High School", "Some College",
         "Associate's Degree", "Bachelor's Degree", "Master's Degree"]
    )

with col2:
    lunch = st.selectbox("Lunch Type", ["Standard", "Free/Reduced"])
    test_prep = st.selectbox("Test Preparation Course", ["None", "Completed"])

st.divider()

encode_map = {
    "gender": {"Male": 1, "Female": 0},
    "ethnicity": {"Group A": 0, "Group B": 1, "Group C": 2,
                  "Group D": 3, "Group E": 4},
    "parent_edu": {"Some High School": 4, "High School": 1,
                   "Some College": 5, "Associate's Degree": 0,
                   "Bachelor's Degree": 2, "Master's Degree": 3},
    "lunch": {"Standard": 1, "Free/Reduced": 0},
    "test_prep": {"None": 1, "Completed": 0}
}

if st.button("🔍 Predict Performance", type="primary", use_container_width=True):
    input_data = pd.DataFrame([[
        encode_map["gender"][gender],
        encode_map["ethnicity"][ethnicity],
        encode_map["parent_edu"][parent_edu],
        encode_map["lunch"][lunch],
        encode_map["test_prep"][test_prep],
    ]], columns=['gender', 'race/ethnicity',
                 'parental level of education',
                 'lunch', 'test preparation course'])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.success("✅ Likely to **PASS**")
        else:
            st.error("⚠️ At **RISK** of Failing")

    with result_col2:
        st.metric("Pass Probability", f"{probability[1]*100:.1f}%")
        st.metric("Fail Probability", f"{probability[0]*100:.1f}%")

    st.progress(float(probability[1]))

    if prediction == 0:
        st.info("💡 **Recommendation:** Consider enrolling in a "
                "test preparation course and seeking academic support.")

st.divider()

st.subheader("📊 What Influences the Prediction Most?")

importance_data = pd.DataFrame({
    'Feature': ['Race/Ethnicity', 'Parental Education',
                'Lunch Type', 'Gender', 'Test Preparation'],
    'Importance': [0.356, 0.338, 0.144, 0.082, 0.079]
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(importance_data['Feature'], importance_data['Importance'], color='#4C72B0')
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance')
st.pyplot(fig)

st.caption("Built for School Student🚀")