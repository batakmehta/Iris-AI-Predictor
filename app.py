import streamlit as st
import pandas as pd
import pickle

# ==========================================
# LOAD MODEL
# ==========================================
with open("all_models.pkl", "rb") as f:
    models = pickle.load(f)

feature_names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

target_names = ["setosa", "versicolor", "virginica"]

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Iris AI Predictor", layout="centered")

# ==========================================
# CUSTOM STYLE
# ==========================================
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: white;
    }
    .subtitle {
        text-align: center;
        color: #9CA3AF;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="title">🌸 Iris AI Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter flower measurements and let AI predict</div>', unsafe_allow_html=True)

# ==========================================
# INPUT SECTION
# ==========================================
st.subheader("📥 Input Features")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.2)

with col2:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 1.5)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 0.3)

# Create DataFrame
input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=feature_names
)

st.write("### 📊 Input Data")
st.dataframe(input_data)

st.write("### 📊 Input Data")
st.dataframe(input_data)

# ==========================================
# MODEL SELECTION  ✅ ADD HERE
# ==========================================
model_choice = st.selectbox(
    "🤖 Select Model",
    list(models.keys())
)

# ==========================================
# PREDICTION
# ==========================================
if st.button("🔍 Predict"):

    selected_model = models[model_choice]

    prediction = selected_model.predict(input_data)[0]
    probabilities = selected_model.predict_proba(input_data)[0]

    predicted_species = target_names[prediction]
    confidence = max(probabilities) * 100

    # Color logic
    if confidence > 90:
        color = "#16A34A"
    elif confidence > 70:
        color = "#F59E0B"
    else:
        color = "#EF4444"
    # RESULT BOX
    st.markdown(
        f"""
        <div class="result-box" style="background-color:{color}; color:white;">
        🌼 Prediction: {predicted_species.capitalize()} <br>
        Confidence: {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # PROBABILITY CHART
    # ==========================================
    st.write("### 📈 Model Confidence Breakdown")

    prob_df = pd.DataFrame({
        "Species": target_names,
        "Probability": probabilities
    })

    st.bar_chart(prob_df.set_index("Species"))

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Built with ❤️ by Ayushi | ML + UI = Product 🚀")