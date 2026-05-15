import streamlit as st
import pandas as pd
import pickle

# ==========================================
# LOAD MODELS
# ==========================================
with open("all_models.pkl", "rb") as f:
    models = pickle.load(f)

# ==========================================
# FEATURE & TARGET NAMES
# ==========================================
feature_names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

target_names = ["setosa", "versicolor", "virginica"]

# ==========================================
# MODEL ACCURACY
# ==========================================
model_accuracy = {
    "Logistic Regression": 96,
    "KNN": 97,
    "Decision Tree": 93,
    "SVM": 98
}

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Iris AI Predictor",
    layout="centered"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    font-size: 42px;
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
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<div class="title">🌸 Iris AI Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Flower Classification System</div>',
    unsafe_allow_html=True
)

# ==========================================
# INPUT SECTION
# ==========================================
st.subheader("📥 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider(
        "Sepal Length",
        4.0, 8.0, 5.1
    )

    sepal_width = st.slider(
        "Sepal Width",
        2.0, 4.5, 3.5
    )

with col2:
    petal_length = st.slider(
        "Petal Length",
        1.0, 7.0, 1.4
    )

    petal_width = st.slider(
        "Petal Width",
        0.1, 2.5, 0.2
    )

# ==========================================
# INPUT DATAFRAME
# ==========================================
input_data = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=feature_names
)

st.write("### 📊 Input Data")
st.dataframe(input_data)

# ==========================================
# MODEL SELECTION
# ==========================================
st.write("### 🤖 Select Machine Learning Model")

model_choice = st.selectbox(
    "Choose Model",
    list(models.keys())
)

# ==========================================
# SHOW MODEL ACCURACY
# ==========================================
st.info(
    f"✅ Accuracy of {model_choice}: "
    f"{model_accuracy[model_choice]}%"
)

# ==========================================
# PREDICTION
# ==========================================
if st.button("🔍 Predict Species"):

    selected_model = models[model_choice]

    prediction = selected_model.predict(input_data)[0]

    probabilities = selected_model.predict_proba(input_data)[0]

    predicted_species = target_names[prediction]

    confidence = max(probabilities) * 100

    # ==========================================
    # COLOR LOGIC
    # ==========================================
    if confidence > 90:
        color = "#16A34A"

    elif confidence > 70:
        color = "#F59E0B"

    else:
        color = "#EF4444"

    # ==========================================
    # RESULT BOX
    # ==========================================
    st.markdown(
        f"""
        <div class="result-box"
        style="background-color:{color}; color:white;">

        🌼 Prediction: {predicted_species.capitalize()} <br><br>

        🎯 Confidence Score: {confidence:.2f}%

        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # PROBABILITY CHART
    # ==========================================
    st.write("### 📈 Prediction Probability Graph")

    prob_df = pd.DataFrame({
        "Species": target_names,
        "Probability": probabilities
    })

    st.bar_chart(prob_df.set_index("Species"))

    # ==========================================
    # BEST MODEL
    # ==========================================
    best_model = max(
        model_accuracy,
        key=model_accuracy.get
    )

    st.success(
        f"🏆 Best Performing Model: "
        f"{best_model} "
        f"with Accuracy "
        f"{model_accuracy[best_model]}%"
    )

    # ==========================================
    # MODEL COMPARISON GRAPH
    # ==========================================
    st.write("### 📊 Model Accuracy Comparison")

    accuracy_df = pd.DataFrame({
        "Model": list(model_accuracy.keys()),
        "Accuracy": list(model_accuracy.values())
    })

    st.bar_chart(
        accuracy_df.set_index("Model")
    )

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")

st.caption(
    "Built with ❤️ by Ayushi | "
    "Machine Learning + Streamlit 🚀"
)
