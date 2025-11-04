import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Flu Identification Dashboard", page_icon="🦠", layout="centered")

st.title("🦠 Type of Flu Identification Dashboard")

# Sidebar for file upload
st.sidebar.header("📁 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a flu dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head())

    st.write("### 📈 Flu Type Distribution")
    fig, ax = plt.subplots()
    df['Type_of_Flu'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel("Type of Flu")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Create a separate encoder for features and target
    feature_encoders = {}
    data = df.copy()

    # Encode input columns (except target)
    for col in data.columns:
        if col != "Type_of_Flu":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            feature_encoders[col] = le

    # Encode target separately
    flu_encoder = LabelEncoder()
    data["Type_of_Flu"] = flu_encoder.fit_transform(data["Type_of_Flu"])

    X = data.drop("Type_of_Flu", axis=1)
    y = data["Type_of_Flu"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    st.success(f"✅ Model Trained Successfully! Accuracy: {accuracy*100:.2f}%")

    st.write("### 🤖 Predict Type of Flu")
    col1, col2 = st.columns(2)
    with col1:
        fever = st.selectbox("Fever", ["High", "Mild", "None"])
        cough = st.selectbox("Cough", ["Dry", "Wet", "None"])
        headache = st.selectbox("Headache", ["Yes", "No"])
    with col2:
        fatigue = st.selectbox("Fatigue", ["Severe", "Moderate", "Mild"])
        body_pain = st.selectbox("Body Pain", ["Yes", "No"])

    # Prepare user input as a DataFrame
    input_df = pd.DataFrame({
        "Fever": [fever],
        "Cough": [cough],
        "Headache": [headache],
        "Fatigue": [fatigue],
        "Body_Pain": [body_pain]
    })

    # Apply same encoding as training
    for col in input_df.columns:
        input_df[col] = feature_encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]
    flu_type = flu_encoder.inverse_transform([prediction])[0]

    st.info(f"🧾 Predicted Type of Flu: **{flu_type}**")

else:
    st.warning("👈 Please upload your 'flu_data.csv' file from the sidebar to start.")
