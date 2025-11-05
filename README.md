🦠 Flu Identification Dashboard

A simple Streamlit-based Flu Type Identification Dashboard (demo project) with features:

📁 Upload flu dataset (CSV file)

📊 View dataset preview and flu type distribution

⚙ Train a Random Forest model automatically

🎯 Predict the type of flu based on symptoms

💾 Encodes and processes categorical features automatically



---

🚀 Run Locally

1. Create virtual environment:

python -m venv venv
source venv/bin/activate        # for Linux/Mac  
venv\Scripts\activate           # for Windows


2. Install dependencies:

pip install -r requirements.txt


3. Run Streamlit app:

streamlit run flu_identification_dashboard.py


4. Open in browser:
👉 http://localhost:8501




---

🧾 Notes

This is a simple educational demo project using Streamlit and scikit-learn.

For real-world use, connect to a database and improve the model with real medical datasets.

Data preprocessing and label encoding are handled automatically inside the app.



---
