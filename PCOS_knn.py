import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi PCOS", layout="centered")

# CSS styling
st.markdown("""
    <style>
        .block-container {
            background-color: #fafcfd;
            padding: 3rem;
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            max-width: 800px;
            margin-bottom: 2rem;
            margin-top: 7rem;
        }
        .header {
            background: linear-gradient(90deg, #B82132, #FFA09B);
            padding: 1.2rem 1rem;
            border-radius: 12px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }
        .header h1 {
            margin-bottom: 0.3rem;
        }
        .header p {
            font-size: 1rem;
            opacity: 0.9;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            font-weight: bold;
            padding: 0.6rem 2rem;
            border-radius: 8px;
            border: none;
        }
        .result-box {
            padding: 1rem;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
            font-size: 1.2rem;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class='header'>
        <h1>🔬 Prediksi Penyakit PCOS</h1>
        <p>Deteksi awal sindrom ovarium polikistik menggunakan metode K-Nearest Neighbors (KNN)</p>
    </div>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("data_without_infertility_final.csv")
features = ['Follicle No. (R)', 'AMH(ng/mL)', 'BMI', 'Weight (Kg)', 'Cycle length(days)']
label = 'PCOS (Y/N)'

label_mapping = {
    'Follicle No. (R)': '🧪 Jumlah Folikel (Kanan)',
    'AMH(ng/mL)': '💉 Kadar AMH (ng/mL)',
    'BMI': '⚖️ Indeks Massa Tubuh (BMI)',
    'Weight (Kg)': '🏋️ Berat Badan (Kg)',
    'Cycle length(days)': '🩸 Panjang Siklus Menstruasi (hari)'
}

# Data cleaning
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=features + [label])

# Pastikan label dalam bentuk numerik
if df[label].dtype == object:
    df[label] = df[label].map({'Y': 1, 'N': 0})
else:
    df[label] = df[label].astype(int)

X = df[features]
y = df[label]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE untuk penyeimbangan kelas
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Split data setelah SMOTE
k = 6  # Sesuai jurnal
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Train model
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Evaluasi model
acc = accuracy_score(y_test, model.predict(X_test))
st.info(f"🎯 Akurasi model pada data uji (K={k} dengan SMOTE): **{acc*100:.2f}%**")

# Input form
st.markdown("### 📋 Masukkan Data Pemeriksaan")
input_data = []
col1, col2 = st.columns(2)
for i, col in enumerate(features):
    label_text = label_mapping.get(col, col)
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        val = st.number_input(label_text, min_value=0.0, step=0.1, format="%.2f")
        input_data.append(val)

# Predict button
if st.button("🔍 PREDIKSI SEKARANG"):
    if any(np.isnan(input_data)) or any(val == 0 for val in input_data):
        st.warning("⚠️ Harap isi semua field dengan benar sebelum melakukan prediksi.")
    else:
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.markdown(f"<div class='result-box' style='background-color:#ffe6e6; color:#cc0000;'>💡 Hasil: Terindikasi PCOS</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box' style='background-color:#e6ffec; color:#007e33;'>✅ Hasil: Tidak Terindikasi PCOS</div>", unsafe_allow_html=True)

        # Visual pie chart
        fig, ax = plt.subplots()
        labels = ["PCOS", "Tidak PCOS"]
        sizes = [1, 0] if prediction == 1 else [0, 1]
        colors = ["#ff9999", "#99ff99"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

# Footer
st.markdown("""
    <hr style="margin-top:3rem;">
    <div style='text-align:center; font-size:0.85rem; color:gray;'>
        🚀 Final Project Kecerdasan Buatan | PCOS Classifier © 2025
    </div>
""", unsafe_allow_html=True)
