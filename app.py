import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.caption("UAS Bengkel Koding Data Science · Gabriella Jovanka Bustan · A11.2023.14861")
st.markdown("---")

# ──────────────────────────────────────────
# LOAD MODEL & SCALER
# ──────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("best_churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.success("✅ Model berhasil dimuat.")
except FileNotFoundError:
    st.error("❌ File `best_churn_model.pkl` atau `scaler.pkl` tidak ditemukan. "
             "Pastikan kedua file ada di folder yang sama dengan app.py.")
    st.stop()

# ──────────────────────────────────────────
# FORM INPUT
# ──────────────────────────────────────────
st.subheader("🔮 Masukkan Data Customer")

col1, col2 = st.columns(2)

with col1:
    age             = st.number_input("Usia", min_value=18, max_value=100, value=30)
    annual_income   = st.number_input("Annual Income", min_value=0, value=50000, step=1000)
    purchase_amount = st.number_input("Purchase Amount", min_value=0.0, value=500.0)
    num_purchases   = st.number_input("Num of Purchases", min_value=0, value=10)
    avg_order_val   = st.number_input("Avg Order Value", min_value=0.0, value=100.0)
    satisfaction    = st.slider("Satisfaction Score", 1, 5, 3)
    loyalty_score   = st.number_input("Loyalty Score", min_value=0.0, value=50.0)
    support_tickets = st.number_input("Support Tickets", min_value=0, value=1)

with col2:
    email_open_rate  = st.slider("Email Open Rate", 0.0, 1.0, 0.3)
    click_through    = st.slider("Click Through Rate", 0.0, 1.0, 0.1)
    promotion_resp   = st.slider("Promotion Response Rate", 0.0, 1.0, 0.2)
    social_media_eng = st.number_input("Social Media Engagement", min_value=0.0, value=5.0)
    website_visits   = st.number_input("Website Visits", min_value=0, value=20)
    tenure_days_val  = st.number_input("Tenure Days", min_value=0, value=365)
    gender           = st.selectbox("Gender", ["Male", "Female", "Other"])
    subscription     = st.selectbox("Subscription Type", ["Basic", "Premium", "Enterprise"])

col3, col4 = st.columns(2)
with col3:
    country        = st.selectbox("Country", ["USA", "UK", "Canada", "Australia", "Germany"])
    acquisition_ch = st.selectbox("Acquisition Channel", ["Online", "Offline", "Referral", "Social Media"])
with col4:
    device_type    = st.selectbox("Device Type", ["Desktop", "Mobile", "Tablet"])
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Cash"])

city        = st.text_input("City", "Jakarta")
coupon_code = st.text_input("Coupon Code", "NONE")

# ──────────────────────────────────────────
# PREPROCESSING INPUT
# ──────────────────────────────────────────
def preprocess_input(raw):
    df = pd.DataFrame([raw])

    # Label Encoding (ordinal / high-cardinality)
    for col in ['city', 'coupon_code', 'subscription_type']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # One-Hot Encoding
    ohe_cols = ['gender', 'country', 'acquisition_channel', 'device_type', 'payment_method']
    df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns], drop_first=False)

    return df

# ──────────────────────────────────────────
# TOMBOL PREDIKSI
# ──────────────────────────────────────────
st.markdown("---")
if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):

    raw_input = {
        'age': age,
        'annual_income': annual_income,
        'purchase_amount': purchase_amount,
        'num_of_purchases': num_purchases,
        'avg_order_value': avg_order_val,
        'satisfaction_score': satisfaction,
        'loyalty_score': loyalty_score,
        'support_tickets': support_tickets,
        'email_open_rate': email_open_rate,
        'click_through_rate': click_through,
        'promotion_response_rate': promotion_resp,
        'social_media_engagement': social_media_eng,
        'website_visits': website_visits,
        'tenure_days': tenure_days_val,
        'gender': gender,
        'country': country,
        'acquisition_channel': acquisition_ch,
        'device_type': device_type,
        'payment_method': payment_method,
        'subscription_type': subscription,
        'city': city,
        'coupon_code': coupon_code,
    }

    input_df = preprocess_input(raw_input)

    # Align kolom ke fitur yang dipakai saat training
    expected_cols = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else input_df.columns
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols]

    # Scale & predict
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]

    # Hasil
    if pred == 1:
        st.error("⚠️ **CHURN** — Customer ini diprediksi akan berhenti berlangganan.")
    else:
        st.success("✅ **TIDAK CHURN** — Customer ini diprediksi akan tetap berlangganan.")

    # Probabilitas (jika model mendukung)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0]
        c1, c2 = st.columns(2)
        c1.metric("Probabilitas Tidak Churn", f"{prob[0]*100:.1f}%")
        c2.metric("Probabilitas Churn",        f"{prob[1]*100:.1f}%")

# ──────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────
st.markdown("---")
st.caption("Universitas Dian Nuswantoro · 2024")
