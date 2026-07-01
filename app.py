import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
# MAPPING LABEL ENCODING
# Mapping ini hasil fit LabelEncoder pada dataset training
# (sales_marketing_customer_dataset.csv) — HARUS sama persis
# dengan yang dipakai saat training supaya prediksi valid.
# ──────────────────────────────────────────
CITY_MAP = {'Berlin': 0, 'Delhi': 1, 'Dhaka': 2, 'Hamburg': 3, 'London': 4, 'Mumbai': 5, 'New York': 6}
COUPON_MAP = {'NEW20': 0, 'REF10': 1, 'SALE15': 2, 'Tidak Ada Kupon': -1}
SUBSCRIPTION_MAP = {'Annual': 0, 'Monthly': 1}
 
# Urutan kolom persis seperti scaler.feature_names_in_ saat training
EXPECTED_COLS = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else None
 
# ──────────────────────────────────────────
# FORM INPUT
# ──────────────────────────────────────────
st.subheader("🔮 Masukkan Data Customer")
 
col1, col2 = st.columns(2)
 
with col1:
    age               = st.number_input("Usia", min_value=17, max_value=95, value=30)
    gender            = st.selectbox("Gender", ["Male", "Female", "Other"])
    country           = st.selectbox("Country", ["USA", "UK", "Germany", "India", "Bangladesh"])
    city              = st.selectbox("City", list(CITY_MAP.keys()))
    subscription_type = st.selectbox("Subscription Type", list(SUBSCRIPTION_MAP.keys()))
    is_premium_user   = st.selectbox("Premium User?", ["Ya", "Tidak"])
    acquisition_ch    = st.selectbox("Acquisition Channel", ["Email", "Organic", "Facebook Ads", "Referral", "Google Ads"])
    device_type       = st.selectbox("Device Type", ["Desktop", "Mobile", "Tablet"])
    payment_method    = st.selectbox("Payment Method", ["UPI", "BKash", "PayPal", "SEPA", "Card"])
    tenure_days_val   = st.number_input("Tenure Days (lama berlangganan)", min_value=0, value=365)
 
with col2:
    total_visits      = st.number_input("Total Visits", min_value=0, value=15)
    avg_session_time  = st.number_input("Avg Session Time (menit)", min_value=0.0, value=8.0)
    pages_per_session  = st.number_input("Pages per Session", min_value=0.0, value=4.0)
    email_open_rate   = st.slider("Email Open Rate", 0.0, 1.0, 0.3)
    email_click_rate  = st.slider("Email Click Rate", 0.0, 1.0, 0.2)
    total_spent       = st.number_input("Total Spent ($)", min_value=0.0, value=500.0)
    avg_order_value   = st.number_input("Avg Order Value ($)", min_value=0.0, value=70.0)
    support_tickets   = st.number_input("Support Tickets", min_value=0, value=1)
    delivery_delay    = st.number_input("Delivery Delay (hari)", min_value=0, value=2)
    satisfaction      = st.slider("Satisfaction Score", 1, 5, 3)
 
col3, col4 = st.columns(2)
with col3:
    discount_used      = st.selectbox("Pernah Pakai Diskon?", ["Ya", "Tidak"])
    refund_requested   = st.selectbox("Pernah Minta Refund?", ["Ya", "Tidak"])
    coupon_code        = st.selectbox("Coupon Code", list(COUPON_MAP.keys()))
with col4:
    nps_score          = st.slider("NPS Score", 0, 10, 5)
    marketing_spend    = st.number_input("Marketing Spend per User ($)", min_value=0.0, value=15.0)
    lifetime_value      = st.number_input("Lifetime Value ($)", min_value=0.0, value=800.0)
 
last_3m_freq = st.number_input("Frekuensi Beli 3 Bulan Terakhir", min_value=0, value=5)
 
# ──────────────────────────────────────────
# PREPROCESSING INPUT
# ──────────────────────────────────────────
def preprocess_input(raw):
    df = pd.DataFrame([raw])
 
    # Label Encoding (pakai mapping hasil training, BUKAN fit ulang)
    df['city'] = df['city'].map(CITY_MAP)
    df['coupon_code'] = df['coupon_code'].map(COUPON_MAP)
    df['subscription_type'] = df['subscription_type'].map(SUBSCRIPTION_MAP)
 
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
        'city': city,
        'subscription_type': subscription_type,
        'is_premium_user': 1 if is_premium_user == "Ya" else 0,
        'total_visits': total_visits,
        'avg_session_time': avg_session_time,
        'pages_per_session': pages_per_session,
        'email_open_rate': email_open_rate,
        'email_click_rate': email_click_rate,
        'total_spent': total_spent,
        'avg_order_value': avg_order_value,
        'discount_used': 1 if discount_used == "Ya" else 0,
        'coupon_code': coupon_code,
        'support_tickets': support_tickets,
        'refund_requested': 1 if refund_requested == "Ya" else 0,
        'delivery_delay_days': delivery_delay,
        'satisfaction_score': satisfaction,
        'nps_score': nps_score,
        'marketing_spend_per_user': marketing_spend,
        'lifetime_value': lifetime_value,
        'last_3_month_purchase_freq': last_3m_freq,
        'tenure_days': tenure_days_val,
        'gender': gender,
        'country': country,
        'acquisition_channel': acquisition_ch,
        'device_type': device_type,
        'payment_method': payment_method,
    }
 
    input_df = preprocess_input(raw_input)
 
    # Align kolom persis ke urutan fitur saat training
    if EXPECTED_COLS is not None:
        for col in EXPECTED_COLS:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[EXPECTED_COLS]
 
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
