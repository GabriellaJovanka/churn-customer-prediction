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
# LOAD MODEL & ARTIFACTS
# ──────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model           = joblib.load("best_churn_model.pkl")
    scaler          = joblib.load("scaler.pkl")
    label_encoders  = joblib.load("label_encoders.pkl")   # dict: {col_name: fitted LabelEncoder}
    feature_columns = joblib.load("feature_columns.pkl")  # urutan kolom X_prep saat training
    return model, scaler, label_encoders, feature_columns
 
try:
    model, scaler, label_encoders, feature_columns = load_artifacts()
    st.success("✅ Model berhasil dimuat.")
except FileNotFoundError:
    st.error("❌ File `best_churn_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, atau "
             "`feature_columns.pkl` tidak ditemukan. Pastikan keempat file ada di folder "
             "yang sama dengan app.py.")
    st.stop()
 
# Kategori-kategori ini diambil langsung dari label_encoders.pkl / dataset asli,
# supaya pilihan di form selalu konsisten dengan apa yang dikenali model.
CITY_OPTIONS         = list(label_encoders['city'].classes_)
SUBSCRIPTION_OPTIONS = list(label_encoders['subscription_type'].classes_)
 
# Saat training, missing value pada coupon_code diberi label eksplisit
# "No Coupon" SEBELUM encoding (lihat notebook, cell 4c) — jadi "No Coupon"
# adalah kelas asli yang dikenali label encoder, bukan nilai "nan" buatan.
NO_COUPON_LABEL = "No Coupon"
COUPON_NONE_LABEL = "Tidak Pakai Kupon"  # label yang ditampilkan ke user di form
 
_coupon_classes = list(label_encoders['coupon_code'].classes_)
# Tampilkan "Tidak Pakai Kupon" di posisi teratas dropdown, tanpa duplikasi
# jika NO_COUPON_LABEL sudah ada di antara classes_.
COUPON_OPTIONS = [COUPON_NONE_LABEL] + [c for c in _coupon_classes if c != NO_COUPON_LABEL]
 
GENDER_OPTIONS      = ["Male", "Female", "Other"]
COUNTRY_OPTIONS     = ["USA", "Germany", "India", "UK", "Bangladesh"]
CHANNEL_OPTIONS     = ["Email", "Organic", "Facebook Ads", "Referral", "Google Ads"]
DEVICE_OPTIONS      = ["Desktop", "Mobile", "Tablet"]
PAYMENT_OPTIONS     = ["UPI", "BKash", "PayPal", "Card", "SEPA"]
 
# ──────────────────────────────────────────
# FORM INPUT
# ──────────────────────────────────────────
st.subheader("🔮 Masukkan Data Customer")
 
col1, col2 = st.columns(2)
 
with col1:
    age               = st.number_input("Usia", min_value=18, max_value=95, value=35)
    gender            = st.selectbox("Gender", GENDER_OPTIONS)
    country           = st.selectbox("Country", COUNTRY_OPTIONS)
    city              = st.selectbox("City", CITY_OPTIONS)
    subscription_type = st.selectbox("Subscription Type", SUBSCRIPTION_OPTIONS)
    is_premium_user   = st.selectbox("Premium User?", ["Tidak", "Ya"])
    acquisition_ch    = st.selectbox("Acquisition Channel", CHANNEL_OPTIONS)
    device_type       = st.selectbox("Device Type", DEVICE_OPTIONS)
    payment_method    = st.selectbox("Payment Method", PAYMENT_OPTIONS)
    tenure_days       = st.number_input("Tenure Days (lama jadi pelanggan)", min_value=0, value=300)
 
with col2:
    total_visits         = st.number_input("Total Visits", min_value=0, value=15)
    avg_session_time     = st.number_input("Avg Session Time (menit)", min_value=0.0, value=8.0)
    pages_per_session    = st.number_input("Pages per Session", min_value=0.0, value=4.0)
    email_open_rate      = st.slider("Email Open Rate", 0.0, 1.0, 0.5)
    email_click_rate     = st.slider("Email Click Rate", 0.0, 0.5, 0.25)
    total_spent          = st.number_input("Total Spent ($)", min_value=0.0, value=500.0)
    avg_order_value      = st.number_input("Avg Order Value ($)", min_value=0.0, value=60.0)
    last_3m_purchase_freq = st.number_input("Frekuensi Beli 3 Bulan Terakhir", min_value=0, value=7)
 
col3, col4 = st.columns(2)
with col3:
    discount_used     = st.selectbox("Pernah Pakai Diskon?", ["Tidak", "Ya"])
    coupon_code       = st.selectbox("Coupon Code", COUPON_OPTIONS)
    support_tickets   = st.number_input("Jumlah Support Tickets", min_value=0, value=2)
    refund_requested  = st.selectbox("Pernah Minta Refund?", ["Tidak", "Ya"])
with col4:
    delivery_delay_days      = st.number_input("Delivery Delay (hari)", min_value=0, value=3)
    satisfaction_score       = st.slider("Satisfaction Score", 1, 5, 4)
    nps_score                = st.slider("NPS Score", 0, 10, 5)
    marketing_spend_per_user = st.number_input("Marketing Spend per User ($)", min_value=0.0, value=17.5)
 
lifetime_value = st.number_input("Lifetime Value ($)", min_value=0.0, value=1200.0)
 
# ──────────────────────────────────────────
# PREPROCESSING INPUT
# ──────────────────────────────────────────
def preprocess_input(raw):
    df = pd.DataFrame([raw])
 
    # Label Encoding — pakai encoder yang SUDAH di-fit saat training,
    # bukan fit_transform baru. fit_transform pada 1 baris data akan selalu
    # menghasilkan 0 apa pun isinya, jadi model jadi tidak sensitif terhadap
    # perubahan input di form.
    for col in ['city', 'coupon_code', 'subscription_type']:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            val = df[col].astype(str)
 
            unseen_mask = ~val.isin(le.classes_)
            if unseen_mask.any():
                st.warning(
                    f"⚠️ Nilai '{val[unseen_mask].iloc[0]}' pada kolom '{col}' tidak "
                    f"dikenali dari data training. Menggunakan kategori paling umum sebagai gantinya."
                )
                fallback_value = le.classes_[0]
                val = val.where(~unseen_mask, fallback_value)
 
            df[col] = le.transform(val)
 
    # One-Hot Encoding untuk kolom nominal low-cardinality
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
        'gender': gender,
        'country': country,
        'city': city,
        'acquisition_channel': acquisition_ch,
        'device_type': device_type,
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
        # Samakan persis dengan cara training: missing coupon -> "No Coupon",
        # BUKAN string "nan" (yang tidak pernah ada di data training).
        'coupon_code': NO_COUPON_LABEL if coupon_code == COUPON_NONE_LABEL else coupon_code,
        'support_tickets': support_tickets,
        'refund_requested': 1 if refund_requested == "Ya" else 0,
        'delivery_delay_days': delivery_delay_days,
        'payment_method': payment_method,
        'satisfaction_score': satisfaction_score,
        'nps_score': nps_score,
        'marketing_spend_per_user': marketing_spend_per_user,
        'lifetime_value': lifetime_value,
        'last_3_month_purchase_freq': last_3m_purchase_freq,
        'tenure_days': tenure_days,
    }
 
    input_df = preprocess_input(raw_input)
 
    # Align kolom ke fitur yang dipakai saat training (urutan & nama harus persis sama)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
 
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
