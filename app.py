import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("best_churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-title   { font-size: 1rem;   color: #555;       margin-top: 4px; margin-bottom: 24px; }
    .section-hdr { font-size: 1.1rem; font-weight: 600;  color: #16213e; border-left: 4px solid #0f3460;
                   padding-left: 10px; margin: 20px 0 12px; }
    .churn-box   { background: #ffe0e0; border-left: 5px solid #e63946; border-radius: 8px;
                   padding: 18px 22px; margin-top: 16px; }
    .safe-box    { background: #e0f7ea; border-left: 5px solid #2dc653; border-radius: 8px;
                   padding: 18px 22px; margin-top: 16px; }
    .prob-label  { font-size: 1.05rem; font-weight: 600; margin-bottom: 4px; }
    .metric-card { background: #f0f4ff; border-radius: 8px; padding: 12px 16px; text-align: center; }
    .stButton > button { background: #0f3460; color: white; font-weight: 600;
                         border-radius: 8px; padding: 10px 28px; border: none; }
    .stButton > button:hover { background: #16213e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">📊 Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Bengkel Koding Data Science — UAS | Gabriella Jovanka Bustan | A11.2023.14861</p>', unsafe_allow_html=True)
st.divider()

if not model_loaded:
    st.error("⚠️ File `best_churn_model.pkl` atau `scaler.pkl` tidak ditemukan. Pastikan file ada di folder yang sama dengan `app.py`.")
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR — FEATURE INFO
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini memprediksi apakah seorang pelanggan akan **churn** (berhenti berlangganan)
    berdasarkan data demografis, perilaku, dan riwayat transaksi.

    **Pipeline:**
    - Missing value → imputasi median/modus
    - Outlier → IQR capping
    - Feature engineering → `tenure_days`
    - Encoding → Label + One-Hot
    - Balancing → SMOTE
    - Scaling → StandardScaler
    - Model → Best estimator dari GridSearchCV
    """)
    st.divider()
    st.header("📌 Penjelasan Fitur Utama")
    st.markdown("""
    | Fitur | Keterangan |
    |---|---|
    | `satisfaction_score` | Kepuasan 1–5 |
    | `nps_score` | Loyalitas −100 s/d 100 |
    | `total_spent` | Total belanja (USD) |
    | `lifetime_value` | Nilai LTV pelanggan |
    | `support_tickets` | Jumlah tiket komplain |
    | `tenure_days` | Lama berlangganan (hari) |
    | `avg_session_time` | Rata-rata durasi sesi |
    """)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Prediksi Churn", "📖 Panduan Fitur"])

# ══════════════════════════════════════════════
# TAB 1 — INPUT FORM
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-hdr">Data Demografis</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        gender            = st.selectbox("Gender", ["Male", "Female"])
        age               = st.number_input("Usia", min_value=18, max_value=90, value=30)
    with c2:
        country           = st.selectbox("Negara", ["USA", "UK", "Canada", "Australia", "Germany",
                                                     "France", "India", "Brazil", "Japan", "Other"])
        city              = st.selectbox("Kota", ["New York", "London", "Toronto", "Sydney",
                                                   "Berlin", "Paris", "Mumbai", "São Paulo",
                                                   "Tokyo", "Other"])
    with c3:
        acquisition_channel = st.selectbox("Acquisition Channel",
                                           ["Email", "Ads", "Organic", "Referral", "Social Media"])
        device_type         = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])

    st.markdown('<p class="section-hdr">Status Langganan</p>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        subscription_type = st.selectbox("Tipe Langganan", ["Basic", "Standard", "Premium"])
        is_premium_user   = st.selectbox("Pengguna Premium?", [0, 1],
                                         format_func=lambda x: "Ya" if x == 1 else "Tidak")
    with c5:
        signup_date        = st.date_input("Tanggal Daftar", value=pd.to_datetime("2022-01-01"))
        last_purchase_date = st.date_input("Tanggal Pembelian Terakhir", value=pd.to_datetime("2024-01-01"))
    with c6:
        payment_method     = st.selectbox("Metode Pembayaran",
                                          ["Credit Card", "Debit Card", "PayPal",
                                           "Bank Transfer", "Digital Wallet"])
        coupon_code        = st.selectbox("Kode Kupon", ["NONE", "DISC10", "DISC20",
                                                          "SAVE15", "PROMO30"])

    st.markdown('<p class="section-hdr">Perilaku & Engagement</p>', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        total_visits       = st.number_input("Total Kunjungan",         min_value=0,   value=20)
        avg_session_time   = st.number_input("Avg Session Time (menit)",min_value=0.0, value=10.0, step=0.5)
        pages_per_session  = st.number_input("Pages per Session",       min_value=0.0, value=3.0,  step=0.1)
    with c8:
        email_open_rate    = st.slider("Email Open Rate (%)",  0.0, 100.0, 30.0)
        email_click_rate   = st.slider("Email Click Rate (%)", 0.0, 100.0, 10.0)
        discount_used      = st.selectbox("Pernah Pakai Diskon?", [0, 1],
                                          format_func=lambda x: "Ya" if x == 1 else "Tidak")
    with c9:
        support_tickets    = st.number_input("Jumlah Support Tickets",  min_value=0,   value=1)
        refund_requested   = st.selectbox("Pernah Minta Refund?", [0, 1],
                                          format_func=lambda x: "Ya" if x == 1 else "Tidak")
        delivery_delay_days= st.number_input("Delivery Delay (hari)",   min_value=0,   value=2)

    st.markdown('<p class="section-hdr">Nilai & Kepuasan Pelanggan</p>', unsafe_allow_html=True)
    c10, c11, c12 = st.columns(3)
    with c10:
        total_spent        = st.number_input("Total Spent (USD)",       min_value=0.0,  value=500.0,  step=10.0)
        avg_order_value    = st.number_input("Avg Order Value (USD)",   min_value=0.0,  value=50.0,   step=5.0)
    with c11:
        satisfaction_score = st.slider("Satisfaction Score", 1.0, 5.0, 3.5, step=0.1)
        nps_score          = st.slider("NPS Score", -100, 100, 20)
    with c12:
        marketing_spend_per_user = st.number_input("Marketing Spend / User (USD)", min_value=0.0, value=10.0, step=1.0)
        lifetime_value           = st.number_input("Lifetime Value (USD)",          min_value=0.0, value=1000.0, step=50.0)
        last_3_month_purchase_freq = st.number_input("Purchase Freq (3 bulan terakhir)", min_value=0, value=2)

    st.divider()
    predict_btn = st.button("🚀 Prediksi Sekarang", use_container_width=True)

    # ──────────────────────────────────────────
    # PREPROCESSING → harus identik dengan notebook
    # ──────────────────────────────────────────
    if predict_btn:
        # 1. Hitung tenure_days dari input tanggal
        tenure_days = (pd.to_datetime(last_purchase_date) - pd.to_datetime(signup_date)).days

        # 2. Label Encoding manual (city, coupon_code, subscription_type)
        #    Urutan kelas mengikuti LabelEncoder yang fit pada data training (alfabetis)
        city_map = {
            "Berlin": 0, "London": 1, "Mumbai": 2, "New York": 3, "Other": 4,
            "Paris": 5, "Sydney": 6, "São Paulo": 7, "Tokyo": 8, "Toronto": 9
        }
        coupon_map = {
            "DISC10": 0, "DISC20": 1, "NONE": 2, "PROMO30": 3, "SAVE15": 4
        }
        sub_map = {
            "Basic": 0, "Premium": 1, "Standard": 2
        }

        city_enc         = city_map.get(city, 4)          # fallback "Other"
        coupon_enc       = coupon_map.get(coupon_code, 2)  # fallback "NONE"
        sub_enc          = sub_map.get(subscription_type, 0)

        # 3. OHE manual — gender
        gender_Female = 1 if gender == "Female" else 0
        gender_Male   = 1 if gender == "Male"   else 0

        # 4. OHE — country
        countries = ["Australia", "Brazil", "Canada", "France", "Germany",
                     "India", "Japan", "Other", "UK", "USA"]
        country_ohe = {f"country_{c}": (1 if country == c else 0) for c in countries}

        # 5. OHE — acquisition_channel
        channels = ["Ads", "Email", "Organic", "Referral", "Social Media"]
        channel_ohe = {f"acquisition_channel_{ch}": (1 if acquisition_channel == ch else 0) for ch in channels}

        # 6. OHE — device_type
        devices = ["Desktop", "Mobile", "Tablet"]
        device_ohe = {f"device_type_{d}": (1 if device_type == d else 0) for d in devices}

        # 7. OHE — payment_method
        payments = ["Bank Transfer", "Credit Card", "Debit Card", "Digital Wallet", "PayPal"]
        payment_ohe = {f"payment_method_{p}": (1 if payment_method == p else 0) for p in payments}

        # 8. Susun dict fitur (urutan kolom sesuai X_prep.columns setelah notebook dijalankan)
        input_dict = {
            "age":                        age,
            "is_premium_user":            is_premium_user,
            "total_visits":               total_visits,
            "avg_session_time":           avg_session_time,
            "pages_per_session":          pages_per_session,
            "email_open_rate":            email_open_rate,
            "email_click_rate":           email_click_rate,
            "total_spent":                total_spent,
            "avg_order_value":            avg_order_value,
            "discount_used":              discount_used,
            "support_tickets":            support_tickets,
            "refund_requested":           refund_requested,
            "delivery_delay_days":        delivery_delay_days,
            "satisfaction_score":         satisfaction_score,
            "nps_score":                  nps_score,
            "marketing_spend_per_user":   marketing_spend_per_user,
            "lifetime_value":             lifetime_value,
            "last_3_month_purchase_freq": last_3_month_purchase_freq,
            "city":                       city_enc,
            "coupon_code":                coupon_enc,
            "subscription_type":          sub_enc,
            "tenure_days":                tenure_days,
            "gender_Female":              gender_Female,
            "gender_Male":                gender_Male,
            **country_ohe,
            **channel_ohe,
            **device_ohe,
            **payment_ohe,
        }

        input_df = pd.DataFrame([input_dict])

        # 9. Scale
        input_scaled = scaler.transform(input_df)

        # 10. Predict
        prediction   = model.predict(input_scaled)[0]
        try:
            prob_arr = model.predict_proba(input_scaled)[0]
            prob_churn    = prob_arr[1]
            prob_no_churn = prob_arr[0]
            has_proba = True
        except AttributeError:
            has_proba = False

        # ──────────────────────────────────────
        # HASIL
        # ──────────────────────────────────────
        st.divider()
        st.subheader("📋 Hasil Prediksi")

        col_res, col_chart = st.columns([1, 1])

        with col_res:
            if prediction == 1:
                st.markdown("""
                <div class="churn-box">
                  <div style="font-size:2rem">⚠️</div>
                  <div style="font-size:1.4rem; font-weight:700; color:#e63946;">Pelanggan BERPOTENSI CHURN</div>
                  <div style="margin-top:8px; color:#555;">
                    Pelanggan ini diprediksi akan <b>berhenti berlangganan</b>.
                    Pertimbangkan tindakan retensi seperti penawaran khusus atau follow-up personal.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="safe-box">
                  <div style="font-size:2rem">✅</div>
                  <div style="font-size:1.4rem; font-weight:700; color:#2dc653;">Pelanggan AMAN (Tidak Churn)</div>
                  <div style="margin-top:8px; color:#555;">
                    Pelanggan ini diprediksi akan <b>tetap berlangganan</b>.
                    Terus pertahankan kualitas layanan dan engagement.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Metrik singkat
            st.markdown("")
            m1, m2, m3 = st.columns(3)
            m1.metric("Satisfaction", f"{satisfaction_score:.1f} / 5.0")
            m2.metric("NPS Score",    f"{nps_score}")
            m3.metric("Tenure",       f"{tenure_days} hari")

        with col_chart:
            if has_proba:
                st.markdown('<p class="prob-label">Probabilitas Prediksi</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(4, 2.5))
                bars = ax.barh(["Tidak Churn", "Churn"],
                               [prob_no_churn, prob_churn],
                               color=["#2dc653", "#e63946"], height=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probabilitas")
                for bar, val in zip(bars, [prob_no_churn, prob_churn]):
                    ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                            f"{val:.1%}", va="center", fontweight="bold")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Ringkasan input kunci
            st.markdown('<p class="prob-label" style="margin-top:12px">Ringkasan Input</p>',
                        unsafe_allow_html=True)
            summary = pd.DataFrame({
                "Fitur": ["Total Spent", "Lifetime Value", "Support Tickets",
                          "Refund?", "Discount Used", "Pembelian 3 Bln"],
                "Nilai": [f"${total_spent:,.0f}", f"${lifetime_value:,.0f}",
                          support_tickets,
                          "Ya" if refund_requested else "Tidak",
                          "Ya" if discount_used else "Tidak",
                          last_3_month_purchase_freq]
            })
            st.dataframe(summary, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — PANDUAN FITUR
# ══════════════════════════════════════════════
with tab2:
    st.subheader("📖 Panduan Lengkap Fitur")
    st.markdown("""
    | No | Fitur | Tipe | Keterangan |
    |---|---|---|---|
    | 1 | `gender` | Kategorikal | Jenis kelamin (Male / Female) |
    | 2 | `age` | Numerik | Usia pelanggan |
    | 3 | `country` | Kategorikal | Negara asal |
    | 4 | `city` | Kategorikal | Kota pelanggan |
    | 5 | `acquisition_channel` | Kategorikal | Sumber akuisisi (Email, Ads, dll) |
    | 6 | `device_type` | Kategorikal | Jenis perangkat yang digunakan |
    | 7 | `subscription_type` | Kategorikal | Tipe langganan (Basic/Standard/Premium) |
    | 8 | `is_premium_user` | Biner | Status pengguna premium |
    | 9 | `signup_date` | Tanggal | Tanggal mendaftar |
    | 10 | `last_purchase_date` | Tanggal | Tanggal transaksi terakhir |
    | 11 | `tenure_days` | Numerik (derived) | Lama berlangganan dalam hari |
    | 12 | `payment_method` | Kategorikal | Metode pembayaran |
    | 13 | `coupon_code` | Kategorikal | Kode kupon yang digunakan |
    | 14 | `total_visits` | Numerik | Total kunjungan ke platform |
    | 15 | `avg_session_time` | Numerik | Rata-rata durasi sesi (menit) |
    | 16 | `pages_per_session` | Numerik | Rata-rata halaman per sesi |
    | 17 | `email_open_rate` | Numerik | Persentase email yang dibuka |
    | 18 | `email_click_rate` | Numerik | Persentase klik pada email |
    | 19 | `discount_used` | Biner | Pernah menggunakan diskon |
    | 20 | `support_tickets` | Numerik | Jumlah tiket komplain |
    | 21 | `refund_requested` | Biner | Pernah meminta refund |
    | 22 | `delivery_delay_days` | Numerik | Keterlambatan pengiriman (hari) |
    | 23 | `total_spent` | Numerik | Total pengeluaran (USD) |
    | 24 | `avg_order_value` | Numerik | Rata-rata nilai transaksi (USD) |
    | 25 | `satisfaction_score` | Numerik | Skor kepuasan (1.0–5.0) |
    | 26 | `nps_score` | Numerik | Net Promoter Score (−100 s/d 100) |
    | 27 | `marketing_spend_per_user` | Numerik | Biaya marketing per pelanggan (USD) |
    | 28 | `lifetime_value` | Numerik | Nilai total pelanggan selama berlangganan |
    | 29 | `last_3_month_purchase_freq` | Numerik | Frekuensi beli 3 bulan terakhir |
    """)

    st.divider()
    st.subheader("🔄 Pipeline Preprocessing")
    st.markdown("""
    1. **Missing Value** → Numerik diisi median, Kategorikal diisi modus
    2. **Duplikasi** → Baris duplikat dihapus
    3. **Outlier** → IQR Capping (clip ke batas Q1−1.5×IQR dan Q3+1.5×IQR)
    4. **Feature Engineering** → `tenure_days = last_purchase_date − signup_date`
    5. **Encoding** → Label Encoding untuk `city`, `coupon_code`, `subscription_type`;
       One-Hot Encoding untuk `gender`, `country`, `acquisition_channel`, `device_type`, `payment_method`
    6. **Drop Fitur** → `customer_id`, `signup_date`, `last_purchase_date` dihapus
    7. **SMOTE** → Oversampling pada data training untuk menangani imbalance kelas
    8. **Scaling** → StandardScaler (fit pada training, transform pada test)
    """)
