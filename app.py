import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =============================================================
# HEADER
# =============================================================
st.title("📊 Customer Churn Prediction")
st.markdown("**UAS Bengkel Koding Data Science** · Gabriella Jovanka Bustan · A11.2023.14861")
st.markdown("---")

# =============================================================
# HELPER: PREPROCESSING (sama persis dengan notebook)
# =============================================================
def preprocess(df_raw):
    df = df_raw.copy()

    # Drop duplikat
    df = df.drop_duplicates()

    # Feature Engineering: tenure_days
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
    df['tenure_days'] = (df['last_purchase_date'] - df['signup_date']).dt.days
    df['tenure_days'] = df['tenure_days'].fillna(df['tenure_days'].median())
    df = df.drop(['customer_id', 'signup_date', 'last_purchase_date'], axis=1, errors='ignore')

    # Imputasi missing value
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Outlier Capping (IQR)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('churn', errors='ignore')
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper, upper, np.where(df[col] < lower, lower, df[col]))

    # Label Encoding
    le_cols = ['city', 'coupon_code', 'subscription_type']
    for col in le_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # One-Hot Encoding
    ohe_cols = ['gender', 'country', 'acquisition_channel', 'device_type', 'payment_method']
    ohe_existing = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_existing, drop_first=False)

    return df


# =============================================================
# SIDEBAR — UPLOAD DATASET
# =============================================================
st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (.csv)",
    type=["csv"],
    help="Upload file sales_marketing_customer_dataset.csv"
)

if uploaded_file is None:
    st.info("👈 Upload dataset CSV di sidebar untuk memulai.")
    st.stop()

# =============================================================
# LOAD DATA
# =============================================================
df_raw = pd.read_csv(uploaded_file)
st.success(f"✅ Dataset berhasil dimuat: **{df_raw.shape[0]:,} baris**, **{df_raw.shape[1]} kolom**")

# =============================================================
# TABS
# =============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 EDA",
    "🤖 Modeling & Tuning",
    "📈 Evaluasi & Perbandingan",
    "🔮 Prediksi Data Baru"
])


# ─────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("1. Eksplorasi Data (EDA)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**5 Baris Pertama**")
        st.dataframe(df_raw.head(), use_container_width=True)
    with col2:
        st.markdown("**Statistik Deskriptif**")
        st.dataframe(df_raw.describe(), use_container_width=True)

    st.markdown("**Informasi Kolom**")
    info_df = pd.DataFrame({
        "Kolom": df_raw.columns,
        "Tipe Data": df_raw.dtypes.values,
        "Jumlah Non-Null": df_raw.notnull().sum().values,
        "Missing (%)": (df_raw.isnull().sum().values / len(df_raw) * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Distribusi Missing Value**")
        missing_pct = (df_raw.isnull().sum() / len(df_raw) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        if missing_pct.empty:
            st.success("Tidak ada missing value!")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            missing_pct.plot(kind='bar', color='salmon', ax=ax)
            ax.set_title('Persentase Missing Value per Kolom')
            ax.set_ylabel('Persentase (%)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col4:
        st.markdown("**Distribusi Target (Churn)**")
        if 'churn' in df_raw.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            churn_counts = df_raw['churn'].value_counts()
            bars = ax.bar(churn_counts.index.astype(str), churn_counts.values,
                          color=['#2ecc71', '#e74c3c'])
            ax.set_title('Distribusi Variabel Target (Churn)')
            ax.set_xlabel('Churn')
            ax.set_ylabel('Jumlah')
            total = churn_counts.sum()
            for bar, val in zip(bars, churn_counts.values):
                ax.annotate(f'{val:,}\n({val/total*100:.1f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("**Heatmap Korelasi Fitur Numerik**")
    numeric_df = df_raw.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='Blues', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Heatmap Korelasi Fitur Numerik')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────
# TAB 2 — MODELING & TUNING
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("2. Preprocessing & Hyperparameter Tuning")

    if st.button("🚀 Jalankan Training & Tuning", type="primary"):
        with st.spinner("Preprocessing data..."):
            df_prep = preprocess(df_raw)
            X = df_prep.drop('churn', axis=1)
            y = df_prep['churn']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            smote = SMOTE(random_state=42)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_sm)
            X_test_sc  = scaler.transform(X_test)

        st.success(f"✅ Preprocessing selesai. Fitur: **{X.shape[1]}** | "
                   f"Train: **{X_train_sc.shape[0]:,}** (setelah SMOTE) | "
                   f"Test: **{X_test_sc.shape[0]:,}**")

        # Feature Importance
        st.markdown("#### Feature Importance")
        with st.spinner("Menghitung feature importance..."):
            rf_temp = RandomForestClassifier(random_state=42)
            rf_temp.fit(X_train_sc, y_train_sm)
            importances = rf_temp.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # top 20
            feat_names = X.columns

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(indices)), importances[indices], color='teal')
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels(feat_names[indices], rotation=90, fontsize=8)
        ax.set_title("Top 20 Feature Importance (RF)")
        ax.set_ylabel("Importance")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # GridSearchCV Tuning
        st.markdown("#### Hyperparameter Tuning (GridSearchCV)")
        progress = st.progress(0, text="Tuning Model A: Logistic Regression...")

        model_a = LogisticRegression(max_iter=2000)
        params_a = {'C': [0.1, 1, 10]}
        grid_a = GridSearchCV(model_a, params_a, cv=3, scoring='f1', n_jobs=-1)
        grid_a.fit(X_train_sc, y_train_sm)
        best_a = grid_a.best_estimator_
        score_a = grid_a.best_score_
        progress.progress(33, text="Tuning Model B: Random Forest...")

        model_b = RandomForestClassifier(random_state=42)
        params_b = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10]}
        grid_b = GridSearchCV(model_b, params_b, cv=3, scoring='f1', n_jobs=-1)
        grid_b.fit(X_train_sc, y_train_sm)
        best_b = grid_b.best_estimator_
        score_b = grid_b.best_score_
        progress.progress(66, text="Tuning Model C: Voting Classifier...")

        model_c = VotingClassifier(estimators=[
            ('lr', LogisticRegression(max_iter=2000)),
            ('svm', SVC(probability=True)),
            ('knn', KNeighborsClassifier())
        ], voting='soft', weights=[2, 1, 1])
        params_c = {
            'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1]],
            'lr__C': [0.1, 1],
            'svm__C': [0.1, 1],
        }
        grid_c = GridSearchCV(model_c, params_c, cv=3, scoring='f1', n_jobs=-1)
        grid_c.fit(X_train_sc, y_train_sm)
        best_c = grid_c.best_estimator_
        score_c = grid_c.best_score_
        progress.progress(100, text="Tuning selesai!")

        # Simpan ke session state
        results_list = [
            ('Konvensional (LR)', score_a, best_a, grid_a.best_params_),
            ('Bagging (RF)',       score_b, best_b, grid_b.best_params_),
            ('Voting Classifier',  score_c, best_c, grid_c.best_params_),
        ]
        best_strategy = max(results_list, key=lambda x: x[1])

        st.session_state['models']        = [best_a, best_b, best_c]
        st.session_state['model_names']   = ['Konvensional (LR)', 'Bagging (RF)', 'Voting Classifier']
        st.session_state['best_model']    = best_strategy[2]
        st.session_state['best_name']     = best_strategy[0]
        st.session_state['best_params']   = best_strategy[3]
        st.session_state['scaler']        = scaler
        st.session_state['X_test_sc']     = X_test_sc
        st.session_state['y_test']        = y_test
        st.session_state['feature_cols']  = X.columns.tolist()

        # Tabel hasil tuning
        tuning_df = pd.DataFrame({
            "Model":            ['Konvensional (LR)', 'Bagging (RF)', 'Voting Classifier'],
            "CV F1-Score":      [f"{score_a:.4f}", f"{score_b:.4f}", f"{score_c:.4f}"],
            "Best Params":      [
                str(grid_a.best_params_),
                str(grid_b.best_params_),
                str(grid_c.best_params_)
            ]
        })
        st.dataframe(tuning_df, use_container_width=True)

        st.success(f"🏆 **Model Terbaik: {best_strategy[0]}** dengan CV F1 = {best_strategy[1]:.4f}")
        st.json(best_strategy[3])

        # Simpan file .pkl
        joblib.dump(best_strategy[2], '/tmp/best_churn_model.pkl')
        joblib.dump(scaler, '/tmp/scaler.pkl')

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            with open('/tmp/best_churn_model.pkl', 'rb') as f:
                st.download_button("⬇️ Download Model (.pkl)", f,
                                   file_name="best_churn_model.pkl")
        with col_dl2:
            with open('/tmp/scaler.pkl', 'rb') as f:
                st.download_button("⬇️ Download Scaler (.pkl)", f,
                                   file_name="scaler.pkl")
    else:
        st.info("Klik tombol di atas untuk memulai training dan hyperparameter tuning.")


# ─────────────────────────────────────────────────────────────
# TAB 3 — EVALUASI & PERBANDINGAN
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("3. Evaluasi & Perbandingan 3 Model")

    if 'models' not in st.session_state:
        st.warning("⚠️ Jalankan training di tab **Modeling & Tuning** terlebih dahulu.")
    else:
        models      = st.session_state['models']
        model_names = st.session_state['model_names']
        X_test_sc   = st.session_state['X_test_sc']
        y_test      = st.session_state['y_test']
        best_name   = st.session_state['best_name']

        results_table = []
        for name, model in zip(model_names, models):
            y_pred = model.predict(X_test_sc)
            results_table.append({
                "Model":     name,
                "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred), 4),
                "Recall":    round(recall_score(y_test, y_pred), 4),
                "F1-Score":  round(f1_score(y_test, y_pred), 4),
            })

        df_res = pd.DataFrame(results_table)
        best_row = df_res[df_res['Model'] == best_name].index[0]

        st.markdown("#### Tabel Perbandingan Metrik")

        def highlight_best(row):
            return ['background-color: #d4edda; font-weight: bold'
                    if row.name == best_row else '' for _ in row]

        st.dataframe(
            df_res.style.apply(highlight_best, axis=1).format(
                {c: "{:.4f}" for c in ['Accuracy','Precision','Recall','F1-Score']}
            ),
            use_container_width=True
        )
        st.caption(f"🟢 Baris hijau = model terbaik: **{best_name}**")

        # Bar chart perbandingan
        st.markdown("#### Visualisasi Metrik")
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(model_names))
        width = 0.2
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors  = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            vals = df_res[metric].values
            bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Perbandingan Metrik Evaluasi (Test Set)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Confusion Matrix ketiga model
        st.markdown("#### Confusion Matrix — 3 Model")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, (name, model) in enumerate(zip(model_names, models)):
            y_pred = model.predict(X_test_sc)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['Not Churn', 'Churn'],
                        yticklabels=['Not Churn', 'Churn'])
            axes[i].set_title(f'CM: {name}')
            axes[i].set_xlabel('Prediksi')
            axes[i].set_ylabel('Aktual')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────
# TAB 4 — PREDIKSI DATA BARU
# ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("4. Prediksi Customer Baru")

    if 'best_model' not in st.session_state:
        st.warning("⚠️ Jalankan training di tab **Modeling & Tuning** terlebih dahulu.")
    else:
        st.markdown("Masukkan data customer untuk diprediksi apakah akan **churn atau tidak**.")

        col1, col2, col3 = st.columns(3)
        with col1:
            age              = st.number_input("Usia (age)", min_value=18, max_value=100, value=35)
            annual_income    = st.number_input("Annual Income", min_value=0, value=50000, step=1000)
            purchase_amount  = st.number_input("Purchase Amount", min_value=0.0, value=500.0)
            num_purchases    = st.number_input("Num of Purchases", min_value=0, value=10)
            avg_order_val    = st.number_input("Avg Order Value", min_value=0.0, value=100.0)
        with col2:
            satisfaction     = st.slider("Satisfaction Score", 1, 5, 3)
            loyalty_score    = st.number_input("Loyalty Score", min_value=0.0, value=50.0)
            support_tickets  = st.number_input("Support Tickets", min_value=0, value=1)
            email_open_rate  = st.slider("Email Open Rate", 0.0, 1.0, 0.3)
            click_through    = st.slider("Click Through Rate", 0.0, 1.0, 0.1)
        with col3:
            promotion_resp   = st.slider("Promotion Response Rate", 0.0, 1.0, 0.2)
            social_media_eng = st.number_input("Social Media Engagement", min_value=0.0, value=5.0)
            website_visits   = st.number_input("Website Visits", min_value=0, value=20)
            tenure_days_val  = st.number_input("Tenure Days", min_value=0, value=365)
            gender           = st.selectbox("Gender", ["Male", "Female", "Other"])

        col4, col5 = st.columns(2)
        with col4:
            country          = st.selectbox("Country", ["USA", "UK", "Canada", "Australia", "Germany"])
            acquisition_ch   = st.selectbox("Acquisition Channel", ["Online", "Offline", "Referral", "Social Media"])
            device_type      = st.selectbox("Device Type", ["Desktop", "Mobile", "Tablet"])
        with col5:
            payment_method   = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Cash"])
            subscription     = st.selectbox("Subscription Type", ["Basic", "Premium", "Enterprise"])
            city             = st.text_input("City", "Jakarta")
            coupon_code      = st.text_input("Coupon Code", "NONE")

        if st.button("🔮 Prediksi Sekarang", type="primary"):
            feature_cols = st.session_state['feature_cols']
            scaler_obj   = st.session_state['scaler']
            best_model   = st.session_state['best_model']
            best_name    = st.session_state['best_name']

            # Buat dataframe input dengan kolom numerik terlebih dulu
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

            input_df = pd.DataFrame([raw_input])

            # Label Encoding kolom yang sama seperti training
            le_cols_pred = ['city', 'coupon_code', 'subscription_type']
            for col in le_cols_pred:
                if col in input_df.columns:
                    input_df[col] = LabelEncoder().fit_transform(input_df[col].astype(str))

            # OHE
            ohe_cols_pred = ['gender', 'country', 'acquisition_channel', 'device_type', 'payment_method']
            ohe_existing  = [c for c in ohe_cols_pred if c in input_df.columns]
            input_df = pd.get_dummies(input_df, columns=ohe_existing, drop_first=False)

            # Align kolom ke feature_cols
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_cols]

            # Scale
            input_scaled = scaler_obj.transform(input_df)

            # Prediksi
            pred = best_model.predict(input_scaled)[0]
            prob = None
            if hasattr(best_model, "predict_proba"):
                prob = best_model.predict_proba(input_scaled)[0]

            st.markdown("---")
            if pred == 1:
                st.error(f"⚠️ **Prediksi: CHURN** — Customer ini diprediksi akan berhenti berlangganan.")
            else:
                st.success(f"✅ **Prediksi: TIDAK CHURN** — Customer ini diprediksi akan tetap berlangganan.")

            if prob is not None:
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Probabilitas Tidak Churn", f"{prob[0]*100:.1f}%")
                with col_p2:
                    st.metric("Probabilitas Churn", f"{prob[1]*100:.1f}%")

            st.caption(f"Model yang digunakan: **{best_name}**")

# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.caption("UAS Bengkel Koding Data Science · Universitas Dian Nuswantoro · 2024")
