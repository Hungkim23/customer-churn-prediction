import streamlit as st
import pandas as pd
import os
import joblib

# Paths
DATA_PATH = os.path.join('data', 'telco_customer_churn.csv')
METRICS_PATH = os.path.join('models', 'model_metrics.csv')
MODEL_PATH = os.path.join('models', 'best_model.pkl')
OUTPUT_DIR = 'outputs'

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

st.title("📉 Telco Customer Churn Prediction Dashboard")
st.markdown("Predict customer churn using Machine Learning models. Explore the dataset, view EDA, compare model performances, and predict churn for new customers.")

# Helper to load data
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

df = load_dataset()

if df is None:
    st.error(f"Dataset not found at `{DATA_PATH}`. Please ensure the data is in the correct directory.")
    st.stop()

# Clean data copy for display
df_display = df.copy()
df_display['TotalCharges'] = pd.to_numeric(df_display['TotalCharges'], errors='coerce')

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "EDA Visualization", "Model Comparison", "Predict New Customer"])

# --- TAB 1: Dataset Overview ---
with tab1:
    st.header("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df_display.shape[0])
    with col2:
        st.metric("Total Columns", df_display.shape[1])
        
    st.subheader("First 5 Rows")
    st.dataframe(df_display.head())
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Churn Ratio")
        churn_counts = df_display['Churn'].value_counts()
        st.bar_chart(churn_counts)
    with col4:
        st.subheader("Missing Values")
        missing_values = df_display.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if missing_values.empty:
            st.success("No missing values found!")
        else:
            st.write(missing_values)

# --- TAB 2: EDA Visualization ---
with tab2:
    st.header("Exploratory Data Analysis")
    st.markdown("Here are some key visualizations generated from the dataset.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        img_path = os.path.join(OUTPUT_DIR, 'churn_distribution.png')
        if os.path.exists(img_path):
            st.image(img_path, caption="Churn Distribution")
            st.markdown("**Insight:** Visualizes the proportion of customers who churned vs stayed.")
        else:
            st.warning("Churn distribution image not found.")
            
        img_path = os.path.join(OUTPUT_DIR, 'contract_churn.png')
        if os.path.exists(img_path):
            st.image(img_path, caption="Churn by Contract Type")
            st.markdown("**Insight:** Month-to-month contracts tend to have much higher churn rates compared to one or two-year contracts.")
            
    with col2:
        img_path = os.path.join(OUTPUT_DIR, 'monthly_charges_churn.png')
        if os.path.exists(img_path):
            st.image(img_path, caption="Monthly Charges Distribution by Churn")
            st.markdown("**Insight:** Higher monthly charges often correlate with a higher likelihood of churn.")
            
        img_path = os.path.join(OUTPUT_DIR, 'tenure_churn.png')
        if os.path.exists(img_path):
            st.image(img_path, caption="Tenure Distribution by Churn")
            st.markdown("**Insight:** Customers with shorter tenure (new customers) are more likely to churn.")

# --- TAB 3: Model Comparison ---
with tab3:
    st.header("Model Performance Comparison")
    
    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics_df, use_container_width=True)
        
        best_model_row = metrics_df.loc[metrics_df['F1-score'].idxmax()]
        st.success(f"**Best Model:** {best_model_row['Model']} (F1-score: {best_model_row['F1-score']:.4f})")
        
        st.subheader("Confusion Matrix & Feature Importance")
        col1, col2 = st.columns(2)
        with col1:
            cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix of the Best Model")
        with col2:
            fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
            if os.path.exists(fi_path):
                st.image(fi_path, caption="Feature Importances")
    else:
        st.warning("Model metrics not found. Please train the models first.")

# --- TAB 4: Predict New Customer ---
with tab4:
    st.header("Predict Churn for a New Customer")
    st.markdown("Fill out the form below to get a prediction on whether a customer will churn.")
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Demographics")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                
                st.subheader("Account")
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                
            with col2:
                st.subheader("Services")
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                online_bkp = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                
            with col3:
                st.subheader("Add-ons")
                device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_mov = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                
                st.subheader("Charges")
                monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
                total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

            submitted = st.form_submit_button("Predict")
            
        if submitted:
            # Create a dataframe from input
            input_data = pd.DataFrame({
                'gender': [gender],
                'SeniorCitizen': [senior],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone],
                'MultipleLines': [multiple],
                'InternetService': [internet],
                'OnlineSecurity': [online_sec],
                'OnlineBackup': [online_bkp],
                'DeviceProtection': [device_prot],
                'TechSupport': [tech_sup],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_mov],
                'Contract': [contract],
                'PaperlessBilling': [paperless],
                'PaymentMethod': [payment],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
            })
            
            # Prediction
            prediction = model.predict(input_data)[0]
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(input_data)[0][1]
            else:
                prob = 1.0 if prediction == 1 else 0.0
                
            st.markdown("---")
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error("🚨 The customer is **LIKELY TO CHURN**.")
            else:
                st.success("✅ The customer is **NOT LIKELY TO CHURN**.")
                
            st.metric("Churn Probability", f"{prob*100:.2f}%")
            
            # Risk Level & Recommendations
            if prob < 0.3:
                st.info("**Risk Level:** Low Risk")
                st.markdown("**Action:** Khách hàng có khả năng tiếp tục sử dụng dịch vụ.")
            elif 0.3 <= prob < 0.6:
                st.warning("**Risk Level:** Medium Risk")
                st.markdown("**Action:** Khách hàng có rủi ro trung bình. Nên theo dõi thêm hành vi sử dụng dịch vụ.")
            else:
                st.error("**Risk Level:** High Risk")
                st.markdown("**Action:** Khách hàng có nguy cơ rời bỏ cao. Nên cân nhắc ưu đãi giữ chân, giảm phí tháng đầu, hoặc chuyển sang hợp đồng dài hạn.")
    else:
        st.warning("Model not found. Please train the models first.")
