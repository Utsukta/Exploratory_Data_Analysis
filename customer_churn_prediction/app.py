"""
Streamlit Frontend for Customer Churn Prediction
Run with: streamlit run app.py
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_churn(customer_data):
    """Call API to predict churn"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"


def create_gauge_chart(probability):
    """Create a gauge chart for churn probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#e8f5e9'},
                {'range': [40, 70], 'color': '#fff3e0'},
                {'range': [70, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 83.9
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">📊 Customer Churn Prediction System</p>', unsafe_allow_html=True)
    
    # Check API health
    api_status = check_api_health()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/customer.png", width=100)
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", "ℹ️ About"]
        )
        
        st.divider()
        
        # API Status
        if api_status:
            st.success("API Connected")
        else:
            st.error("API Offline")
            st.info("Start API with: `uvicorn main:app --reload`")
    
    # Home Page
    if page == "🏠 Home":
        st.header("Welcome to Churn Prediction System")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Model Accuracy",
                value="83.9%",
                delta="Recall Rate"
            )
        
        with col2:
            st.metric(
                label="Missed Churners",
                value="60",
                delta="-42% from baseline",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="ROC-AUC Score",
                value="0.86",
                delta="Excellent"
            )
        
        st.divider()
        
        st.subheader("📈 Model Performance")
        
        # Create performance comparison chart
        performance_data = pd.DataFrame({
            'Model': ['Baseline', 'Feature Engineering', 'Final Tuned'],
            'Missed Churners': [104, 102, 60],
            'Recall (%)': [72.1, 72.7, 83.9]
        })
        
        fig = px.bar(
            performance_data,
            x='Model',
            y='Missed Churners',
            title='Model Evolution - Missed Churners Reduction',
            color='Missed Churners',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Use the sidebar to navigate to Single Prediction or Batch Prediction")
    
    # Single Prediction Page
    elif page == "🔮 Single Prediction":
        st.header("Single Customer Churn Prediction")
        
        if not api_status:
            st.error("⚠️ API is not running. Please start the FastAPI server.")
            st.code("uvicorn main:app --reload")
            return
        
        st.write("Enter customer details to predict churn probability:")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📋 Personal Info")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            
            with col2:
                st.subheader("📞 Services")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            
            with col3:
                st.subheader("💼 Account Info")
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ])
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
            
            submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
        
        if submitted:
            # Prepare customer data
            customer_data = {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges
            }
            
            # Make prediction
            with st.spinner("Analyzing customer data..."):
                result, error = predict_churn(customer_data)
            
            if error:
                st.error(f"Prediction failed: {error}")
            else:
                st.success("Prediction Complete!")
                
                # Display results
                st.divider()
                
                # Gauge chart
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    fig = create_gauge_chart(result['churn_probability'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Risk level card
                    risk_class = f"{result['risk_level'].lower()}-risk"
                    
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h2>Risk Level: {result['risk_level']}</h2>
                        <h3>Churn Prediction: {result['churn_prediction']}</h3>
                        <p><strong>Probability:</strong> {result['churn_probability']*100:.2f}%</p>
                        <p><strong>Confidence:</strong> {result['confidence']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendations
                st.divider()
                st.subheader("💡 Recommended Actions")
                
                for i, rec in enumerate(result['recommendations'], 1):
                    st.write(f"{i}. {rec}")
                
                # Export option
                st.divider()
                if st.button("📥 Export Results as JSON"):
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(result, indent=2),
                        file_name="churn_prediction.json",
                        mime="application/json"
                    )
    
    # Batch Prediction Page
    elif page == "📊 Batch Prediction":
        st.header("Batch Customer Churn Prediction")
        
        if not api_status:
            st.error("⚠️ API is not running. Please start the FastAPI server.")
            return
        
        st.write("Upload a CSV file with customer data for batch prediction.")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df.head(10))
            
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner(f"Processing {len(df)} customers..."):
                    # Convert to list of dicts
                    customers = df.to_dict('records')
                    
                    try:
                        response = requests.post(
                            f"{API_URL}/predict/batch",
                            json=customers,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            st.success(f"✅ Processed {results['total_customers']} customers!")
                            
                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Customers", results['total_customers'])
                            
                            with col2:
                                st.metric("High Risk", results['high_risk_count'])
                            
                            with col3:
                                churn_count = sum(1 for p in results['predictions'] if p['churn_prediction'] == 'Yes')
                                st.metric("Predicted Churn", churn_count)
                            
                            # Create results DataFrame
                            predictions_df = pd.DataFrame(results['predictions'])
                            results_df = pd.concat([df, predictions_df], axis=1)
                            
                            st.subheader("📊 Results")
                            st.dataframe(results_df)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
                            
                        else:
                            st.error(f"❌ Batch prediction failed: {response.status_code}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # About Page
    elif page == "ℹ️ About":
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 Customer Churn Prediction System
        
        This system uses machine learning to predict customer churn probability and provide actionable insights.
        
        #### 📊 Model Performance
        - **Accuracy:** 83.9% recall rate
        - **Missed Churners:** Only 60 out of 373 (83.9% caught)
        - **ROC-AUC Score:** 0.86
        - **Threshold:** 0.32 (optimized for business impact)
        
        #### 🔬 Model Features
        - **Algorithm:** Gradient Boosting Classifier
        - **Feature Engineering:** 5 custom features based on EDA
        - **Data Balancing:** SMOTE oversampling
        - **Hyperparameter Tuning:** GridSearchCV optimized
        
        #### 📈 Key Predictors (from EDA)
        1. **Contract Type** (39.9% impact)
        2. **Online Security** (34.4% impact)
        3. **Tech Support** (34.3% impact)
        4. **Payment Method** (30.1% impact)
        5. **Internet Service** (34.5% impact)
        
        #### 🛠️ Technology Stack
        - **Backend:** FastAPI
        - **Frontend:** Streamlit
        - **ML:** Scikit-learn, XGBoost
        - **Data:** Pandas, NumPy
        
        #### 👨‍💻 Developer
        Built as part of a Customer Churn Analysis project
        
        ---
        
        ### How to Use
        
        1. **Single Prediction:** Enter individual customer details for instant prediction
        2. **Batch Prediction:** Upload a CSV file to predict for multiple customers
        3. **Recommendations:** Get personalized retention strategies for each customer
        
        ### Getting Started
        
        ```bash
        # Start the FastAPI backend
        uvicorn main:app --reload
        
        # Run Streamlit frontend (in another terminal)
        streamlit run app.py
        ```
        
        ### Support
        For issues or questions, please contact your system administrator.
        """)


if __name__ == "__main__":
    main()