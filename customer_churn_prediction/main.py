"""
FastAPI Backend for Customer Churn Prediction
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
import numpy as np
from typing import Optional


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability using ML model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerData(BaseModel):
    """Input schema for customer data"""
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., ge=0, description="Number of months")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges amount")
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction"""
    churn_probability: float
    churn_prediction: str
    risk_level: str
    confidence: float
    recommendations: list


class ChurnPredictor:
    """Churn prediction pipeline"""
    
    def __init__(self):
        try:
            self.model = joblib.load('churn_model_tuned.pkl')
            self.preprocessor = joblib.load('preprocessor_fe.pkl')
            
            with open('model_config.json', 'r') as f:
                self.config = json.load(f)
            
            with open('feature_engineering_params.json', 'r') as f:
                self.fe_params = json.load(f)
            
            self.threshold = self.config['optimal_threshold']
            self.median_charges = self.fe_params['monthly_charges_median']
            self.weak_features = self.fe_params['weak_features_to_drop']
            
            print("Model loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            raise
    
    def engineer_features(self, df):
        """Apply feature engineering"""
        df_eng = df.copy()
        
        # High-Risk Score
        df_eng['HighRisk_Score'] = (
            (df['Contract'] == 'Month-to-month').astype(int) * 5 +
            (df['OnlineSecurity'] == 'No').astype(int) * 4 +
            (df['TechSupport'] == 'No').astype(int) * 4 +
            (df['OnlineBackup'] == 'No').astype(int) * 3 +
            (df['PaymentMethod'] == 'Electronic check').astype(int) * 3 +
            (df['PaperlessBilling'] == 'Yes').astype(int) * 2
        )
        
        # No Support Services
        df_eng['NoSupport'] = (
            ((df['OnlineSecurity'] == 'No') | (df['OnlineSecurity'] == 'No internet service')) &
            ((df['TechSupport'] == 'No') | (df['TechSupport'] == 'No internet service'))
        ).astype(int)
        
        # Vulnerable Customer
        df_eng['Vulnerable'] = (
            (df['Contract'] == 'Month-to-month') & 
            (df['PaymentMethod'] == 'Electronic check')
        ).astype(int)
        
        # Tenure to Charge Ratio
        df_eng['Tenure_Charge_Ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1)
        
        # Short tenure + high charges
        df_eng['ShortTenure_HighCharge'] = (
            (df['tenure'] < 12) & 
            (df['MonthlyCharges'] > self.median_charges)
        ).astype(int)
        
        return df_eng
    
    def preprocess(self, df):
        """Complete preprocessing pipeline"""
        # Drop weak features
        df_processed = df.drop(columns=self.weak_features, errors='ignore')
        
        # Apply feature engineering
        df_processed = self.engineer_features(df_processed)
        
        # Apply trained preprocessor
        X_processed = self.preprocessor.transform(df_processed)
        
        return X_processed
    
    def get_recommendations(self, customer_data, probability):
        """Generate personalized recommendations"""
        recommendations = []
        
        if probability > 0.7:
            recommendations.append("🚨 URGENT: High churn risk - immediate action required!")
        
        # Contract-based recommendations
        if customer_data['Contract'] == 'Month-to-month':
            recommendations.append("📝 Offer long-term contract with discount (1 or 2 years)")
        
        # Service-based recommendations
        if customer_data['OnlineSecurity'] == 'No' and customer_data['InternetService'] != 'No':
            recommendations.append("🔒 Promote Online Security add-on service")
        
        if customer_data['TechSupport'] == 'No' and customer_data['InternetService'] != 'No':
            recommendations.append("🛠️ Offer Tech Support service")
        
        # Payment method recommendation
        if customer_data['PaymentMethod'] == 'Electronic check':
            recommendations.append("💳 Encourage automatic payment method (reduces churn by 30%)")
        
        # Tenure-based
        if customer_data['tenure'] < 12:
            recommendations.append("🎁 Provide loyalty rewards for first-year customers")
        
        # Price-based
        if customer_data['MonthlyCharges'] > 70 and customer_data['Contract'] == 'Month-to-month':
            recommendations.append("💰 Consider offering promotional discount or bundle services")
        
        if not recommendations:
            recommendations.append("Customer appears stable - maintain service quality")
        
        return recommendations
    
    def predict(self, customer_data: dict):
        """Make prediction for a customer"""
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Preprocess
        X_processed = self.preprocess(df)
        
        # Get probability
        probability = float(self.model.predict_proba(X_processed)[0, 1])
        
        # Apply threshold
        prediction = 1 if probability >= self.threshold else 0
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Get recommendations
        recommendations = self.get_recommendations(customer_data, probability)
        
        return {
            "churn_probability": round(probability, 4),
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "risk_level": risk_level,
            "confidence": round(abs(probability - 0.5) * 2, 4),  # 0 to 1 scale
            "recommendations": recommendations
        }


# Initialize predictor
try:
    predictor = ChurnPredictor()
except Exception as e:
    print(f"Failed to initialize predictor: {e}")
    predictor = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "status": "active",
        "model_loaded": predictor is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_threshold": predictor.threshold,
        "model_config": predictor.config.get("performance", {})
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Predict churn probability for a customer
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic model to dict
        customer_dict = customer.dict()
        
        # Make prediction
        result = predictor.predict(customer_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(customers: list[CustomerData]):
    """
    Predict churn for multiple customers
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for customer in customers:
            customer_dict = customer.dict()
            result = predictor.predict(customer_dict)
            results.append(result)
        
        return {
            "total_customers": len(results),
            "high_risk_count": sum(1 for r in results if r["risk_level"] == "High"),
            "predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)