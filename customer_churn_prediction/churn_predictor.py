import joblib
import json
import pandas as pd

class ChurnPredictor:
    """
    Complete pipeline for churn prediction on new data
    """
    
    def __init__(self, model_path, preprocessor_path, config_path, fe_params_path):
        """Load saved model and configurations"""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        with open(fe_params_path, 'r') as f:
            self.fe_params = json.load(f)
        
        self.threshold = self.config['optimal_threshold']
        self.median_charges = self.fe_params['monthly_charges_median']
        self.weak_features = self.fe_params['weak_features_to_drop']
        
        print("✅ Churn Predictor loaded successfully!")
        print(f"   Model: {model_path}")
        print(f"   Threshold: {self.threshold}")
    
    def engineer_features(self, df):
        """Apply feature engineering to new data"""
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
        
        # Short tenure + high charges (use saved median)
        df_eng['ShortTenure_HighCharge'] = (
            (df['tenure'] < 12) & 
            (df['MonthlyCharges'] > self.median_charges)
        ).astype(int)
        
        return df_eng
    
    def preprocess(self, df):
        """Complete preprocessing pipeline"""
        # 1. Drop weak features
        df_processed = df.drop(columns=self.weak_features, errors='ignore')
        
        # 2. Apply feature engineering
        df_processed = self.engineer_features(df_processed)
        
        # 3. Apply trained preprocessor (encoding + scaling)
        X_processed = self.preprocessor.transform(df_processed)
        
        return X_processed
    
    def predict(self, df):
        """
        Predict churn for new customers
        
        Parameters:
        -----------
        df : DataFrame
            New customer data (same format as training data)
            
        Returns:
        --------
        predictions : dict
            Dictionary with predictions and probabilities
        """
        # Preprocess
        X_processed = self.preprocess(df)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        # Apply threshold
        predictions = (probabilities >= self.threshold).astype(int)
        
        # Create results dataframe
        results = pd.DataFrame({
            'Churn_Probability': probabilities,
            'Churn_Prediction': predictions,
            'Churn_Label': ['Yes' if p == 1 else 'No' for p in predictions],
            'Risk_Level': ['High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low' 
                          for prob in probabilities]
        })
        
        return results
    
    def predict_single(self, customer_data):
        """Predict for a single customer (as dictionary)"""
        df = pd.DataFrame([customer_data])
        return self.predict(df)


# Initialize the predictor
predictor = ChurnPredictor(
    model_path='churn_model_tuned.pkl',
    preprocessor_path='preprocessor_fe.pkl',
    config_path='model_config.json',
    fe_params_path='feature_engineering_params.json'
)