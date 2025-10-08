import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Data Generation and Loading
@st.cache_data
def load_data(use_synthetic=True):
    """Load or generate household grocery data."""
    try:
        if use_synthetic or not os.path.exists('household_data.csv'):
            logger.info("Generating synthetic data...")
            np.random.seed(42)
            n_samples = 1000
            data = {
                'item': np.random.choice(['milk', 'bread', 'veggies', 'meat', 'fruit', 'canned'], n_samples),
                'quantity_kg': np.random.uniform(0.5, 5.0, n_samples),
                'exp_days': np.random.randint(1, 30, n_samples),
                'family_size': np.random.randint(1, 8, n_samples),
                'consumption_rate': np.random.uniform(0.1, 1.0, n_samples),  # kg/day
                'wasted_kg': np.zeros(n_samples)
            }
            df = pd.DataFrame(data)
            # Simulate waste: high if short expiration or low consumption
            df['wasted_kg'] = np.where(
                (df['exp_days'] < 7) | (df['quantity_kg'] / df['consumption_rate'] > df['exp_days']),
                df['quantity_kg'] * np.random.uniform(0.2, 0.8, n_samples),
                df['quantity_kg'] * np.random.uniform(0.0, 0.2, n_samples)
            )
            df['waste_risk'] = (df['wasted_kg'] > 0.5).astype(int)  # Binary classification
            df.to_csv('household_data.csv', index=False)
        else:
            logger.info("Loading data from CSV...")
            df = pd.read_csv('household_data.csv')
            if not all(col in df for col in ['item', 'quantity_kg', 'exp_days', 'family_size', 'consumption_rate', 'wasted_kg']):
                raise ValueError("CSV missing required columns.")
            df['waste_risk'] = (df['wasted_kg'] > 0.5).astype(int)

        # Encode categorical variables
        le = LabelEncoder()
        df['item_encoded'] = le.fit_transform(df['item'])
        return df, le
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        st.error(f"Failed to load data: {e}")
        return None, None

# Step 2: Model Training
@st.cache_data
def train_models(df):
    """Train regression and classification models."""
    try:
        features = ['item_encoded', 'quantity_kg', 'exp_days', 'family_size', 'consumption_rate']
        X = df[features]
        y_reg = df['wasted_kg']
        y_clf = df['waste_risk']
        
        # Split data
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)
        
        # Regression: Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train_reg)
        y_pred_reg = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        
        # Classification: Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train_clf)
        y_pred_clf = lr_model.predict(X_test)
        f1 = f1_score(y_test_clf, y_pred_clf)
        
        # Feature importance plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=rf_model.feature_importances_, y=features, ax=ax)
        ax.set_title('Feature Importance (Random Forest)')
        plt.tight_layout()
        
        # Simulated waste reduction
        baseline_waste = y_test_reg.mean()
        optimized_waste = y_pred_reg[y_pred_clf == 0].mean() if sum(y_pred_clf == 0) > 0 else baseline_waste
        reduction_pct = ((baseline_waste - optimized_waste) / baseline_waste * 100) if baseline_waste > 0 else 0
        
        return rf_model, lr_model, mae, f1, fig, reduction_pct
    except Exception as e:
        logger.error(f"Model training error: {e}")
        st.error(f"Failed to train models: {e}")
        return None, None, None, None, None, None

# Step 3: Prediction and Recommendation
def predict_and_recommend(rf_model, lr_model, le, inputs):
    """Predict waste and suggest low-waste alternatives."""
    try:
        input_df = pd.DataFrame([inputs])
        input_df['item_encoded'] = le.transform([inputs['item']])
        features = ['item_encoded', 'quantity_kg', 'exp_days', 'family_size', 'consumption_rate']
        
        # Predict waste (kg) and risk
        waste_pred = rf_model.predict(input_df[features])[0]
        risk_pred = lr_model.predict(input_df[features])[0]
        
        # Recommendation logic
        reco = []
        if risk_pred == 1 or waste_pred > 0.5:
            reco.append("High waste risk detected!")
            if inputs['exp_days'] < 7:
                reco.append("Choose items with longer expiration (e.g., canned goods or frozen veggies).")
            if inputs['quantity_kg'] / inputs['consumption_rate'] > inputs['exp_days']:
                reco.append(f"Reduce quantity to ~{inputs['exp_days'] * inputs['consumption_rate']:.1f} kg.")
        else:
            reco.append("Low waste risk. Good choice!")
        
        return waste_pred, risk_pred, reco
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error(f"Prediction failed: {e}")
        return None, None, ["Error in prediction."]

# Step 4: Streamlit App
def main():
    st.set_page_config(page_title="Food Waste Predictor", layout="wide")
    st.title("🏡 Household Food Waste Predictor")
    st.markdown("""
    Reduce food waste with AI! Enter your grocery details to predict waste risk and get smart shopping tips.
    Built with scikit-learn and Streamlit. [View source on GitHub](#).
    """)
    
    # Load data
    df, le = load_data(use_synthetic=True)
    if df is None or le is None:
        return
    
    # Train models
    rf_model, lr_model, mae, f1, fig, reduction_pct = train_models(df)
    if rf_model is None:
        return
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Waste kg)", f"{mae:.2f}", "Lower is better")
    col2.metric("F1-Score (Risk)", f"{f1:.2f}", "Higher is better")
    col3.metric("Simulated Waste Reduction", f"{reduction_pct:.1f}%", "Impact")
    
    # Feature importance
    st.subheader("Feature Importance")
    st.pyplot(fig)
    
    # User input form
    st.subheader("Predict Your Waste")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            item = st.selectbox("Item", ['milk', 'bread', 'veggies', 'meat', 'fruit', 'canned'])
            quantity_kg = st.number_input("Quantity (kg)", min_value=0.1, max_value=10.0, value=1.0)
            exp_days = st.number_input("Expiration Days", min_value=1, max_value=60, value=7)
        with col2:
            family_size = st.number_input("Family Size", min_value=1, max_value=10, value=1)
            consumption_rate = st.number_input("Consumption Rate (kg/day)", min_value=0.01, max_value=2.0, value=0.5)
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            try:
                inputs = {
                    'item': item,
                    'quantity_kg': quantity_kg,
                    'exp_days': exp_days,
                    'family_size': family_size,
                    'consumption_rate': consumption_rate
                }
                waste_pred, risk_pred, reco = predict_and_recommend(rf_model, lr_model, le, inputs)
                if waste_pred is not None:
                    st.subheader("Results")
                    st.metric("Predicted Waste (kg)", f"{waste_pred:.2f}")
                    st.metric("Waste Risk", "High" if risk_pred == 1 else "Low")
                    st.write("**Recommendations**:")
                    for r in reco:
                        st.write(f"- {r}")
            except Exception as e:
                logger.error(f"Form submission error: {e}")
                st.error("Invalid input. Please check your values.")
    
    # Upload CSV option
    st.subheader("Upload Your Data (Optional)")
    uploaded_file = st.file_uploader("Upload grocery data (CSV)", type="csv")
    if uploaded_file:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(df_uploaded.head())
            # Re-train with uploaded data
            df_uploaded, le_uploaded = load_data(use_synthetic=False)
            rf_model, lr_model, mae, f1, fig, reduction_pct = train_models(df_uploaded)
            st.success("Models retrained with uploaded data!")
            st.pyplot(fig)
        except Exception as e:
            logger.error(f"File upload error: {e}")
            st.error(f"Failed to process file: {e}")

if __name__ == "__main__":
    main()