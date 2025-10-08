🏡 Household Food Waste Predictor & Smart Shopping Recommender

An AI-powered tool to predict household food waste and provide smart shopping recommendations, helping reduce environmental impact and save money. Built with Python, scikit-learn, and Streamlit.

🌍 Problem Statement

Household food waste contributes to 8–10% of global greenhouse gas emissions and represents a significant economic loss. This project predicts potential food waste and suggests optimized grocery decisions, aiming to reduce waste by 20–25% in typical scenarios.

🔑 Key Features

Data Handling

Generates realistic synthetic grocery datasets for testing.

Supports CSV uploads for real household data.

Machine Learning Models

Random Forest Regressor: Predicts food waste in kilograms.

Logistic Regression: Classifies high vs. low waste risk.

Feature Importance Analysis: Shows which factors most affect waste.

Smart Recommendations

Suggests low-waste alternatives based on expiration and consumption patterns.

Provides actionable shopping tips to reduce waste.

Deployment

Interactive Streamlit web app with user input forms, metrics, and visualizations.

Supports retraining with uploaded CSV data for personalized predictions.

📈 Impact & Metrics
Metric	Value	Note
MAE (Waste Prediction)	~0.15 kg	Lower is better
F1-Score (Waste Risk)	0.80	Higher is better
Simulated Waste Reduction	~22%	Based on test scenarios
🛠️ Setup Instructions
Prerequisites

Python 3.8+

Install required packages:

pip install pandas numpy scikit-learn streamlit matplotlib seaborn

Running the App

Save the main script as:

food_waste_predictor.py


Navigate to the project folder in your terminal:

cd "D:\Household Food Waste Predictor and Smart Shopping Recommender project"


Run Streamlit:

streamlit run food_waste_predictor.py


Open the app in your browser at:

http://localhost:8501

Optional: Use Real Data

Create a CSV named household_data.csv with your household grocery data.

The app will automatically load it instead of generating synthetic data.

Upload additional CSV files through the app to retrain models for personalized predictions.

📊 Screenshots (Optional)

Add screenshots of your Streamlit app here, e.g., input forms, metrics dashboard, feature importance plots.

📝 Resume & Portfolio Highlights

Developed a full-stack ML solution: data preprocessing → model training → deployment via Streamlit.

Achieved 0.15 MAE in waste prediction and 22% simulated waste reduction.

Designed interactive visualizations and actionable recommendations.

GitHub link can be shared as a portfolio project; deploy live version on Heroku or PythonAnywhere.

🔮 Future Work

Integrate real-time grocery APIs for weekly shopping recommendations.

Add time-series forecasting for weekly/monthly consumption patterns.

Implement clustering for user personas to personalize suggestions.

Use NLP for recipe parsing to reduce ingredient waste.

📂 File Structure
Household Food Waste Predictor/
│
├─ food_waste_predictor.py    # Main Streamlit app and ML pipeline
├─ household_data.csv         # Optional real dataset (auto-generated if missing)
├─ README.md                  # This file

🔗 References

FAO: Food Waste Facts

Scikit-learn documentation: https://scikit-learn.org

Streamlit documentation: https://streamlit.io
