🏥Regression - Insurance Cost Prediction
A Streamlit-based interactive web application that predicts individual medical insurance charges using various regression models. This project demonstrates the end-to-end data science workflow — from EDA and model training to real-time deployment.

🚀 Project Overview
This project tackles the challenge of predicting insurance costs using personal demographics such as age, sex, BMI, smoking status, and region. We built and evaluated multiple regression models and deployed the best-performing one via an interactive Streamlit interface.

📊 Dataset
Source: insurance.csv

Records: 1,338 individuals

Features:

age: Age of the person

sex: Gender

bmi: Body Mass Index

children: Number of dependents

smoker: Smoking status

region: Residential region (US)

charges: Medical cost billed (Target variable)

🧪 Technologies Used
Python 3.11

Pandas, NumPy

Matplotlib, Seaborn – for EDA and visualization

Scikit-learn – for model training and evaluation

Streamlit – for deployment

Joblib – for model persistence

🔍 EDA Highlights
Class imbalance in smokers vs. non-smokers

Strong correlation between smoking and insurance charges

Added custom feature: obese_smoker (BMI ≥ 30 & smoker)

Visualized distributions and relationships with pair plots and bar charts

🧠 Models Trained
Linear Regression

Ridge Regression

Random Forest Regressor

Gradient Boosting Regressor ✅ (Best Performer)

📈 Model Evaluation (Test Set)
Model	R-squared (R²)	Mean Absolute Error (MAE)
Gradient Boosting	0.8676	$2045.68
Random Forest	0.8472	$2092.97
Ridge Regression	0.8215	$3674.08
Linear Regression	0.8215	$3698.86

✅ Gradient Boosting outperformed all others — best R² and lowest MAE.

🎮 App Demo
The web app allows users to:

Select a model (e.g., Gradient Boosting)

Input personal details (age, BMI, etc.)

Get real-time cost predictions

View model performance comparison charts

Main script: app.py
Uses saved models: *.pkl files

💻 How to Run Locally
bash
Copy
Edit
# 1. Clone the repository
git clone https://github.com/yourusername/insurance-cost-prediction.git
cd insurance-cost-prediction

# 2. Install required packages
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
Ensure the following model files exist in the same directory:

linear_regression_model.pkl

ridge_regression_model.pkl

random_forest_model.pkl

gradient_boosting_model.pkl

model_columns.pkl

🚧 Project Challenges & Limitations
While the project showcases the power of regression models in predicting insurance costs, several real-world challenges and constraints were encountered:

🔍 Limited Dataset Size
After cleaning and removing outliers, the dataset shrank to just 1,116 records — increasing the risk of overfitting and reducing the model’s ability to generalize.

🧍‍♂️ User Input Accuracy
Predictions rely on user-submitted inputs like BMI and smoking status. If this information is incorrect or misreported, the output may be misleading.

🌍 Restricted Feature Scope
The model only considers a handful of variables (age, BMI, smoking, etc.). Real-world healthcare costs depend on many more factors (e.g., medical history, income, lifestyle), which are not included.

📈 Scalability Concerns
The model may not perform well when applied to larger or more diverse populations, especially outside the original dataset's demographics (e.g., international data).

🧪 Model Bias Potential
With a skewed smoker distribution and limited regional diversity, there's a risk of model bias in underrepresented categories.

