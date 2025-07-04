# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide" # Use wide layout for a better look
)

# --- Load Models and Columns ---
@st.cache_resource
def load_models():
    models = {
        "Gradient Boosting": joblib.load('gradient_boosting_model.pkl'),
        "Random Forest": joblib.load('random_forest_model.pkl'),
        "Ridge Regression": joblib.load('ridge_regression_model.pkl'),
        "Linear Regression": joblib.load('linear_regression_model.pkl')
    }
    return models

@st.cache_resource
def load_columns():
    return joblib.load('model_columns.pkl')

models = load_models()
model_columns = load_columns()

# --- Title and Introduction ---
st.title("🏥 Insurance Cost Predictor")
st.write("""
This interactive app predicts insurance costs using several regression models. 
The **Gradient Boosting** model is recommended as it performed the best during evaluation.
Adjust the features in the sidebar to generate a prediction.
""")

# --- Sidebar for User Input ---
st.sidebar.header("⚙️ User Input Features")

# Model selection
model_options = list(models.keys())
model_choice = st.sidebar.selectbox("Choose a Prediction Model", model_options)

# Input fields
age = st.sidebar.slider("Age", 18, 65, 30)
bmi = st.sidebar.slider("Body Mass Index (BMI)", 15.0, 55.0, 25.0, 0.1)
children = st.sidebar.slider("Number of Children", 0, 5, 0)
sex = st.sidebar.radio("Sex", ("Male", "Female"))
smoker = st.sidebar.radio("Smoker", ("Yes", "No"))
region = st.sidebar.selectbox("Region", ("Southwest", "Southeast", "Northwest", "Northeast"))

# --- Prepare Input for Prediction ---
input_data = {
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'sex_male': [1 if sex == 'Male' else 0],
    'smoker_yes': [1 if smoker == 'Yes' else 0],
    'region_northwest': [1 if region == 'Northwest' else 0],
    'region_southeast': [1 if region == 'Southeast' else 0],
    'region_southwest': [1 if region == 'Southwest' else 0],
}
input_data['obese_smoker'] = [1 if (bmi >= 30 and smoker == 'Yes') else 0]
input_df = pd.DataFrame(input_data).reindex(columns=model_columns, fill_value=0)

# --- Prediction Logic ---
chosen_model = models[model_choice]
prediction_log = chosen_model.predict(input_df)
predicted_cost = np.expm1(prediction_log)[0]

# --- Main Page Layout ---

# **CORRECTION HERE**: Define user_data before it is used in the columns
user_data = {
    'Model': model_choice,
    'Age': age,
    'BMI': f"{bmi:.1f}",
    'Children': children,
    'Sex': sex,
    'Smoker': smoker,
    'Region': region
}

col1, col2 = st.columns([1, 2])

# Column 1: User Selections and Data
with col1:
    st.subheader("📝 Your Selections")
    # Display the selections in a more organized way
    st.write(f"**Model:** `{user_data['Model']}`")
    st.write(f"**Age:** {user_data['Age']}")
    st.write(f"**BMI:** {user_data['BMI']}")
    st.write(f"**Children:** {user_data['Children']}")
    st.write(f"**Sex:** {user_data['Sex']}")
    st.write(f"**Smoker:** {user_data['Smoker']}")
    st.write(f"**Region:** {user_data['Region']}")

# Column 2: Prediction Result
with col2:
    st.subheader("✅ Prediction Result")
    st.metric(
        label=f"Estimated Cost ({model_choice})",
        value=f"${predicted_cost:,.2f}"
    )
    st.info("This prediction is based on the provided data and the selected model. It should be used for informational purposes only.", icon="ℹ️")

st.divider()

# --- Model Performance Section ---
with st.expander("📊 Click to see Model Performance Comparison"):
    st.write("""
    The models were evaluated on a test set (20% of the data). The **Gradient Boosting** model showed the highest R-squared score and the lowest error, making it the most reliable for this task.
    """)
    
    # Hardcode the results from your notebook
    results_data = {
        'Model': ['Gradient Boosting', 'Random Forest', 'Ridge Regression', 'Linear Regression'],
        'R-squared (R²)': [0.8676, 0.8472, 0.8215, 0.8215],
        'Mean Absolute Error (MAE)': [2045.68, 2092.97, 3674.08, 3698.86]
    }
    results_df = pd.DataFrame(results_data).sort_values(by='R-squared (R²)', ascending=False)
    
    # Create the comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R-squared plot
    sns.barplot(ax=axes[0], x='R-squared (R²)', y='Model', data=results_df, palette='Blues_d', orient='h')
    axes[0].set_title('R-squared (Higher is Better)')
    axes[0].set_xlim(0.80, 0.90)

    # MAE plot
    sns.barplot(ax=axes[1], x='Mean Absolute Error (MAE)', y='Model', data=results_df, palette='Reds_d', orient='h')
    axes[1].set_title('Mean Absolute Error (Lower is Better)')

    plt.tight_layout()
    st.pyplot(fig)