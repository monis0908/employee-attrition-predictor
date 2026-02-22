import streamlit as st
import pickle
import pandas as pd
from catboost import CatBoostClassifier

# 1. Load the trained model and unique values
try:
    with open('model_and_key_components.pkl', 'rb') as file:
        saved_components = pickle.load(file)
    model = saved_components['model']
    unique_values = saved_components['unique_values']
except Exception as e:
    st.error(f"Error loading model: {e}")

def main():
    st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
    st.title("Employee Attrition Prediction App")
    
    # --- USER INPUT SECTION ---
    st.subheader("Employee Data Input")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 70, 30)
        distance_from_home = st.slider("Distance From Home (km)", 1, 30, 10)
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 2)
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 2) # Added missing
        hourly_rate = st.slider("Hourly Rate", 30, 100, 65)

    with col2:
        monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
        num_companies_worked = st.slider("Num Companies Worked", 0, 10, 2)
        job_involvement = st.slider("Job Involvement", 1, 4, 2)
        job_level = st.slider("Job Level", 1, 5, 3)
        # Missing Categorical Inputs (Using selectboxes based on unique_values)
        department = st.selectbox("Department", unique_values.get('Department', ['Sales', 'Research & Development', 'Human Resources']))
        job_role = st.selectbox("Job Role", unique_values.get('JobRole', ['Sales Executive', 'Research Scientist', 'Laboratory Technician']))

    with col3:
        over_time = st.checkbox("Working Overtime?")
        percent_salary_hike = st.slider("Salary Hike (%)", 10, 25, 15)
        stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
        training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)
        work_life_balance = st.slider("Work Life Balance", 1, 4, 2)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 3)
        years_with_curr_manager = st.slider("Years With Current Manager", 0, 15, 3)

    # --- DATA PREPARATION ---
    data_dict = {
        'Age': [age],
        'DistanceFromHome': [distance_from_home],
        'EnvironmentSatisfaction': [environment_satisfaction],
        'RelationshipSatisfaction': [relationship_satisfaction], # Added
        'HourlyRate': [hourly_rate],
        'JobSatisfaction': [2], # Defaulting if missing from UI, or add a slider
        'MonthlyIncome': [monthly_income],
        'NumCompaniesWorked': [num_companies_worked],
        'JobInvolvement': [job_involvement],
        'JobLevel': [job_level],
        'Department': [department], # Added
        'JobRole': [job_role],       # Added
        'OverTime': [1 if over_time else 0], # Fixed: Changed to numeric 0/1
        'PercentSalaryHike': [percent_salary_hike],
        'StockOptionLevel': [stock_option_level],
        'TrainingTimesLastYear': [training_times_last_year],
        'WorkLifeBalance': [work_life_balance],
        'YearsSinceLastPromotion': [years_since_last_promotion],
        'YearsWithCurrManager': [years_with_curr_manager]
    }

    input_df = pd.DataFrame(data_dict)

    # --- THE SMART ALIGNMENT ---
    try:
        correct_order = model.feature_names_
        input_df = input_df[correct_order]
        
        # We need to make sure Categorical columns stay as strings for CatBoost
        cat_features = ['Department', 'JobRole'] 
        for col in cat_features:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)
                
    except Exception as e:
        st.warning(f"Feature Alignment Note: {e}")

    # --- PREDICTION SECTION ---
    st.divider()
    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[:, 1]

            if prediction[0] == 0:
                st.success("### Result: Employee is likely to STAY")
            else:
                st.error("### Result: Employee is at risk of LEAVING")
            
            st.metric(label="Risk Probability", value=f"{probability[0]:.2%}")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()