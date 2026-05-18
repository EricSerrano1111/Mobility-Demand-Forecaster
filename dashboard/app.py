import streamlit as st
import requests
from datetime import date

# Page Configuration
st.set_page_config(page_title="Mobility Dispatch Predictor", page_icon="🚕", layout="centered")

st.title("Enterprise Fleet Demand Forecaster")
st.markdown("Enter the historical parameters below to generate a vehicle forecast from the live API.")

# Define the Cloud Run API Endpoint
API_URL = "https://mobility-demand-forecaster-994588599194.us-central1.run.app/predict"

# Create UI Input Form
with st.form("forecast_form"):
    st.subheader("Forecast Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_date = st.date_input("Target Date", min_value=date.today())
        vehicle_type = st.selectbox("Vehicle Type", ["SEDN", "SUV", "MC", "MINI", "MID-MINI", "Sprinter"])
        
    with col2:
        demand_7_days_ago = st.number_input("Demand 7 Days Ago", min_value=0, value=20)
        demand_14_days_ago = st.number_input("Demand 14 Days Ago", min_value=0, value=20)
        
    # Submit Button
    submitted = st.form_submit_button("Generate Forecast")

# Handle API Request
if submitted:
    # Package UI inputs into the JSON payload API expects
    payload = {
        "target_date": target_date.strftime("%Y-%m-%d"),
        "vehicle_type": vehicle_type,
        "demand_7_days_ago": demand_7_days_ago,
        "demand_14_days_ago": demand_14_days_ago
    }
    
    with st.spinner("Querying Cloud Run API.."):
        try:
            # Send POST request to cloud
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                prediction = response.json()["predicted_demand"]
                st.success("Forecast generated successfully!")
                st.metric(label=f"Predicted {vehicle_type}s Needed", value=prediction)
            else:
                st.error(f"API Error: {response.text}")
                
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")