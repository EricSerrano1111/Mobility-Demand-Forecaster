from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import holidays
import joblib
from pathlib import Path
from datetime import datetime

# Initialize API
app = FastAPI(
    title="Mobility Demand Forecasting API",
    description="Predicts fleet vehicle demand using a serialized Linear Regression model.",
    version="1.0.0"
)

# Define exact features model was trained on (Order matters..)
# 2. Define the exact features our model was trained on (Order matters!)
EXPECTED_COLUMNS = [
    'Demand_7_Days_Ago', 
    'Demand_14_Days_Ago', 
    'DayOfWeek', 
    'Month', 
    'Year', 
    'is_Holiday',
    'type_MC', 
    'type_MID-MINI', 
    'type_MINI', 
    'type_SEDN', 
    'type_SUV', 
    'type_Sprinter'
]

# Load Model Artifact on Startup to ensure model loaded into memory (Not every time a request is made)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "linear_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully into memory..")
except Exception as e:
    print(f"Warning: Model not found at {MODEL_PATH}. Error: {e}")
    model = None

# Define Expected JSON Payload
class ForecastRequest(BaseModel):
    target_date: str                 # Format: YYYY-MM-DD
    vehicle_type: str                # ex. 'SEDN', 'SUV', 'MC'
    demand_7_days_ago: float         # Historical lookup provided by client
    demand_14_days_ago: float        # Historical lookup provided by client

@app.get("/")
def health_check():
    return {"status": "operational", "model_loaded": model is not None}

@app.post("/predict")
def predict_demand(request: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")
        
    try:
        # Parse date
        target_dt = datetime.strptime(request.target_date, "%Y-%m-%d").date()
        
        # Initialize a dictionary with zeros for all expected columns
        feature_dict = {col: 0 for col in EXPECTED_COLUMNS}
        
        # Inject provided historical lags
        feature_dict['Demand_7_Days_Ago'] = request.demand_7_days_ago
        feature_dict['Demand_14_Days_Ago'] = request.demand_14_days_ago
        
        # Engineer Time Features
        feature_dict['DayOfWeek'] = target_dt.weekday()
        feature_dict['Month'] = target_dt.month
        feature_dict['Year'] = target_dt.year
        
        # Engineer Holiday Feature
        us_holidays = holidays.US()
        feature_dict['is_Holiday'] = 1 if target_dt in us_holidays else 0
        
        # Engineer Categorical Vehicle Type (One-Hot toggle)
        type_col = f"type_{request.vehicle_type}"
        if type_col in feature_dict:
            feature_dict[type_col] = 1
        else:
            raise ValueError(f"Unknown vehicle type: {request.vehicle_type}")
            
        # Convert to 1-row DataFrame with exact column order
        features_df = pd.DataFrame([feature_dict])[EXPECTED_COLUMNS]
        
        # Execute math
        prediction = model.predict(features_df)[0]
        
        # Return clean JSON response (floored to avoid predicting fractions of a car)
        return {
            "target_date": request.target_date,
            "vehicle_type": request.vehicle_type,
            "predicted_demand": max(0, int(prediction)) # Ensure never predict negative cars
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")