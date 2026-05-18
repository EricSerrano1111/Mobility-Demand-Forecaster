import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

def train_and_evaluate():
    """
    Loads engineered features, splits chronologically, trains the model,
    and serializes the artifact for production.
    """
    # Define Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"
    MODEL_DIR = PROJECT_ROOT / "models"
    MODEL_PATH = MODEL_DIR / "linear_model.pkl"
    
    print(f"Loading engineered features from: {PROCESSED_DATA_PATH}..")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Ensure Date is a datetime object for splitting
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Chronological Split 
    split_date = pd.to_datetime('2026-01-01')
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()
    
    # Isolate Features (X) and Target (y) / Drop Date column
    drop_cols = ['TotalTrips', 'Date', 'PrefferedVehicleType'] 
    
    # Eensure PrefferedVehicleType is dropped if it somehow survived One-Hot Encoding
    existing_drop_cols = [col for col in drop_cols if col in train_df.columns]
    
    X_train = train_df.drop(columns=existing_drop_cols)
    y_train = train_df['TotalTrips']
    
    X_test = test_df.drop(columns=existing_drop_cols)
    y_test = test_df['TotalTrips']
    
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")
    
    # Train model
    print("Training Linear Regression model..")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("-" * 40)
    print("Model Evaluation on 2026 Test Data:")
    print(f"MAE:  {mae:.2f} trips")
    print(f"RMSE: {rmse:.2f} trips")
    print("-" * 40)
    
    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model successfully saved to: {MODEL_PATH}..")

if __name__ == "__main__":
    train_and_evaluate()