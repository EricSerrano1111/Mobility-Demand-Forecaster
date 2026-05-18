import pandas as pd
import holidays
from pathlib import Path

def load_and_aggregate_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Loads raw dispatch data and aggregate into Daily Volume per Vehicle Type.
    """
    print(f"Loading data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Parse dates
    date_format = '%b %d %Y %I:%M%p'
    df['PickupDateTimeScheduled'] = pd.to_datetime(df['PickupDateTimeScheduled'], format=date_format)
    df['Date'] = df['PickupDateTimeScheduled'].dt.date
    
    # Aggregate
    daily_demand = df.groupby(['Date', 'PrefferedVehicleType']).size().reset_index(name='TotalTrips')
    daily_demand['Date'] = pd.to_datetime(daily_demand['Date'])
    
    return daily_demand

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies lags, time-series extractions, and holiday flags.
    """
    print("Starting feature engineering pipeline..")
    
    # Sort for accurate lag shifting
    df = df.sort_values(by=['PrefferedVehicleType', 'Date']).reset_index(drop=True)
    
    # Lag Features
    df['Demand_7_Days_Ago'] = df.groupby('PrefferedVehicleType')['TotalTrips'].shift(7)
    df['Demand_14_Days_Ago'] = df.groupby('PrefferedVehicleType')['TotalTrips'].shift(14)
    df = df.dropna().reset_index(drop=True)
    
    # Time Extraction
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    # Holidays
    us_holidays = holidays.US()
    df['is_Holiday'] = df['Date'].apply(lambda x: 1 if x in us_holidays else 0)
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['PrefferedVehicleType'], prefix='type', dtype=int)
    
    print(f"Feature engineering complete. Final shape: {df.shape}")
    return df

# Allows to run file directly from the terminal for testing
if __name__ == "__main__":
    # Define absolute paths using Pathlib
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_mobility_data.csv"
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"
    
    # Execute pipeline
    raw_df = load_and_aggregate_data(RAW_DATA_PATH)
    final_df = engineer_features(raw_df)
    
    # Save output
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f" Pipeline successful.. Data saved to {PROCESSED_DATA_PATH}..")