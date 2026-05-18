from fastapi.testclient import TestClient
from api.app import app

# Initialize test client with FastAPI app
client = TestClient(app)

def test_health_check():
    """Test that the API boots up and the model is loaded."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"
    assert response.json()["model_loaded"] is True

def test_predict_endpoint_success():
    """Test that the API correctly processes a valid payload."""
    payload = {
        "target_date": "2026-05-18",
        "vehicle_type": "SEDN",
        "demand_7_days_ago": 25,
        "demand_14_days_ago": 22
    }
    
    response = client.post("/predict", json=payload)
    
    # Check that the web request was successful
    assert response.status_code == 200
    
    # Check that the response contains the right data types
    data = response.json()
    assert "predicted_demand" in data
    assert isinstance(data["predicted_demand"], int)
    assert data["predicted_demand"] >= 0