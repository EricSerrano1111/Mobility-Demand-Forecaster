```
fleet-demand-forecaster/
├── data/                    
│   ├── raw/                 # synthetic data script - initial files
│   └── processed/           # Cleaned, engineered data ready for MLflow training runs
├── docs/                    
│   └── FinalProject.pdf     # required assignment documentation
├── notebooks/               
│   ├── 01_data_generation.ipynb   # Script to generate the synthetic schema
│   ├── 02_eda.ipynb               # Proving the synthetic data distributions are valid
│   └── 03_model_prototyping.ipynb # Scratchpad for early Linear Reg vs XGBoost tests
├── src/                     
│   ├── __init__.py
│   ├── data_prep.py         # Modularized code moved out of notebooks for production
│   └── train.py             # MLflow tracking script (logs models, params, metrics)
├── api/                     
│   ├── app.py               # Flask REST API that exposes chosen model
│   └── requirements.txt     # API-specific dependencies for lighter containers
├── tests/                   
│   ├── test_data.py         # Pytest: Validates synthetic data schema and missing values
│   ├── test_model.py        # Pytest: Ensures the model outputs a valid prediction format
│   └── test_api.py          # Pytest: Mocks an API call to ensure 200 OK responses
├── Dockerfile               # Instructions to containerize the api/ folder
├── requirements.txt         # Root dependencies for the whole workspace (MLflow, etc.)
├── .gitignore               # Ensures data/ and sensitive files are kept out of GitHub
└── README.md                # Setup instructions and architecture overview
```
