# Customer Churn Prediction API (v2)

Redesigned to match the provided dataset schema:

| Column            | Type    | Description                          |
|-------------------|---------|--------------------------------------|
| CustomerID        | int     | Identifier (not used as feature)     |
| Age               | int     | Customer age                         |
| Tenure            | int     | Months with the company              |
| MonthlyCharges    | float   | Monthly bill amount                  |
| TotalCharges      | float   | Lifetime charges                     |
| Contract          | string  | Month-to-month / One Year / Two Year |
| InternetService   | string  | Fiber Optic / DSL / No               |
| SupportCalls      | int     | Number of support calls              |
| PaymentMethod     | string  | Electronic Check / Mailed Check / Credit Card / Bank Transfer |
| Churn             | Yes/No  | Target variable                      |

## Project Structure

```
customer-churn-api/
├── app/
│   ├── main.py          # FastAPI endpoints
│   ├── model.py         # Model loading & inference
│   └── schemas.py       # Pydantic models (new schema)
├── model/
│   ├── churn_model.pkl  # Trained pipeline
│   └── preprocessor.pkl
├── train_model.py       # Generates synthetic data + trains model
├── Dockerfile
├── requirements.txt
├── tests/test_api.py
└── README.md
```

## Quick Start

```bash
# 1. Install
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. (Re)train model if needed
python train_model.py

# 3. Run API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

### Example request (matches first sample row)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 56,
    "tenure": 1,
    "monthly_charges": 35.17,
    "total_charges": 35.17,
    "contract": "Month-to-month",
    "internet_service": "Fiber Optic",
    "support_calls": 6,
    "payment_method": "Electronic Check"
  }'
```

Expected response shape:
```json
{
  "churn_probability": 0.81,
  "churn_prediction": "Yes",
  "confidence": 0.81
}
```

## Docker

```bash
docker build -t customer-churn-api:v2 .
docker run -p 8000:8000 customer-churn-api:v2
```

## Azure Deployment (unchanged flow)

1. **ACR**
   ```bash
   az group create --name churn-rg --location eastus
   az acr create -g churn-rg -n <unique-acr-name> --sku Basic --admin-enabled true
   az acr login -n <unique-acr-name>
   az acr build --registry <unique-acr-name> --image customer-churn-api:v2 .
   ```

2. **ACI**
   ```bash
   az container create \
     --resource-group churn-rg \
     --name churn-api-instance \
     --image <unique-acr-name>.azurecr.io/customer-churn-api:v2 \
     --registry-login-server <unique-acr-name>.azurecr.io \
     --registry-username $(az acr credential show -n <unique-acr-name> --query username -o tsv) \
     --registry-password $(az acr credential show -n <unique-acr-name> --query passwords[0].value -o tsv) \
     --dns-name-label churn-api-<unique> \
     --ports 8000 --cpu 1 --memory 1.5
   ```

3. Browser testing → `http://<fqdn>:8000/docs`

4. Monitoring
   ```bash
   az container logs -g churn-rg -n churn-api-instance
   az container attach -g churn-rg -n churn-api-instance
   ```

## API Endpoints

| Method | Path      | Description                  |
|--------|-----------|------------------------------|
| GET    | /         | Service info & feature list  |
| GET    | /health   | Health check                 |
| POST   | /predict  | Predict churn                |
| GET    | /docs     | Swagger UI                   |
| GET    | /redoc    | ReDoc                        |
