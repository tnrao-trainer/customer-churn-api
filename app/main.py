from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import CustomerData, PredictionResponse
from app.model import model

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "REST API for predicting customer churn based on: "
        "Age, Tenure, MonthlyCharges, TotalCharges, Contract, "
        "InternetService, SupportCalls, PaymentMethod."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint – service info."""
    return {
        "message": "Customer Churn Prediction API is running",
        "version": "2.0.0",
        "features": [
            "age", "tenure", "monthly_charges", "total_charges",
            "contract", "internet_service", "support_calls", "payment_method"
        ],
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    """Health check endpoint for monitoring / Azure Container Instances."""
    return {"status": "healthy", "service": "customer-churn-api", "version": "2.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Predict whether a customer is likely to churn.

    **Required fields**
    - age, tenure, monthly_charges, total_charges
    - contract (Month-to-month | One Year | Two Year)
    - internet_service (Fiber Optic | DSL | No)
    - support_calls
    - payment_method (Electronic Check | Mailed Check | Credit Card | Bank Transfer)
    """
    try:
        result = model.predict(customer.model_dump())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
