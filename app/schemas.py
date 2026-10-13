from pydantic import BaseModel, Field
from typing import Optional


class CustomerData(BaseModel):
    """Input schema matching the provided customer churn dataset."""
    age: int = Field(..., ge=18, le=100, description="Customer age in years")
    tenure: int = Field(..., ge=0, description="Number of months the customer has stayed with the company")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges amount")
    total_charges: float = Field(..., ge=0, description="Total charges amount")
    contract: str = Field(
        ...,
        description="Contract type: Month-to-month, One Year, Two Year"
    )
    internet_service: str = Field(
        ...,
        description="Internet service: Fiber Optic, DSL, No"
    )
    support_calls: int = Field(..., ge=0, description="Number of support calls made by the customer")
    payment_method: str = Field(
        ...,
        description="Payment method: Electronic Check, Mailed Check, Credit Card, Bank Transfer"
    )


class PredictionResponse(BaseModel):
    """Response schema for churn prediction."""
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: str = Field(..., description="Yes or No")
    confidence: Optional[float] = Field(None, description="Confidence of the prediction")
