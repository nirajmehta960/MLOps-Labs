"""
FastAPI application for the financial condition prediction API.

Exposes a single prediction endpoint that uses the trained Linear Booster model
to classify whether a user is in good financial condition based on their profile.
"""

from fastapi import FastAPI, HTTPException, status, Body

from src.predict import predict_data_financial
from src.schemas import FinancialRequest, FinancialResponse


app = FastAPI(title="Financial Condition API", version="1.0.0")


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """
    Health check endpoint for load balancers and monitoring.
    Returns 200 OK if the service is running.
    """
    return {"status": "healthy"}


@app.post("/predict/finance", response_model=FinancialResponse)
async def predict_finance(
    financial_features: FinancialRequest = Body(
        ...,
        # Pre-filled example in Swagger UI for quick testing
        example={
            "age": 34,
            "gender": "Male",
            "education_level": "Bachelor",
            "employment_status": "Employed",
            "job_title": "Software Engineer",
            "monthly_income_usd": 7200.0,
            "monthly_expenses_usd": 2800.0,
            "savings_usd": 185000.0,
            "has_loan": "Yes",
            "loan_type": "Home",
            "loan_amount_usd": 220000.0,
            "loan_term_months": 240,
            "monthly_emi_usd": 1450.0,
            "loan_interest_rate_pct": 5.8,
            "region": "Asia",
        },
    )
):
    """
    Predict whether a user is in a 'good financial condition' (1) or 'needs improvement' (0).
    Accepts JSON with demographic and financial features; returns prediction + human-readable label.
    """
    try:
        # Step 1: Convert Pydantic model to dict for preprocessing
        request_dict = financial_features.model_dump()

        # Step 2: Run inference through feature pipeline + model
        prediction = predict_data_financial(request_dict)

        # Step 3: Extract scalar prediction (model returns 1D array)
        pred_val = int(prediction[0])

        # Step 4: Map numeric label to human-readable string
        status_label = "Good" if pred_val == 1 else "Needs Improvement"

        return FinancialResponse(prediction=pred_val, status_label=status_label)

    except Exception as e:
        # Surface any preprocessing or model errors as 500
        raise HTTPException(status_code=500, detail=str(e))

