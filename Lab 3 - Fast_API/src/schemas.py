"""
Pydantic request/response schemas for the financial condition API.

Validates incoming JSON and shapes the API response for clients.
"""

from typing import Optional

from pydantic import BaseModel


class FinancialRequest(BaseModel):
    """
    Input schema for POST /predict/finance.
    Must match the feature columns used by the model (after dropping non-features).
    """

    age: int
    gender: str
    education_level: str
    employment_status: str
    job_title: str
    monthly_income_usd: float
    monthly_expenses_usd: float
    savings_usd: float
    has_loan: str
    loan_type: Optional[str] = "None"
    loan_amount_usd: float
    loan_term_months: int
    monthly_emi_usd: float
    loan_interest_rate_pct: float
    region: str


class FinancialResponse(BaseModel):
    """
    Output schema for POST /predict/finance.
    """

    prediction: int  # 1 = Good Financial Condition, 0 = Needs Improvement
    status_label: str  # "Good" or "Needs Improvement"

    # Example shown in Swagger/OpenAPI docs; avoids generic "string" placeholder
    model_config = {
        "json_schema_extra": {
            "example": {"prediction": 0, "status_label": "Needs Improvement"}
        }
    }

