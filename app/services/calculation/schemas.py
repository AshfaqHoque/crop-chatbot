from pydantic import BaseModel, Field


# ---------- Type 1: Rate scaling ----------

class RateFromContext(BaseModel):
    """
    A rate statement extracted from context, e.g. "16-20 kg seed needed
    per hectare" -> quantity_min=16, quantity_max=20, quantity_unit="kg",
    per_quantity=1, per_unit="hectare".

    quantity_max is None when context gives a single number, not a range
    (e.g. "45 taka per decimal" for cost line items).
    """
    subject: str = Field(..., description="What is being measured, e.g. 'seed', 'urea fertilizer'")
    quantity_min: float = Field(..., description="Lower (or only) bound of the amount")
    quantity_max: float | None = Field(None, description="Upper bound, if context gives a range. Null if a single value.")
    quantity_unit: str = Field(..., description="Unit of the amount, e.g. 'kg', 'gram'")
    per_quantity: float = Field(1, description="Denominator amount, usually 1")
    per_unit: str = Field(..., description="Unit the rate is 'per', e.g. 'hectare', 'decimal', 'acre'")


class TargetFromQuestion(BaseModel):
    quantity: float = Field(..., description="The amount the user asked about, e.g. 2.5")
    unit: str = Field(..., description="Unit the user used, e.g. 'acre', 'bigha', 'shotangsho'")


class RateCalculationResult(BaseModel):
    subject: str
    result_min: float
    result_max: float | None
    result_unit: str
    target_quantity: float
    target_unit: str
    source_rate: str  # human-readable trace, useful for debugging/logging

