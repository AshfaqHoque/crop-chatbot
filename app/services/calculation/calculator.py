"""
Pure, deterministic math for both calculation types. No LLM calls here.
"""
from app.services.calculation.schemas import (
    RateCalculationResult,
    RateFromContext,
    TargetFromQuestion,
)
from app.services.calculation.units import UnitConversionError, convert


class CalculationError(Exception):
    pass


def calculate_rate(rate: RateFromContext, target: TargetFromQuestion) -> RateCalculationResult:
    try:
        target_in_rate_unit = convert(target.quantity, target.unit, rate.per_unit)
    except UnitConversionError as exc:
        raise CalculationError(f"Can't relate '{target.unit}' to '{rate.per_unit}': {exc}") from exc

    factor = target_in_rate_unit / rate.per_quantity

    result_min = round(rate.quantity_min * factor, 3)
    result_max = round(rate.quantity_max * factor, 3) if rate.quantity_max is not None else None

    range_str = (
        f"{rate.quantity_min}-{rate.quantity_max}" if rate.quantity_max is not None else str(rate.quantity_min)
    )
    return RateCalculationResult(
        subject=rate.subject,
        result_min=result_min,
        result_max=result_max,
        result_unit=rate.quantity_unit,
        target_quantity=target.quantity,
        target_unit=target.unit,
        source_rate=f"{range_str} {rate.quantity_unit} per {rate.per_quantity} {rate.per_unit}",
    )