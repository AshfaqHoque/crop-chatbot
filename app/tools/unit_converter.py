from enum import Enum
from decimal import Decimal

class UnitDimension(str, Enum):
    AREA = "area"
    MASS = "mass"
    VOLUME = "volume"

UNIT_FACTORS = {
    UnitDimension.AREA: {
        "decimal": Decimal("1"),
        "hectare": Decimal("247.105"),
        "acre": Decimal("100"),
        "bigha": Decimal("33"),
        "katha": Decimal("1.65"),
    },
    
    UnitDimension.MASS: {
        "gram": Decimal("1"),
        "kilogram": Decimal("1000"),
        "metric_tonne": Decimal("1000000"),
    },
    
    UnitDimension.VOLUME: {
        "liter": Decimal("1"),
        "milliliter": Decimal("0.001"),
    },
}

UNIT_ALIASES: dict[str, str] = {
    # Area
    "hector": "hectare",
    "hectors": "hectare",
    "acres": "acre",
    "shotok": "decimal",
    "shatak": "decimal",

    # Mass
    "g": "gram",
    "gm": "gram",
    "grams": "gram",
    "kg": "kilogram",
    "kgs": "kilogram",
    "ton": "metric_tonne",
    "tonne": "metric_tonne",
    "metric_ton": "metric_tonne",

    # Volume
    "l": "liter",
    "litre": "liter",
    "litres": "liter",
    "ml": "milliliter",

    # Bangla
    "গ্রাম": "gram",
    "কেজি": "kilogram",
    "কিলোগ্রাম": "kilogram",
    "হেক্টর": "hectare",
    "একর": "acre",
    "শতক": "decimal",
    "বিঘা": "bigha",
    "কাঠা": "katha",
    "লিটার": "liter",
    "মিলিলিটার": "milliliter",
}

# AREA_FACTORS = {
#     "decimal": 1.0,
#     "hectare": 247.105,
#     "acre": 100.0,
#     "bigha": 33.0,
#     "katha": 1.65,
# }

# class UnitConversionError(ValueError):
#     pass
# def normalize_unit(unit):
#     return unit.strip().lower().replace(" ", "_")

# def convert_area(value, from_unit, to_unit):
#     from_unit = normalize_unit(from_unit)
#     to_unit = normalize_unit(to_unit)
#     if from_unit not in AREA_FACTORS:
#         raise UnitConversionError(
#             f"Unknown unit: {from_unit}"
#         )
#     if to_unit not in AREA_FACTORS:
#         raise UnitConversionError(
#             f"Unknown unit: {to_unit}"
#         )
#     value_in_decimals = value * AREA_FACTORS[from_unit]
#     result = value_in_decimals / AREA_FACTORS[to_unit]
#     return result


# try:
#     answer = convert_area(1, "hectare", "acre")
#     print(answer)

# except UnitConversionError as error:
#     print(error)