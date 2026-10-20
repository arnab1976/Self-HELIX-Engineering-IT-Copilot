"""Compensation calculations (CONFIDENTIAL repository)."""


def compute_bonus(employee) -> float:
    """Bonus = base salary x band percentage x rating multiplier."""
    band_pct = {"B1": 0.08, "B2": 0.10, "B3": 0.15}[employee.band]
    return employee.base * band_pct * employee.rating_multiplier
