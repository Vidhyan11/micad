"""Contribution 3: fairness-of-reasoning audit + mitigation."""
from .calibrate import apply_group_temperature, fit_group_temperature

__all__ = ["fit_group_temperature", "apply_group_temperature"]
