"""Order pricing and discount application."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Money:
    cents: int
    currency: str = "INR"

    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("currency mismatch")
        return Money(self.cents + other.cents, self.currency)


def apply_percent_discount(amount: Money, percent: float) -> Money:
    if percent < 0 or percent > 100:
        raise ValueError("percent out of range")
    cut = int(round(amount.cents * (percent / 100.0)))
    return Money(max(0, amount.cents - cut), amount.currency)


def tax_inclusive(amount: Money, rate: float = 0.18) -> Money:
    tax = int(round(amount.cents * rate))
    return Money(amount.cents + tax, amount.currency)
