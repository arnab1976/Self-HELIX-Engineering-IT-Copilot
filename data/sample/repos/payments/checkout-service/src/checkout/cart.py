"""Shopping cart."""
from decimal import Decimal


class Cart:
    def __init__(self, items=None):
        self.items = items or []

    def total(self) -> Decimal:
        return sum((item.price * item.qty for item in self.items), Decimal("0"))

    def apply_discount(self, pct: Decimal) -> Decimal:
        return (self.total() * (1 - pct / 100)).quantize(Decimal("0.01"))
