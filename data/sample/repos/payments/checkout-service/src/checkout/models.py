"""Domain models."""
from dataclasses import dataclass
from datetime import date


@dataclass
class Card:
    number: str
    expiry: date
    holder: str = ""

    @property
    def last4(self) -> str:
        return self.number[-4:]


@dataclass
class Order:
    id: int
    customer_id: str
    total: float
    status: str = "NEW"
