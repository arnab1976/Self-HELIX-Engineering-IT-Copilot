"""Card and order validation for the checkout service."""
import re
from datetime import date

from checkout.errors import ValidationError
from checkout.models import Card


class CardValidator:
    """Validates payment cards before tokenisation by the card vault (see ADR-012)."""

    BIN_PATTERN = re.compile(r"^\d{6}")

    def validate_card(self, card: Card) -> None:
        if not card.number:
            raise ValidationError("card number is required")
        if not self.luhn_ok(card.number):
            raise ValidationError("invalid card number")
        if card.expiry < date.today():
            raise ValidationError("card has expired")
        return None

    def luhn_ok(self, number: str) -> bool:
        digits = [int(d) for d in number][::-1]
        total = sum(digits[0::2]) + sum(sum(divmod(2 * d, 10)) for d in digits[1::2])
        return total % 10 == 0
