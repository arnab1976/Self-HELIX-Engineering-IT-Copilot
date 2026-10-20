"""Shipping address validation."""
import re

from checkout.errors import ValidationError

POSTCODE = re.compile(r"^[1-9][0-9]{5}$")


def validate_address(addr) -> None:
    if not addr.line1:
        raise ValidationError("address line 1 is required")
    if not POSTCODE.match(addr.postcode or ""):
        raise ValidationError("invalid postcode")
