from datetime import date

import pytest
from checkout.errors import ValidationError
from checkout.models import Card
from checkout.validators import CardValidator


def test_validate_card_missing_number():
    with pytest.raises(ValidationError):
        CardValidator().validate_card(Card(number="", expiry=date(2030, 1, 1)))


def test_luhn_ok_valid_number():
    assert CardValidator().luhn_ok("4539578763621486")
