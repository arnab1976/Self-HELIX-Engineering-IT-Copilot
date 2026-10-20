"""Checkout errors."""


class ValidationError(Exception):
    """Raised when a request fails validation; mapped to HTTP 422."""


class PaymentError(Exception):
    """Raised when the payment gateway declines or fails (E1043 = token decode failure)."""
