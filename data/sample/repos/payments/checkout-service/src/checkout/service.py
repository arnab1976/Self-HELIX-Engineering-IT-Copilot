"""Checkout orchestration: validate, tokenise, charge."""
import logging

from checkout.gateway import gateway
from checkout.repository import OrderRepository
from checkout.validators import CardValidator
from vault.client import tokenise_card

logger = logging.getLogger(__name__)


class CheckoutService:
    def __init__(self, repo: OrderRepository, validator: CardValidator | None = None):
        self.repo = repo
        self.validator = validator or CardValidator()

    def charge(self, order, card):
        """Validate the card, tokenise it in the vault and charge the payment gateway."""
        self.validator.validate_card(card)
        token = tokenise_card(card)
        result = gateway.charge(token, order.total)
        logger.info("order charged", extra={"order_id": order.id, "status": result.status})
        return result

    def get_order(self, order_id):
        return self.repo.find(order_id)
