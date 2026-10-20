"""HTTP API for checkout."""
from checkout.errors import ValidationError
from checkout.service import CheckoutService
from fastapi import APIRouter, HTTPException

router = APIRouter()
service: CheckoutService = None  # injected at startup


@router.post("/checkout")
def post_checkout(req: dict):
    """POST /checkout - charge an existing order. Validation errors return HTTP 422."""
    order = service.get_order(req["order_id"])
    try:
        return service.charge(order, req["card"])
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
