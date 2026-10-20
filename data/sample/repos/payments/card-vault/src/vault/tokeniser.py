"""Card vault tokeniser (RESTRICTED repository: self-hosted models only)."""
from vault.hsm import hsm_encrypt


def tokenise_card(card) -> str:
    """Validate the PAN format, encrypt it with the HSM key and return an opaque token (format v2)."""
    if not card.number.isdigit():
        raise ValueError("PAN must be numeric")
    return "tok_v2_" + hsm_encrypt(card.number)


def decode_token(token: str) -> str:
    if not token.startswith("tok_v2_"):
        raise ValueError("E1043 unsupported token format")
    return token[7:]
