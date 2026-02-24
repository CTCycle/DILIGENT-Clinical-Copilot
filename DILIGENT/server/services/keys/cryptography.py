from __future__ import annotations

import hashlib

from cryptography.fernet import Fernet, InvalidToken

from DILIGENT.common.utils.variables import env_variables


###############################################################################
def _load_fernet() -> Fernet:
    key_value = env_variables.get("ACCESS_KEY_ENCRYPTION_KEY")
    if not isinstance(key_value, str) or not key_value.strip():
        raise RuntimeError("ACCESS_KEY_ENCRYPTION_KEY is not configured")
    try:
        return Fernet(key_value.strip().encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ACCESS_KEY_ENCRYPTION_KEY is invalid") from exc


###############################################################################
def encrypt(plaintext: str) -> str:
    normalized = str(plaintext or "").strip()
    if not normalized:
        raise ValueError("Access key must not be empty")
    token = _load_fernet().encrypt(normalized.encode("utf-8"))
    return token.decode("utf-8")


###############################################################################
def decrypt(ciphertext: str) -> str:
    normalized = str(ciphertext or "").strip()
    if not normalized:
        raise ValueError("Encrypted access key must not be empty")
    try:
        decoded = _load_fernet().decrypt(normalized.encode("utf-8"))
    except InvalidToken as exc:
        raise RuntimeError("Encrypted access key is invalid") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to decrypt access key") from exc
    return decoded.decode("utf-8")


###############################################################################
def fingerprint(ciphertext: str) -> str:
    normalized = str(ciphertext or "").strip()
    if not normalized:
        raise ValueError("Encrypted access key must not be empty")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
