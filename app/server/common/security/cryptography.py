from __future__ import annotations

import hashlib

from cryptography.fernet import Fernet, InvalidToken


###############################################################################
def _load_fernet_from_material(key_material: str) -> Fernet:
    normalized = str(key_material or "").strip()
    if not normalized:
        raise RuntimeError("Encryption key material is missing")
    try:
        return Fernet(normalized.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Encryption key material is invalid") from exc


###############################################################################
def encrypt_with_key_material(plaintext: str, key_material: str) -> str:
    normalized = str(plaintext or "").strip()
    if not normalized:
        raise ValueError("Access key must not be empty")
    token = _load_fernet_from_material(key_material).encrypt(normalized.encode("utf-8"))
    return token.decode("utf-8")


###############################################################################
def decrypt_with_key_material(ciphertext: str, key_material: str) -> str:
    normalized = str(ciphertext or "").strip()
    if not normalized:
        raise ValueError("Encrypted access key must not be empty")
    try:
        decoded = _load_fernet_from_material(key_material).decrypt(
            normalized.encode("utf-8")
        )
    except InvalidToken as exc:
        raise RuntimeError("Encrypted access key is invalid") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to decrypt access key") from exc
    return decoded.decode("utf-8")


###############################################################################
def fingerprint_plaintext(plaintext: str) -> str:
    normalized = str(plaintext or "").strip()
    if not normalized:
        raise ValueError("Access key must not be empty")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
