# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote

OCTET_STREAM = "application/octet-stream"
DEFAULT_FILE_NAME = "attachment"
MAX_FILE_NAME_LENGTH = 255

# Extensions whose content a browser or webview may execute or render actively.
ACTIVE_EXTENSIONS = frozenset(
    {
        "html", "htm", "xhtml", "shtml", "svg", "svgz", "xml", "xsl", "xslt",
        "js", "mjs", "jsx", "hta", "php", "asp", "aspx", "jsp", "swf",
        "wasm", "vbs", "ps1", "bat", "cmd", "exe", "msi", "dll", "jar",
    }
)  # fmt: skip

# Raster images that may be served inline once the leading bytes are verified.
IMAGE_MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
}

# Inert document formats with a fixed media type; always served as downloads.
DOCUMENT_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "json": "application/json",
    "txt": "text/plain",
    "csv": "text/csv",
    "md": "text/markdown",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "rtf": "application/rtf",
    "odt": "application/vnd.oasis.opendocument.text",
}

ATTACHMENT_RESPONSE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
}

CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
UNSAFE_ASCII_FILENAME_CHARACTERS = frozenset('"\\;%')


###############################################################################
class AttachmentPolicyError(ValueError):
    """Raised when an uploaded attachment violates the media policy."""


###############################################################################
@dataclass(frozen=True)
class AttachmentPolicy:
    file_name: str
    kind: str
    media_type: str
    inline: bool


# -----------------------------------------------------------------------------
def sanitize_file_name(file_name: str) -> str:
    name = CONTROL_CHARACTERS.sub("", file_name)
    name = name.replace("\\", "/").rsplit("/", 1)[-1].strip().strip(".")
    return name[:MAX_FILE_NAME_LENGTH] or DEFAULT_FILE_NAME


# -----------------------------------------------------------------------------
def file_extension(file_name: str) -> str:
    _, dot, extension = file_name.rpartition(".")
    return extension.lower() if dot else ""


# -----------------------------------------------------------------------------
def matches_image_signature(extension: str, content: bytes) -> bool:
    if extension == "png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if extension in {"jpg", "jpeg"}:
        return content.startswith(b"\xff\xd8\xff")
    if extension == "gif":
        return content.startswith((b"GIF87a", b"GIF89a"))
    if extension == "webp":
        return content[:4] == b"RIFF" and content[8:12] == b"WEBP"
    if extension == "bmp":
        return content.startswith(b"BM")
    return False


# -----------------------------------------------------------------------------
def resolve_attachment_policy(file_name: str, content: bytes) -> AttachmentPolicy:
    """Derive the stored and served media type from the name and verified bytes."""
    safe_name = sanitize_file_name(file_name)
    extension = file_extension(safe_name)
    if extension in IMAGE_MEDIA_TYPES and matches_image_signature(extension, content):
        return AttachmentPolicy(safe_name, "image", IMAGE_MEDIA_TYPES[extension], True)
    if extension in DOCUMENT_MEDIA_TYPES:
        kind = "json" if extension == "json" else "document"
        return AttachmentPolicy(safe_name, kind, DOCUMENT_MEDIA_TYPES[extension], False)
    return AttachmentPolicy(safe_name, "document", OCTET_STREAM, False)


# -----------------------------------------------------------------------------
def validate_upload(file_name: str, content: bytes) -> AttachmentPolicy:
    safe_name = sanitize_file_name(file_name)
    extension = file_extension(safe_name)
    if extension in ACTIVE_EXTENSIONS:
        raise AttachmentPolicyError(
            f"Attachments of type .{extension} are not allowed because they can "
            "contain active content."
        )
    if extension in IMAGE_MEDIA_TYPES and not matches_image_signature(
        extension, content
    ):
        raise AttachmentPolicyError(
            f"Attachment content does not match the .{extension} image format."
        )
    return resolve_attachment_policy(safe_name, content)


# -----------------------------------------------------------------------------
def build_content_disposition(disposition: str, file_name: str) -> str:
    ascii_name = "".join(
        char
        if 0x20 <= ord(char) < 0x7F and char not in UNSAFE_ASCII_FILENAME_CHARACTERS
        else "_"
        for char in file_name
    )
    return (
        f'{disposition}; filename="{ascii_name}"; '
        f"filename*=UTF-8''{quote(file_name, safe='')}"
    )


# -----------------------------------------------------------------------------
def build_download_response_parts(
    file_name: str, content: bytes
) -> tuple[str, dict[str, str]]:
    """Recompute serving policy from stored data so legacy rows are neutralized."""
    policy = resolve_attachment_policy(file_name, content)
    disposition = "inline" if policy.inline else "attachment"
    headers = {
        **ATTACHMENT_RESPONSE_HEADERS,
        "Content-Disposition": build_content_disposition(disposition, policy.file_name),
    }
    return policy.media_type, headers
