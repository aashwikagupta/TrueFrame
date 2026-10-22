import hashlib
import io

from PIL import Image, UnidentifiedImageError

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_BYTES = 5 * 1024 * 1024  # 5 MB


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_and_decode(data: bytes, content_type: str) -> Image.Image:
    """Validate type/size, decode the image, return an RGB PIL image."""
    if content_type not in ALLOWED_TYPES:
        raise ValueError("Only JPEG and PNG images are allowed")
    if len(data) > MAX_BYTES:
        raise ValueError("File too large (max 5 MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError("File is not a valid image")
    return img.convert("RGB")
