import cv2
import numpy as np
import tempfile
import asyncio
from fastapi import UploadFile
from pdf2image import convert_from_bytes
from app.core.config import settings
from app.core.exceptions import PayloadTooLargeError, UnsupportedMediaTypeError, FileCorruptionError

async def normalize(upload: UploadFile) -> tuple[np.ndarray, int, int]:
    # Validate mime type
    valid_mimes = ["image/jpeg", "image/png", "application/pdf"]
    if upload.content_type not in valid_mimes:
        raise UnsupportedMediaTypeError(f"Unsupported media type: {upload.content_type}. Supported types: png, jpeg, pdf")
        
    # Read file data to check size and content
    # For large files we could use spooled file, but for simplicity here we read into memory.
    # The requirement asks for SpooledTemporaryFile but FastAPI's UploadFile is already a SpooledTemporaryFile
    # Max size check
    max_bytes = settings.HVAC_MAX_FILE_MB * 1024 * 1024
    
    file_bytes = await upload.read()
    if len(file_bytes) > max_bytes:
        raise PayloadTooLargeError(f"File size exceeds maximum allowed size of {settings.HVAC_MAX_FILE_MB}MB")

    # If PDF
    if upload.content_type == "application/pdf":
        try:
            # First page only, 400 DPI for better small text detection
            images = convert_from_bytes(file_bytes, dpi=400, first_page=1, last_page=1)
            if not images:
                raise FileCorruptionError("Could not extract image from PDF")
            # Convert PIL Image to BGR OpenCV format
            pil_image = images[0].convert('RGB')
            img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise FileCorruptionError(f"Error processing PDF: {str(e)}")
    else:
        # If Image
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileCorruptionError("Could not decode image file")
            
    height, width = img_bgr.shape[:2]
    return img_bgr, width, height
