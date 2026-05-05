"""
PDF and image ingestion module.
Converts PDFs to high-DPI images for processing.
"""
import io
import logging
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import cv2
from pdf2image import convert_from_bytes
from PIL import Image

# Increase PIL image size limit to handle large blueprints
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check

from app.core.config import settings
from app.core.exceptions import FileCorruptionError, PayloadTooLargeError

logger = logging.getLogger("hvac_analyzer.pipeline")


class IngestionService:
    """Handles file ingestion from PDF and image sources."""
    
    def __init__(self):
        self.max_file_size = settings.HVAC_MAX_FILE_MB * 1024 * 1024
        self.pdf_dpi = settings.HVAC_PDF_DPI
    
    def ingest(self, file_bytes: bytes, content_type: str, filename: str) -> Tuple[np.ndarray, int, int]:
        """
        Ingest file and return BGR image array with dimensions.
        
        Args:
            file_bytes: Raw file bytes
            content_type: MIME type (e.g., 'application/pdf', 'image/jpeg')
            filename: Original filename
            
        Returns:
            Tuple of (BGR image array, width, height)
            
        Raises:
            PayloadTooLargeError: If file exceeds max size
            FileCorruptionError: If file cannot be processed
        """
        logger.info("pipeline.ingestion.started", extra={
            "file_name": filename,
            "content_type": content_type,
            "size_bytes": len(file_bytes)
        })
        
        # Size validation
        if len(file_bytes) > self.max_file_size:
            raise PayloadTooLargeError(
                f"File size {len(file_bytes)} exceeds limit of {settings.HVAC_MAX_FILE_MB}MB"
            )
        
        try:
            if content_type == "application/pdf":
                image = self._convert_pdf(file_bytes)
            elif content_type in ["image/jpeg", "image/jpg", "image/png"]:
                image = self._load_image(file_bytes)
            else:
                raise FileCorruptionError(f"Unsupported content type: {content_type}")
            
            height, width = image.shape[:2]
            
            logger.info("pipeline.ingestion.completed", extra={
                "file_name": filename,
                "width": width,
                "height": height
            })
            
            return image, width, height
            
        except Exception as e:
            logger.error("pipeline.ingestion.failed", extra={
                "file_name": filename,
                "error_msg": str(e)
            })
            raise FileCorruptionError(f"Failed to process file: {str(e)}")
    
    def _convert_pdf(self, file_bytes: bytes) -> np.ndarray:
        """Convert PDF bytes to BGR image array."""
        logger.info("pipeline.ingestion.converting_pdf", extra={"dpi": self.pdf_dpi})
        
        # Convert first page only
        pil_images = convert_from_bytes(
            file_bytes,
            dpi=self.pdf_dpi,
            first_page=1,
            last_page=1,
            fmt='ppm'  # Faster than PNG
        )
        
        if not pil_images:
            raise FileCorruptionError("Could not extract any pages from PDF")
        
        # Convert PIL to OpenCV BGR
        pil_image = pil_images[0].convert('RGB')
        np_array = np.array(pil_image)
        bgr_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
        
        logger.debug("pipeline.ingestion.pdf_converted", extra={
            "width": bgr_image.shape[1],
            "height": bgr_image.shape[0]
        })
        
        return bgr_image
    
    def _load_image(self, file_bytes: bytes) -> np.ndarray:
        """Load image bytes to BGR array."""
        logger.info("pipeline.ingestion.loading_image")
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise FileCorruptionError("Could not decode image")
        
        return image
    
    def save_debug_image(self, image: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save intermediate image for debugging."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        logger.debug("pipeline.ingestion.saved_debug", extra={"path": str(output_path)})
