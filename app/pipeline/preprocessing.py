"""
Image preprocessing module.
Applies Otsu adaptive thresholding and morphological operations
to isolate thick duct lines while erasing thin text/lines.
"""
import logging
from typing import Tuple
import numpy as np
import cv2

from app.core.config import settings

logger = logging.getLogger("hvac_analyzer.pipeline")


class PreprocessingService:
    """Preprocesses images for geometric detection."""
    
    def __init__(self):
        self.gaussian_kernel = self._ensure_odd(settings.CV_OTSU_GAUSSIAN_BLUR_KERNEL)
        self.morph_kernel_width = settings.CV_MORPH_KERNEL_WIDTH
        self.morph_kernel_height = settings.CV_MORPH_KERNEL_HEIGHT
        self.morph_iterations = settings.CV_MORPH_ITERATIONS
    
    @staticmethod
    def _ensure_odd(value: int) -> int:
        """Ensure kernel size is odd (required by OpenCV)."""
        return value if value % 2 == 1 else value + 1
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for duct line detection.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (preprocessed_binary, original_grayscale)
            - preprocessed_binary: Binary image with ducts highlighted
            - original_grayscale: Grayscale for visualization/debugging
        """
        logger.debug("pipeline.preprocessing.started")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Light Gaussian blur - just enough to remove pixel noise, not duct lines
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Otsu's adaptive thresholding
        # Binary inverse: dark lines become white on black background
        _, binary = cv2.threshold(
            blurred, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # SKIP morphological OPEN (was erasing duct lines)
        # Only do a small CLOSE to connect broken line segments
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        
        logger.debug("pipeline.preprocessing.completed")
        
        return closed, gray

    def inpaint_text(self, binary: np.ndarray, text_blocks: list) -> np.ndarray:
        """
        Erase text bounding boxes from a binary image before line detection.

        Draws filled white rectangles (background) over each OCR text block,
        so that text strokes don't interfere with HoughLinesP or morphological
        line extraction.

        Args:
            binary: Binary image (white lines on black background)
            text_blocks: List of TextBlock objects with x, y, width, height

        Returns:
            Binary image with text regions filled to background (black = 0)
        """
        if not text_blocks:
            return binary

        result = binary.copy()
        pad = 5  # Small padding around each text box
        for tb in text_blocks:
            x1 = max(0, tb.x - pad)
            y1 = max(0, tb.y - pad)
            x2 = min(binary.shape[1] - 1, tb.x + tb.width + pad)
            y2 = min(binary.shape[0] - 1, tb.y + tb.height + pad)
            cv2.rectangle(result, (x1, y1), (x2, y2), 0, -1)  # Fill with black (background)

        logger.info("pipeline.preprocessing.inpaint_text", extra={
            "text_blocks_erased": len(text_blocks),
            "pixels_before": int(cv2.countNonZero(binary)),
            "pixels_after": int(cv2.countNonZero(result))
        })
        return result

    def create_color_mask(self, image: np.ndarray, target_color: str = "blue") -> np.ndarray:
        """
        Create mask for specific duct colors (e.g., blue for supply).
        Useful for HVAC drawings where ducts are color-coded.
        
        Args:
            image: Input BGR image
            target_color: 'blue' for supply, 'red' for return (HVAC convention)
            
        Returns:
            Binary mask of detected colored lines
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if target_color == "blue":
            # Blue color range in HSV
            lower = np.array([100, 50, 50])
            upper = np.array([130, 255, 255])
        elif target_color == "red":
            # Red color range in HSV (wraps around 0/180)
            lower1 = np.array([0, 50, 50])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 50, 50])
            upper2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            final_pixels = cv2.countNonZero(red_mask)
            
            logger.info("pipeline.preprocessing.color_mask", extra={
                "target_color": target_color,
                "initial_pixels": cv2.countNonZero(mask1) + cv2.countNonZero(mask2),
                "final_pixels": final_pixels,
                "image_shape": image.shape[:2]
            })
            
            return red_mask
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower, upper)
        initial_pixels = cv2.countNonZero(mask)
        
        # Dilate to connect fragmented color regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        final_pixels = cv2.countNonZero(mask)
        
        logger.info("pipeline.preprocessing.color_mask", extra={
            "target_color": target_color,
            "initial_pixels": initial_pixels,
            "final_pixels": final_pixels,
            "image_shape": image.shape[:2]
        })
        
        return mask
