"""
OCR extraction and dimension parsing module.
Extracts text from images and parses HVAC dimensions using regex.
"""
import re
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
import pytesseract
from pytesseract import Output

from app.core.config import settings
from app.services.google_vision import GoogleVisionOCR

logger = logging.getLogger("hvac_analyzer.pipeline")


@dataclass
class TextBlock:
    """Represents a block of extracted text with position."""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounding box."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class DimensionLabel:
    """Represents a parsed dimension label."""
    text: str
    width_in: Optional[float]
    height_in: Optional[float]
    diameter_in: Optional[float]
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def is_rectangular(self) -> bool:
        return self.width_in is not None and self.height_in is not None
    
    @property
    def is_round(self) -> bool:
        return self.diameter_in is not None
    
    @property
    def dims_str(self) -> str:
        """Return formatted dimension string."""
        if self.is_rectangular:
            return f"{int(self.width_in)}×{int(self.height_in)}"
        elif self.is_round:
            return f"{int(self.diameter_in)}Ø"
        return self.text


class ExtractionService:
    """Extracts and parses text from HVAC drawings."""
    
    # Dimension regex patterns
    # Rectangular: "12x8", "12×8", "12"x8"", "12 X 8", etc.
    RECT_PATTERN = re.compile(
        r'(\d{1,3})\s*["\u2033\']?\s*[xX×\*]\s*(\d{1,3})\s*["\u2033\']?',
        re.IGNORECASE
    )
    
    # Round: "14Ø", "14D", "14"D", "14 dia", "Ø14"
    ROUND_PATTERN = re.compile(
        r'(\d{1,3})\s*["\u2033\']?\s*[ØDd](?:ia)?|'
        r'[Ø](\d{1,3})',
        re.IGNORECASE
    )
    
    # Pressure class: "LP", "MP", "HP", "Low Pressure", etc.
    PRESSURE_PATTERN = re.compile(
        r'\b(LP|MP|HP|Low\s+Pressure|Med(?:ium)?\s+Pressure|High\s+Pressure)\b',
        re.IGNORECASE
    )
    
    # Scale: "1/8" = 1'-0", "1:96", etc.
    SCALE_PATTERN = re.compile(
        r'(\d+)[/:](\d+)|'
        r'"(\d+)["\u2033]?\s*=\s*1\'-0"|'
        r'(\d+)\s*:\s*1',
        re.IGNORECASE
    )
    
    def __init__(self):
        self.min_confidence = settings.OCR_MIN_CONFIDENCE
        self.psm_mode = settings.OCR_PSM_MODE
        self.google_vision = GoogleVisionOCR()
    
    def extract(self, image: np.ndarray) -> List[TextBlock]:
        """
        Extract all text blocks from image.
        
        Uses Google Vision API as primary OCR, falls back to Tesseract
        if Vision API is unavailable or fails.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of text blocks with positions
        """
        logger.debug("pipeline.extraction.started")
        
        # Use Tesseract OCR directly
        logger.info("pipeline.extraction.using_tesseract")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Run OCR with sparse text mode
        config = f'--psm {self.psm_mode}'
        data = pytesseract.image_to_data(gray, output_type=Output.DICT, config=config)
        
        blocks = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            conf = data['conf'][i]
            text = data['text'][i].strip()
            
            # Filter by confidence and minimum length
            if conf > self.min_confidence * 100 and len(text) >= 2:
                blocks.append(TextBlock(
                    text=text,
                    x=data['left'][i],
                    y=data['top'][i],
                    width=data['width'][i],
                    height=data['height'][i],
                    confidence=conf / 100.0
                ))
        
        logger.debug("pipeline.extraction.completed", extra={"count": len(blocks)})
        return blocks
    
    def parse_dimensions(self, text_blocks: List[TextBlock]) -> List[DimensionLabel]:
        """
        Parse dimension labels from text blocks.
        
        Args:
            text_blocks: List of extracted text blocks
            
        Returns:
            List of parsed dimension labels
        """
        dimensions = []
        
        for block in text_blocks:
            rect_match = self.RECT_PATTERN.search(block.text)
            round_match = self.ROUND_PATTERN.search(block.text)
            
            if rect_match:
                width_in = float(rect_match.group(1))
                height_in = float(rect_match.group(2))
                dimensions.append(DimensionLabel(
                    text=block.text,
                    width_in=width_in,
                    height_in=height_in,
                    diameter_in=None,
                    x=block.x,
                    y=block.y,
                    width=block.width,
                    height=block.height,
                    confidence=block.confidence
                ))
            elif round_match:
                # Find which group matched (group 1 or 2)
                diameter_in = None
                for group in round_match.groups():
                    if group:
                        diameter_in = float(group)
                        break
                
                if diameter_in:
                    dimensions.append(DimensionLabel(
                        text=block.text,
                        width_in=None,
                        height_in=None,
                        diameter_in=diameter_in,
                        x=block.x,
                        y=block.y,
                        width=block.width,
                        height=block.height,
                        confidence=block.confidence
                    ))
        
        logger.debug("pipeline.extraction.dimensions_found", extra={"count": len(dimensions)})
        return dimensions
    
    def extract_scale(
        self, 
        text_blocks: List[TextBlock],
        image_width_px: int,
        image_height_px: int
    ) -> Optional[float]:
        """
        Extract drawing scale from text blocks.
        
        Returns:
            Pixels per foot, or None if not found
        """
        for block in text_blocks:
            match = self.SCALE_PATTERN.search(block.text)
            if match:
                # Try different pattern groups
                if match.group(1) and match.group(2):
                    # Format: "1/8" = 1'-0" or "1:96"
                    numerator = int(match.group(1))
                    denominator = int(match.group(2))
                    if denominator > 0:
                        # Standard scale: 1 inch on drawing = X feet in reality
                        # At 300 DPI, 1 inch = 300 pixels
                        # So pixels per foot = 300 / X
                        return 300.0 / denominator
                
                elif match.group(3):
                    # Format: "1" = 1'-0"
                    inches = float(match.group(3))
                    if inches > 0:
                        return 300.0 / inches
        
        # Fallback: Assume 1/8" = 1'-0" (common HVAC scale)
        logger.warning("pipeline.extraction.scale_not_found", extra={"using_default": "1/8=1ft"})
        return 96.0  # 1/8" scale = 96 pixels per foot at 300 DPI
    
    def find_pressure_labels(self, text_blocks: List[TextBlock]) -> List[Tuple[str, TextBlock]]:
        """
        Find pressure class labels (LP, MP, HP).
        
        Returns:
            List of (pressure_class, text_block) tuples
        """
        results = []
        
        for block in text_blocks:
            match = self.PRESSURE_PATTERN.search(block.text)
            if match:
                pressure = match.group(1).upper()
                # Normalize
                if 'LOW' in pressure or pressure == 'LP':
                    pressure = 'LP'
                elif 'MED' in pressure or pressure == 'MP':
                    pressure = 'MP'
                elif 'HIGH' in pressure or pressure == 'HP':
                    pressure = 'HP'
                
                results.append((pressure, block))
        
        return results
