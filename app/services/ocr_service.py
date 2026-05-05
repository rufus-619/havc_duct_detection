import pytesseract
import json
import cv2
import numpy as np
from pytesseract import Output
from typing import List
import concurrent.futures
from app.models.domain import ParsedText, BoundingBox
from app.core.config import settings
from app.core.exceptions import OCRTimeoutError

class OCRAdapter:
    """Stable interface. Implementation can be swapped to AWS Textract,
    Google Cloud Vision, etc. by modifying only this class."""
    def extract(self, image: np.ndarray) -> List[ParsedText]:
        raise NotImplementedError

class TesseractOCRAdapter(OCRAdapter):
    def _preprocess_for_ocr(self, image: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Preprocess image to enhance text detection."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if scale != 1.0
        if scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Morphological operation to thicken thin text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thickened = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return thickened
    
    def _run_ocr_at_scale(self, image: np.ndarray, scale: float, min_conf: float) -> List[ParsedText]:
        """Run OCR at a specific scale and return results scaled back to original coordinates."""
        processed = self._preprocess_for_ocr(image, scale)
        
        config = '--psm 11 --oem 3'  # Sparse text, LSTM only
        data = pytesseract.image_to_data(processed, output_type=Output.DICT, config=config)
        
        results = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            conf = data['conf'][i]
            text = data['text'][i].strip()
            if conf != -1 and conf > min_conf and text and len(text) >= 2:  # At least 2 chars
                # Scale coordinates back to original
                results.append(ParsedText(
                    text=text,
                    confidence=float(conf) / 100.0,
                    geometry=BoundingBox(
                        x=int(data['left'][i] / scale),
                        y=int(data['top'][i] / scale),
                        width=int(data['width'][i] / scale),
                        height=int(data['height'][i] / scale)
                    )
                ))
        return results
    
    def extract(self, image: np.ndarray) -> List[ParsedText]:
        """OCR extraction with preprocessing optimized for technical drawings."""
        # Single scale with enhanced preprocessing
        min_conf = settings.HVAC_MIN_OCR_CONF * 100
        return self._run_ocr_at_scale(image, 1.0, min_conf)

class MockOCRAdapter(OCRAdapter):
    """Injected in all tests. Reads from tests/mocks/ocr_fixtures.json."""
    def __init__(self, fixture_path: str = "tests/mocks/ocr_fixtures.json"):
        self.fixture_path = fixture_path
        
    def extract(self, image: np.ndarray) -> List[ParsedText]:
        try:
            with open(self.fixture_path, "r") as f:
                data = json.load(f)
                return [ParsedText(**item) for item in data]
        except FileNotFoundError:
            return []
