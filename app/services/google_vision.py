"""
Google Vision API integration for OCR text extraction.
Provides superior accuracy compared to Tesseract for small/dense text.
"""
import os
import base64
import logging
import time
from typing import List, Dict, Optional
import requests
import numpy as np
import cv2
from app.core.config import settings

API_KEY = settings.GOOGLE_API_KEY


logger = logging.getLogger("hvac_analyzer.google_vision")


class GoogleVisionOCR:
    """
    Google Cloud Vision API client for text extraction.
    
    Requires GOOGLE_API_KEY environment variable to be set.
    Falls back to Tesseract if API key is not available.
    """
    
    API_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
    
    def __init__(self):
        self.api_key = API_KEY
        if not self.api_key:
            logger.warning("google_vision.no_api_key", extra={
    "info": "GOOGLE_API_KEY not set, Vision API unavailable"
})
        else:
            logger.info("google_vision.initialized")
    
    def is_available(self) -> bool:
        """Check if Google Vision API is configured."""
        return self.api_key is not None
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text from image using Google Vision API.
        
        Args:
            image: BGR numpy array
            
        Returns:
            List of text blocks with description, bounding box, and confidence
            Each block: {
                "text": str,
                "x": int,
                "y": int,
                "width": int,
                "height": int,
                "confidence": float
            }
        """
        if not self.is_available():
            logger.error("google_vision.not_available")
            return []
        
        try:
            start_time = time.time()
            
            # Encode image to base64
            _, buffer = cv2.imencode('.png', image)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            image_size_kb = len(buffer) / 1024
            
            # Log request details
            logger.info("google_vision.request", extra={
                "image_shape": image.shape,
                "image_size_kb": round(image_size_kb, 1),
                "endpoint": self.API_ENDPOINT,
                "features": ["TEXT_DETECTION", "DOCUMENT_TEXT_DETECTION"],
                "api_key_preview": self.api_key[:10] + "..." if self.api_key else "None"
            })
            
            # DEBUG: Print request summary
            print(f"\n[GOOGLE VISION REQUEST] Image: {image.shape}, Size: {round(image_size_kb, 1)}KB")
            
            # Prepare request
            request_body = {
                "requests": [{
                    "image": {"content": image_b64},
                    "features": [
                        {"type": "TEXT_DETECTION", "maxResults": 100},
                        {"type": "DOCUMENT_TEXT_DETECTION"}
                    ],
                    "imageContext": {
                        "languageHints": ["en"]  # English text
                    }
                }]
            }
            
            # Make API call
            url = f"{self.API_ENDPOINT}?key={self.api_key}"
            response = requests.post(
                url,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            elapsed_ms = round((time.time() - start_time) * 1000, 1)
            
            # Log response details
            logger.info("google_vision.response", extra={
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "content_length": len(response.content)
            })
            
            # DEBUG: Print response summary
            print(f"[GOOGLE VISION RESPONSE] Status: {response.status_code}, Time: {elapsed_ms}ms")
            
            if response.status_code != 200:
                logger.error("google_vision.api_error", extra={
                    "status_code": response.status_code,
                    "response": response.text[:1000],
                    "elapsed_ms": elapsed_ms
                })
                return []
            
            result = response.json()
            
            if "responses" not in result or not result["responses"]:
                logger.warning("google_vision.no_response", extra={
                    "elapsed_ms": elapsed_ms
                })
                return []
            
            vision_response = result["responses"][0]
            
            # Check for errors
            if "error" in vision_response:
                error_details = vision_response["error"]
                # DEBUG: Print full error to stdout
                print(f"\n[GOOGLE VISION ERROR] {error_details}")
                print(f"[GOOGLE VISION RESPONSE KEYS] {list(vision_response.keys())}")
                
                # Log individual fields
                logger.error("google_vision.api_error_details", extra={
                    "error_code": error_details.get("code") or "N/A",
                    "error_message": error_details.get("message") or "N/A",
                    "error_status": error_details.get("status") or "N/A",
                    "elapsed_ms": elapsed_ms
                })
                # Also log full error details for debugging
                logger.error("google_vision.api_error_full", extra={
                    "full_error": str(error_details),
                    "response_keys": list(vision_response.keys())
                })
                return []
            
            # Parse text annotations
            text_annotations = vision_response.get("textAnnotations", [])
            
            if not text_annotations:
                logger.info("google_vision.no_text_found")
                return []
            
            # Skip first annotation (full image text), process individual blocks
            text_blocks = []
            for annotation in text_annotations[1:]:  # Skip index 0 (full text)
                text = annotation.get("description", "").strip()
                if not text:
                    continue
                
                # Extract bounding box
                vertices = annotation.get("boundingPoly", {}).get("vertices", [])
                if len(vertices) >= 4:
                    xs = [v.get("x", 0) for v in vertices]
                    ys = [v.get("y", 0) for v in vertices]
                    x, y = min(xs), min(ys)
                    width = max(xs) - x
                    height = max(ys) - y
                else:
                    x, y, width, height = 0, 0, 0, 0
                
                # Confidence not provided for TEXT_DETECTION, use default
                confidence = annotation.get("confidence", 0.9)
                
                text_blocks.append({
                    "text": text,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "confidence": confidence
                })
            
            logger.info("google_vision.extraction_complete", extra={
                "blocks_found": len(text_blocks),
                "elapsed_ms": elapsed_ms
            })
            
            return text_blocks
            
        except requests.exceptions.Timeout:
            elapsed_ms = round((time.time() - start_time) * 1000, 1)
            logger.error("google_vision.timeout", extra={"elapsed_ms": elapsed_ms})
            return []
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 1)
            logger.error("google_vision.exception", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "elapsed_ms": elapsed_ms
            })
            return []
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects using Google Vision API object detection.
        
        Args:
            image: BGR numpy array
            
        Returns:
            List of detected objects with bounding boxes
        """
        if not self.is_available():
            logger.error("google_vision.not_available")
            return []
        
        try:
            start_time = time.time()
            
            # Encode image to base64
            _, buffer = cv2.imencode('.png', image)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            image_size_kb = len(buffer) / 1024
            
            logger.info("google_vision.object_detection.request", extra={
                "image_shape": image.shape,
                "image_size_kb": round(image_size_kb, 1)
            })
            
            # Prepare request with OBJECT_LOCALIZATION feature
            request_body = {
                "requests": [{
                    "image": {"content": image_b64},
                    "features": [
                        {"type": "OBJECT_LOCALIZATION", "maxResults": 50}
                    ]
                }]
            }
            
            # Make API call
            url = f"{self.API_ENDPOINT}?key={self.api_key}"
            response = requests.post(
                url,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            elapsed_ms = round((time.time() - start_time) * 1000, 1)
            
            print(f"[GOOGLE VISION OBJECT DETECTION] Status: {response.status_code}, Time: {elapsed_ms}ms")
            
            if response.status_code != 200:
                error_text = response.text
                print(f"[GOOGLE VISION ERROR] Status {response.status_code}")
                print(f"[GOOGLE VISION ERROR BODY] {error_text[:1000]}")
                logger.error("google_vision.object_detection.error", extra={
                    "status_code": response.status_code,
                    "response": error_text[:500]
                })
                return []
            
            result = response.json()
            
            if "responses" not in result or not result["responses"]:
                logger.warning("google_vision.object_detection.no_response")
                return []
            
            vision_response = result["responses"][0]
            
            # Check for errors
            if "error" in vision_response:
                error_details = vision_response["error"]
                print(f"[GOOGLE VISION OBJECT ERROR] {error_details}")
                logger.error("google_vision.object_detection.api_error", extra={
                    "error": str(error_details)
                })
                return []
            
            # Debug: print raw response
            print(f"[GOOGLE VISION RAW RESPONSE KEYS] {list(vision_response.keys())}")
            if "localizedObjectAnnotations" in vision_response:
                print(f"[GOOGLE VISION OBJECTS COUNT] {len(vision_response['localizedObjectAnnotations'])}")
            else:
                print(f"[GOOGLE VISION NO OBJECTS KEY] Response: {str(vision_response)[:500]}")
            
            # Parse object annotations
            objects = vision_response.get("localizedObjectAnnotations", [])
            
            detected_objects = []
            for obj in objects:
                name = obj.get("name", "unknown")
                score = obj.get("score", 0.0)
                
                # Extract bounding box (normalized vertices)
                vertices = obj.get("boundingPoly", {}).get("normalizedVertices", [])
                if len(vertices) >= 4:
                    xs = [v.get("x", 0) for v in vertices]
                    ys = [v.get("y", 0) for v in vertices]
                    
                    # Convert normalized (0-1) to pixel coordinates
                    x = int(min(xs) * image.shape[1])
                    y = int(min(ys) * image.shape[0])
                    width = int((max(xs) - min(xs)) * image.shape[1])
                    height = int((max(ys) - min(ys)) * image.shape[0])
                else:
                    x, y, width, height = 0, 0, 0, 0
                
                detected_objects.append({
                    "name": name,
                    "confidence": score,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                })
            
            logger.info("google_vision.object_detection.complete", extra={
                "objects_found": len(detected_objects),
                "elapsed_ms": elapsed_ms
            })
            
            print(f"[GOOGLE VISION OBJECTS] Found: {len(detected_objects)}")
            for obj in detected_objects[:10]:  # Print first 10
                print(f"  - {obj['name']}: {obj['confidence']:.2f} at ({obj['x']},{obj['y']}) {obj['width']}x{obj['height']}")
            
            return detected_objects
            
        except Exception as e:
            logger.error("google_vision.object_detection.exception", extra={
                "error": str(e)
            })
            return []
    
    def extract_text_regions(self, image: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """
        Extract text from specific regions of interest.
        
        Args:
            image: Full BGR image
            regions: List of regions with x, y, width, height
            
        Returns:
            List of text blocks with region metadata
        """
        all_text_blocks = []
        
        for i, region in enumerate(regions):
            x, y = int(region["x"]), int(region["y"])
            w, h = int(region["width"]), int(region["height"])
            
            # Extract region
            region_img = image[y:y+h, x:x+w]
            if region_img.size == 0:
                continue
            
            # OCR on region
            blocks = self.extract_text(region_img)
            
            # Adjust coordinates to global image space
            for block in blocks:
                block["x"] += x
                block["y"] += y
                block["region_id"] = i
            
            all_text_blocks.extend(blocks)
        
        logger.info("google_vision.regions_complete", extra={
            "regions_processed": len(regions),
            "total_blocks": len(all_text_blocks)
        })
        
        return all_text_blocks
