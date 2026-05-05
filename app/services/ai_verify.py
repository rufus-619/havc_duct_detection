import asyncio
import base64
import json
import cv2
import numpy as np
from typing import List
from anthropic import AsyncAnthropic
from app.models.domain import DuctSegment, DuctType, PressureClass
from app.core.config import settings
from app.core.logging import logger

async def verify_segments(
    image: np.ndarray,
    segments: List[DuctSegment]
) -> List[DuctSegment]:
    
    if not settings.ANTHROPIC_API_KEY:
        logger.info("ai_verify_skipped", reason="ANTHROPIC_API_KEY not set")
        return segments
        
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    
    for segment in segments:
        if segment.detection_confidence is not None and segment.detection_confidence >= settings.HVAC_AI_VERIFY_THRESHOLD and not segment.fallback_used:
            continue
            
        # 1. Crop 200x200 centered on label_geometry
        cx = segment.label_geometry.x + segment.label_geometry.width // 2
        cy = segment.label_geometry.y + segment.label_geometry.height // 2
        
        h_img, w_img = image.shape[:2]
        x_min = max(0, cx - 100)
        y_min = max(0, cy - 100)
        x_max = min(w_img, cx + 100)
        y_max = min(h_img, cy + 100)
        
        crop = image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
            
        # 2. Encode as base64 PNG
        _, buffer = cv2.imencode('.png', crop)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        
        # 3. Call Claude
        try:
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                system="You are an HVAC mechanical drawing parser. Analyze the image crop from a mechanical engineering drawing. Return ONLY valid JSON with no markdown fences, no preamble:\n{\n  \"dims_str\": \"12x8\",\n  \"duct_type\": \"supply\" | \"return\" | \"unknown\",\n  \"pressure_class\": \"Low Pressure\" | \"Medium Pressure\" | \"High Pressure\" | \"Unknown\",\n  \"confidence\": 0.0\n}",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64_str,
                                }
                            },
                            {
                                "type": "text",
                                "text": "Extract duct metadata from this mechanical drawing crop."
                            }
                        ]
                    }
                ]
            )
            
            text = response.content[0].text.strip()
            # 4. Strip fences
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            data = json.loads(text)
            
            # 5. Update segment
            segment.dims_str = data.get("dims_str", segment.dims_str)
            try:
                segment.duct_type = DuctType(data.get("duct_type", "unknown"))
            except ValueError:
                pass
            try:
                segment.pressure_class = PressureClass(data.get("pressure_class", "Unknown"))
            except ValueError:
                pass
                
            segment.detection_confidence = data.get("confidence", segment.detection_confidence)
            
        except Exception as e:
            logger.warning("ai_verify_failed", error=str(e), segment_id=segment.id)
            
    return segments
