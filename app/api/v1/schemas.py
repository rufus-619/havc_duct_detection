from pydantic import BaseModel
from typing import List, Optional
from app.models.domain import DuctSegment

class ImageMetadata(BaseModel):
    original_width: int
    original_height: int

class PipelineTelemetry(BaseModel):
    ocr_avg_confidence: float
    total_ocr_blocks: int
    domain_labels_matched: int
    semantic_match_rate: float
    segments_detected: int
    fallback_count: int
    fallback_rate: float
    ai_verify_invocations: int
    duration_ms: dict[str, int]

class DetectionResponse(BaseModel):
    status: str = "success"
    trace_id: str
    message: str
    image_metadata: ImageMetadata
    pipeline_telemetry: PipelineTelemetry
    data: List[DuctSegment]
    annotated_image: Optional[str] = None # Base64 encoded string if requested or return separately

class ErrorResponse(BaseModel):
    status: str = "error"
    trace_id: str
    error_code: str
    message: str
