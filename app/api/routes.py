"""
HVAC Duct Analyzer API Routes.
Uses the new geometry-first pipeline.
"""
import base64
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.pipeline.orchestrator import PipelineOrchestrator
from app.core.exceptions import HVACDomainError, FileCorruptionError, PayloadTooLargeError

logger = logging.getLogger("hvac_analyzer.api")
router = APIRouter()
orchestrator = PipelineOrchestrator()


class AnalysisResponse(BaseModel):
    status: str
    trace_id: str
    message: str
    image_metadata: dict
    processing_stats: dict
    ducts: list
    annotated_image: Optional[str]  # base64 encoded


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_drawing(request: Request, file: UploadFile = File(...)):
    """
    Analyze HVAC mechanical drawing and return annotated image.
    
    - **file**: PDF or image file (JPEG/PNG) of HVAC drawing
    
    Returns annotated image with detected ducts highlighted and dimension labels.
    """
    logger.info("api.analyze.request", extra={
        "file_name": file.filename,
        "content_type": file.content_type
    })
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Run pipeline
        output_bytes, metadata = orchestrator.process(
            file_bytes=file_bytes,
            content_type=file.content_type or "application/pdf",
            filename=file.filename or "unknown"
        )
        
        # Encode output to base64
        image_b64 = base64.b64encode(output_bytes).decode('utf-8')
        
        return AnalysisResponse(
            status="success",
            trace_id=metadata["trace_id"],
            message=f"Successfully analyzed drawing. Detected {metadata['processing_stats']['ducts_detected']} ducts.",
            image_metadata=metadata["image_dimensions"],
            processing_stats=metadata["processing_stats"],
            ducts=metadata["ducts"],
            annotated_image=image_b64
        )
        
    except PayloadTooLargeError as e:
        logger.warning("api.analyze.payload_too_large", extra={"error_msg": str(e)})
        raise HTTPException(status_code=413, detail=str(e))
        
    except FileCorruptionError as e:
        logger.warning("api.analyze.file_corruption", extra={"error_msg": str(e)})
        raise HTTPException(status_code=400, detail=str(e))
        
    except HVACDomainError as e:
        logger.error("api.analyze.domain_error", extra={"error_msg": str(e)})
        raise HTTPException(status_code=422, detail=str(e))
        
    except Exception as e:
        logger.error("api.analyze.unexpected_error", extra={
            "error_msg": str(e),
            "error_type": type(e).__name__
        })
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "hvac-duct-analyzer"}


@router.get("/config")
def get_config():
    """Get current CV configuration parameters."""
    from app.core.config import settings
    
    return {
        "pdf_dpi": settings.HVAC_PDF_DPI,
        "preprocessing": {
            "gaussian_kernel": settings.CV_OTSU_GAUSSIAN_BLUR_KERNEL,
            "morph_kernel": (settings.CV_MORPH_KERNEL_WIDTH, settings.CV_MORPH_KERNEL_HEIGHT),
            "morph_iterations": settings.CV_MORPH_ITERATIONS
        },
        "hough_lines": {
            "threshold": settings.CV_HOUGH_THRESHOLD,
            "min_line_length": settings.CV_HOUGH_MIN_LINE_LENGTH,
            "max_line_gap": settings.CV_HOUGH_MAX_LINE_GAP
        },
        "duct_detection": {
            "angle_tolerance": settings.CV_PARALLEL_ANGLE_TOLERANCE,
            "min_width_px": settings.CV_DUCT_MIN_WIDTH_PX,
            "max_width_px": settings.CV_DUCT_MAX_WIDTH_PX
        },
        "ocr": {
            "min_confidence": settings.OCR_MIN_CONFIDENCE,
            "psm_mode": settings.OCR_PSM_MODE
        },
        "rendering": {
            "line_thickness": settings.RENDER_DUCT_LINE_THICKNESS,
            "overlay_alpha": settings.RENDER_OVERLAY_ALPHA
        }
    }
