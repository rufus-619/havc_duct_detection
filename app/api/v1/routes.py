import uuid
import time
import base64
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.v1.schemas import DetectionResponse, ImageMetadata, PipelineTelemetry
from app.core.logging import logger, trace_id_var
from app.utils.file_normalizer import normalize
from app.services.ocr_service import TesseractOCRAdapter
from app.services.extractor import filter_domain_text, parse_dimension, parse_pressure_class
from app.services.duct_detector import detect_duct_for_label
from app.services.spatial import associate_pressure_class, deduplicate_segments
from app.services.scale_extractor import extract_scale, measure_length
from app.services.ai_verify import verify_segments
from app.services.annotator import render_annotated_image
from app.models.domain import DuctSegment

router = APIRouter()
ocr_adapter = TesseractOCRAdapter()

@router.post("/analyze-blueprint", response_model=DetectionResponse)
async def analyze_blueprint(request: Request, file: UploadFile = File(...)):
    trace_id = trace_id_var.get()
    start_total = time.time()
    
    durations = {}
    telemetry: Dict[str, Any] = {}
    
    logger.info("pipeline_started", filename=file.filename)
    
    # 1. File Ingestion
    img_bgr, orig_w, orig_h = await normalize(file)
    
    # 2. OCR Adapter
    start_ocr = time.time()
    ocr_results = ocr_adapter.extract(img_bgr)
    durations["ocr"] = int((time.time() - start_ocr) * 1000)
    
    if not ocr_results:
        from app.core.exceptions import UnreadableBlueprintError
        raise UnreadableBlueprintError("Could not extract text from the drawing. Ensure the file is >300 DPI and correctly oriented.")
    
    # DEBUG: Log all OCR results
    print(f"[DEBUG] OCR found {len(ocr_results)} text blocks:")
    for r in ocr_results[:20]:  # Limit to first 20
        print(f"  - '{r.text}' (conf: {r.confidence:.1f}) at ({r.geometry.x}, {r.geometry.y})")
    
    avg_conf = sum(r.confidence for r in ocr_results) / len(ocr_results)
    telemetry["ocr_avg_confidence"] = round(avg_conf, 2)
    telemetry["total_ocr_blocks"] = len(ocr_results)
    
    # 3. Semantic Parser
    start_extractor = time.time()
    domain_labels = filter_domain_text(ocr_results)
    durations["extractor"] = int((time.time() - start_extractor) * 1000)
    
    # DEBUG: Log domain labels found
    print(f"[DEBUG] Domain labels found: {len(domain_labels)}")
    for lbl in domain_labels[:10]:
        print(f"  - '{lbl.text}' at ({lbl.geometry.x}, {lbl.geometry.y})")
    
    if not domain_labels:
        from app.core.exceptions import UnreadableBlueprintError
        raise UnreadableBlueprintError("Could not find any HVAC domain labels in the drawing.")
        
    telemetry["domain_labels_matched"] = len(domain_labels)
    telemetry["semantic_match_rate"] = round(len(domain_labels) / len(ocr_results), 2)
    
    # Extract scale
    start_scale = time.time()
    scale = extract_scale(img_bgr, ocr_adapter)
    durations["scale"] = int((time.time() - start_scale) * 1000)
    
    # 4. Localized Duct Detector
    start_detector = time.time()
    segments = []
    fallback_count = 0
    print(f"[DEBUG] Processing {len(domain_labels)} domain labels for duct detection...")
    for idx, label in enumerate(domain_labels):
        dims = parse_dimension(label.text)
        print(f"[DEBUG] Label {idx}: '{label.text}' -> dims: {dims}")
        if not dims:
            continue
            
        w_in, h_in, d_in = dims
        duct_geom, duct_lines, det_conf, fallback = detect_duct_for_label(img_bgr, label)
        print(f"[DEBUG]   -> duct_geom: {duct_geom}, lines: {len(duct_lines)}, fallback: {fallback}")
        
        # Determine simple dummy duct type for now, actual implementation in duct_detector
        from app.models.domain import DuctType, PressureClass
        # Calculate length based on geometry
        pt1 = (duct_geom.x, duct_geom.y + duct_geom.height // 2)
        pt2 = (duct_geom.x + duct_geom.width, duct_geom.y + duct_geom.height // 2)
        length_ft = measure_length(pt1, pt2, scale)
        
        if fallback:
            fallback_count += 1
            
        seg = DuctSegment(
            id=f"duct_{uuid.uuid4().hex[:6]}",
            duct_type=DuctType.UNKNOWN, # Will be set by detector
            pressure_class=PressureClass.UNKNOWN,
            dims_str=label.text,
            width_in=w_in,
            height_in=h_in,
            diameter_in=d_in,
            length_ft=length_ft,
            label_geometry=label.geometry,
            duct_geometry=duct_geom,
            duct_lines=duct_lines,
            detection_confidence=det_conf,
            fallback_used=fallback,
            trace_id=trace_id
        )
        # refine duct_type if possible by detector color check (mocked for now in detector)
        # We will handle it fully in detector, just updating here.
        segments.append(seg)
        
    durations["detector"] = int((time.time() - start_detector) * 1000)
    
    # 5. Spatial Engine
    start_spatial = time.time()
    segments = associate_pressure_class(segments, ocr_results)
    segments = deduplicate_segments(segments)
    durations["spatial"] = int((time.time() - start_spatial) * 1000)
    
    # 6. AI Verify
    start_ai = time.time()
    segments = await verify_segments(img_bgr, segments)
    durations["ai_verify"] = int((time.time() - start_ai) * 1000)
    
    # 7. Annotator
    start_annotator = time.time()
    annotated_png = render_annotated_image(img_bgr, segments)
    durations["annotator"] = int((time.time() - start_annotator) * 1000)
    
    # Encode image as base64 for response
    annotated_image_b64 = base64.b64encode(annotated_png).decode('utf-8')
    
    durations["total"] = int((time.time() - start_total) * 1000)
    
    telemetry["segments_detected"] = len(segments)
    telemetry["fallback_count"] = fallback_count
    telemetry["fallback_rate"] = round(fallback_count / max(1, len(segments)), 2)
    telemetry["ai_verify_invocations"] = sum(1 for s in segments if s.detection_confidence is not None and s.detection_confidence < 0.6) + fallback_count
    telemetry["duration_ms"] = durations
    
    logger.info("pipeline_complete", **telemetry)
    
    # Critical Silent Failure Alert
    if telemetry.get("ocr_avg_confidence", 0) > 0.80 and len(segments) == 0:
        logger.warning("high_confidence_zero_yield")
        
    # Optional: We return image as base64 in the response or separate endpoint. For simplicity, omitting or providing it.
    
    return DetectionResponse(
        trace_id=trace_id,
        message=f"Successfully extracted {len(segments)} duct segments.",
        image_metadata=ImageMetadata(original_width=orig_w, original_height=orig_h),
        pipeline_telemetry=PipelineTelemetry(**telemetry),
        data=segments,
        annotated_image=annotated_image_b64
    )
