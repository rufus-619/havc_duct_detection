"""
Pipeline orchestrator.
Coordinates all pipeline stages: ingestion → preprocessing → geometric detection → 
OCR extraction → spatial mapping → annotation rendering.
"""
import logging
import os
import uuid
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2

from app.core.config import settings
from app.pipeline.ingestion import IngestionService
from app.pipeline.preprocessing import PreprocessingService
from app.pipeline.layout_segmenter import LayoutSegmenter
from app.pipeline.geometric import GeometricDetectionService
from app.pipeline.region_based_detection import RegionBasedDetectionService
from app.pipeline.vector_extraction import VectorExtractionService
from app.pipeline.vector_duct_detector import VectorDuctDetectionService, debug_draw_ducts
from app.pipeline.extraction import ExtractionService, TextBlock
from app.pipeline.mapping import SpatialMappingService, MappedDuct
from app.pipeline.annotation import AnnotationService

logger = logging.getLogger("hvac_analyzer.pipeline")

DEBUG_IMAGES_DIR = Path(__file__).resolve().parents[2] / "debug_images"


def _save_debug(name: str, image: np.ndarray) -> None:
    """Save a debug image to debug_images/ folder. Skips silently on error."""
    try:
        DEBUG_IMAGES_DIR.mkdir(exist_ok=True)
        path = DEBUG_IMAGES_DIR / f"{name}.png"
        # If grayscale or binary, save as-is; otherwise save BGR
        cv2.imwrite(str(path), image)
        logger.info(f"debug.image_saved", extra={"path": str(path), "shape": image.shape})
    except Exception as e:
        logger.warning(f"debug.image_save_failed", extra={"name": name, "error": str(e)})


class PipelineOrchestrator:
    """
    Coordinates the complete HVAC drawing processing pipeline.
    
    Pipeline Flow:
    1. Ingestion: PDF/Image → BGR array
    2. Layout Segmentation: Detect structural borders → Extract core diagram ROI
    3. Preprocessing: Isolate duct lines (Otsu + Morphology)
    4. Geometric Detection: HoughLinesP → Parallel line grouping → Duct segments
    5. OCR Extraction: Text detection → Dimension parsing
    6. Spatial Mapping: Map dimensions to nearest duct segments
    7. Annotation: Render highlighted ducts with dimension labels
    """
    
    def __init__(self):
        self.ingestion = IngestionService()
        self.preprocessing = PreprocessingService()
        self.layout_segmenter = LayoutSegmenter()
        self.geometric = GeometricDetectionService()          # legacy fallback
        self.region_detector = RegionBasedDetectionService()  # raster region pipeline
        self.vector_extractor = VectorExtractionService()     # PDF vector primitives
        self.vector_detector = VectorDuctDetectionService()   # vector duct pairing
        self.extraction = ExtractionService()
        self.mapping = SpatialMappingService()
        self.annotation = AnnotationService()
    
    def process(
        self,
        file_bytes: bytes,
        content_type: str,
        filename: str
    ) -> Tuple[bytes, dict]:
        """
        Execute complete pipeline.
        
        Args:
            file_bytes: Raw file data
            content_type: MIME type
            filename: Original filename
            
        Returns:
            Tuple of (annotated_image_bytes, metadata_dict)
        """
        trace_id = str(uuid.uuid4())
        logger.info("pipeline.orchestrator.started", extra={
            "trace_id": trace_id,
            "file_name": filename
        })
        
        try:
            # ----------------------------------------------------------
            # Vector pipeline (PDF-only, takes precedence)
            # ----------------------------------------------------------
            vector_result = None
            if (settings.USE_VECTOR_PIPELINE
                    and content_type == "application/pdf"):
                try:
                    vector_result = self._run_vector_pipeline(
                        file_bytes, trace_id
                    )
                except Exception as ve:
                    logger.warning("pipeline.orchestrator.vector_failed", extra={
                        "trace_id": trace_id,
                        "error_msg": str(ve),
                        "error_type": type(ve).__name__,
                    })
                    vector_result = None

            if vector_result is not None:
                output_bytes, metadata = vector_result
                metadata.update({"trace_id": trace_id, "filename": filename})
                logger.info("pipeline.orchestrator.completed", extra={
                    "trace_id": trace_id,
                    "ducts": metadata.get("processing_stats", {}).get("ducts_detected", 0),
                    "source": "vector",
                })
                return output_bytes, metadata

            # ----------------------------------------------------------
            # Raster pipeline (fallback / images / scanned PDFs)
            # ----------------------------------------------------------

            # Stage 1: Ingestion
            raw_image, width, height = self.ingestion.ingest(
                file_bytes, content_type, filename
            )
            _save_debug("01_raw_image", raw_image)
            
            # Stage 2: Layout Segmentation
            # Extract core diagram and metadata panels
            segmentation = self.layout_segmenter.segment(raw_image)
            core_image = segmentation.core_diagram
            _save_debug("02_core_diagram", core_image)
            
            logger.info("pipeline.orchestrator.segmentation", extra={
                "trace_id": trace_id,
                "core_shape": core_image.shape,
                "metadata_panels": len(segmentation.metadata_panels)
            })
            
            # Stage 3: Preprocess core diagram for duct detection
            binary, grayscale = self.preprocessing.preprocess(core_image)
            _save_debug("03_grayscale", grayscale)
            _save_debug("04_binary", binary)
            
            # Stage 4: OCR first - get all text blocks including dimension labels
            # We need these BEFORE geometric detection to validate duct candidates
            text_blocks = self.extraction.extract(core_image)
            dimensions = self.extraction.parse_dimensions(text_blocks)
            pressure_labels = self.extraction.find_pressure_labels(text_blocks)
            scale = self.extraction.extract_scale(
                text_blocks,
                core_image.shape[1],
                core_image.shape[0]
            )
            logger.info("pipeline.orchestrator.extraction", extra={
                "trace_id": trace_id,
                "text_blocks": len(text_blocks),
                "dimensions": len(dimensions),
                "pressure_labels": len(pressure_labels),
                "scale": scale
            })
            
            # Stage 5: Duct detection — region-based (new) or legacy geometric
            if settings.USE_REGION_BASED_PIPELINE:
                logger.info("pipeline.orchestrator.using_region_pipeline")
                ducts = self.region_detector.detect(
                    binary,
                    grayscale=grayscale,
                    text_blocks=text_blocks,
                )
                detector = self.region_detector
            else:
                logger.info("pipeline.orchestrator.using_legacy_pipeline")
                binary_clean = self.preprocessing.inpaint_text(binary, text_blocks)
                _save_debug("04e_binary_inpainted", binary_clean)
                ducts = self.geometric.detect(
                    binary_clean,
                    grayscale=grayscale,
                    color_mask=None,
                    text_blocks=text_blocks,
                )
                detector = self.geometric

            if not ducts:
                logger.warning("pipeline.orchestrator.no_ducts_detected", extra={
                    "trace_id": trace_id,
                    "core_shape": core_image.shape
                })
            else:
                logger.info("pipeline.orchestrator.ducts_detected", extra={
                    "trace_id": trace_id,
                    "duct_count": len(ducts),
                    "source": "region" if settings.USE_REGION_BASED_PIPELINE else "geometric"
                })

            # Stage 5b: Classify ducts by colour (blue=supply, red=return)
            blue_mask = self.preprocessing.create_color_mask(core_image, target_color="blue")
            red_mask = self.preprocessing.create_color_mask(core_image, target_color="red")

            supply_count = 0
            return_count = 0
            for duct in ducts:
                color_type = detector.classify_duct_color(duct, blue_mask, red_mask)
                duct.duct_type = color_type
                if color_type == "supply":
                    supply_count += 1
                elif color_type == "return":
                    return_count += 1
            
            logger.info("pipeline.orchestrator.color_classification", extra={
                "trace_id": trace_id,
                "supply_ducts": supply_count,
                "return_ducts": return_count,
                "unknown_ducts": len(ducts) - supply_count - return_count,
                "total": len(ducts)
            })
            
            # Stage 6: Convert DuctRegion → DuctSegment if using region pipeline
            if settings.USE_REGION_BASED_PIPELINE:
                from app.pipeline.region_based_detection import DuctRegion
                duct_segments = [
                    d.to_duct_segment() if isinstance(d, DuctRegion) else d
                    for d in ducts
                ]
            else:
                duct_segments = ducts

            # Stage 6: Spatial Mapping
            dimension_mapping = self.mapping.map_dimensions(duct_segments, dimensions)
            pressure_mapping = self.mapping.map_pressure_classes(duct_segments, pressure_labels)
            
            # Build final mapped ducts
            mapped_ducts = []
            for i, (duct, dim) in enumerate(dimension_mapping):
                _, pressure = pressure_mapping[i]
                
                # Calculate real length
                real_length = self.mapping.calculate_real_length(duct, scale)
                
                mapped_ducts.append(MappedDuct(
                    duct=duct,
                    dimension=dim,
                    pressure_class=pressure,
                    real_length_ft=real_length,
                    id=f"duct_{i+1:03d}"
                ))
            
            # Stage 7: Annotation (on core diagram only)
            annotated_image = self.annotation.render(core_image, mapped_ducts)
            _save_debug("05_annotated", annotated_image)
            output_bytes = self.annotation.encode_to_bytes(annotated_image, "png")
            
            # Build metadata
            metadata = {
                "trace_id": trace_id,
                "filename": filename,
                "image_dimensions": {"width": width, "height": height},
                "processing_stats": {
                    "ducts_detected": len(ducts),
                    "dimensions_found": len(dimensions),
                    "dimensions_mapped": sum(1 for m in mapped_ducts if m.dimension),
                    "scale_px_per_foot": scale
                },
                "ducts": [
                    {
                        "id": m.id,
                        "type": m.duct.duct_type,
                        "width_px": m.duct.width_px,
                        "length_px": m.duct.length_px,
                        "real_length_ft": m.real_length_ft,
                        "dimension": m.dimension.dims_str if m.dimension else None,
                        "pressure_class": m.pressure_class,
                        "center": {"x": m.duct.center[0], "y": m.duct.center[1]}
                    }
                    for m in mapped_ducts
                ]
            }
            
            logger.info("pipeline.orchestrator.completed", extra={
                "trace_id": trace_id,
                "ducts": len(ducts),
                "dimensions_mapped": metadata["processing_stats"]["dimensions_mapped"]
            })
            
            return output_bytes, metadata
            
        except Exception as e:
            logger.error("pipeline.orchestrator.failed", extra={
                "trace_id": trace_id,
                "error_msg": str(e),
                "error_type": type(e).__name__
            })
            raise

    # ------------------------------------------------------------------
    # Vector pipeline runner
    # ------------------------------------------------------------------
    def _run_vector_pipeline(
        self,
        file_bytes: bytes,
        trace_id: str,
    ) -> Optional[Tuple[bytes, dict]]:
        """
        Try the PDF-vector pipeline.  Returns (annotated_bytes, metadata) on
        success, or None to signal the caller should fall back to raster.
        """
        # Stage 1 — Vector primitive extraction
        vpage = self.vector_extractor.open(file_bytes)

        if vpage.primitive_count < settings.VEC_MIN_PRIMITIVES_FOR_VECTOR:
            logger.warning("vector.too_few_primitives", extra={
                "trace_id": trace_id,
                "count": vpage.primitive_count,
            })
            return None

        # Render PDF page to BGR (for annotation + color sampling + segmentation)
        bgr_image, scale = self.vector_extractor.render_page_to_bgr(
            vpage.page, dpi=settings.HVAC_PDF_DPI,
        )
        _save_debug("v00_page_render", bgr_image)
        height, width = bgr_image.shape[:2]

        # Stage 1b — Layout segmentation: isolate the core diagram bbox
        # to drop title blocks, notes panels, and drawing borders.
        roi_pts = None
        try:
            seg = self.layout_segmenter.segment(bgr_image)
            if seg.regions:
                core_region = seg.regions[0]
                x0_px, y0_px, x1_px, y1_px = core_region.bbox
                # Convert pixel bbox -> PDF point bbox
                roi_pts = (
                    x0_px / scale, y0_px / scale,
                    x1_px / scale, y1_px / scale,
                )
                _save_debug("v01_core_diagram", seg.core_diagram)
                logger.info("vector.layout_roi", extra={
                    "trace_id": trace_id,
                    "roi_px": (x0_px, y0_px, x1_px, y1_px),
                    "roi_pts": roi_pts,
                })
        except Exception as le:
            logger.warning("vector.layout_segment_failed", extra={
                "trace_id": trace_id, "error_msg": str(le),
            })

        # Stage 2 — Vector duct detection (ROI-aware)
        ducts = self.vector_detector.detect(vpage, scale, roi_pts=roi_pts)
        debug_draw_ducts(ducts, bgr_image.shape, "v04_merged_ducts.png", background=bgr_image)

        if not ducts:
            logger.warning("vector.no_ducts", extra={"trace_id": trace_id})
            return None

        # Stage 3 — Build TextBlocks from vector text spans (no OCR needed)
        text_blocks: List[TextBlock] = []
        for t in vpage.texts:
            text_blocks.append(TextBlock(
                text=t.text,
                x=int(t.x * scale),
                y=int(t.y * scale),
                width=int(t.width * scale),
                height=int(t.height * scale),
                confidence=1.0,  # vector text is exact
            ))

        # Stage 4 — Reuse existing dimension/scale parsing
        dimensions = self.extraction.parse_dimensions(text_blocks)
        pressure_labels = self.extraction.find_pressure_labels(text_blocks)
        drawing_scale = self.extraction.extract_scale(text_blocks, width, height)

        logger.info("vector.text_parsing", extra={
            "trace_id": trace_id,
            "text_blocks": len(text_blocks),
            "dimensions": len(dimensions),
            "pressure_labels": len(pressure_labels),
            "scale": drawing_scale,
        })

        # Stage 5 — Color classification (reuse raster colour masks)
        blue_mask = self.preprocessing.create_color_mask(bgr_image, target_color="blue")
        red_mask = self.preprocessing.create_color_mask(bgr_image, target_color="red")
        supply_count = return_count = 0
        for d in ducts:
            d.duct_type = self.geometric.classify_duct_color(d, blue_mask, red_mask)
            if d.duct_type == "supply":
                supply_count += 1
            elif d.duct_type == "return":
                return_count += 1

        # Stage 6 — Spatial mapping (reused unchanged)
        dimension_mapping = self.mapping.map_dimensions(ducts, dimensions)
        pressure_mapping = self.mapping.map_pressure_classes(ducts, pressure_labels)

        mapped_ducts: List[MappedDuct] = []
        for i, (duct, dim) in enumerate(dimension_mapping):
            _, pressure = pressure_mapping[i]
            real_length = self.mapping.calculate_real_length(duct, drawing_scale)
            mapped_ducts.append(MappedDuct(
                duct=duct,
                dimension=dim,
                pressure_class=pressure,
                real_length_ft=real_length,
                id=f"duct_{i+1:03d}",
            ))

        # Stage 7 — Annotation (reused unchanged)
        annotated = self.annotation.render(bgr_image, mapped_ducts)
        _save_debug("v06_annotated", annotated)
        output_bytes = self.annotation.encode_to_bytes(annotated, "png")

        metadata = {
            "image_dimensions": {"width": width, "height": height},
            "source": "vector",
            "processing_stats": {
                "ducts_detected": len(ducts),
                "dimensions_found": len(dimensions),
                "dimensions_mapped": sum(1 for m in mapped_ducts if m.dimension),
                "scale_px_per_foot": drawing_scale,
                "supply_ducts": supply_count,
                "return_ducts": return_count,
            },
            "ducts": [
                {
                    "id": m.id,
                    "type": m.duct.duct_type,
                    "width_px": m.duct.width_px,
                    "length_px": m.duct.length_px,
                    "real_length_ft": m.real_length_ft,
                    "dimension": m.dimension.dims_str if m.dimension else None,
                    "pressure_class": m.pressure_class,
                    "center": {"x": m.duct.center[0], "y": m.duct.center[1]},
                }
                for m in mapped_ducts
            ],
        }
        return output_bytes, metadata
