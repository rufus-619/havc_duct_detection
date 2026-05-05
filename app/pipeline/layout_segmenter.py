"""
Layout Segmentation Module (Module 0).

Uses Directional Morphological Opening to extract the structural grid
and isolate the Core Diagram from title blocks and notes.

Strategy:
1. Adaptive Binarization (handle uneven scans)
2. Horizontal MORPH_OPEN (destroy text, keep horizontal borders)
3. Vertical MORPH_OPEN (destroy text, keep vertical borders)
4. Combine and seal gaps
5. Find contour bounding boxes
6. Filter by area and position to select Core Diagram
"""
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger("hvac_analyzer.pipeline")


@dataclass
class LayoutRegion:
    """Represents a detected layout region."""
    x: int
    y: int
    width: int
    height: int
    region_type: str
    confidence: float
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class SegmentationResult:
    """Result of layout segmentation."""
    core_diagram: np.ndarray  # ROI image
    metadata_panels: List[Tuple[str, np.ndarray]]  # [(region_type, image), ...]
    regions: List[LayoutRegion]  # All detected regions
    scale_roi: Optional[np.ndarray] = None  # Region likely containing scale info


class LayoutSegmenter:
    """
    Segments blueprint layout using Directional Morphological Opening.
    
    Isolates the structural grid by mathematically destroying text
    and angled ductwork, leaving only horizontal/vertical borders.
    """
    
    def __init__(self):
        # Area thresholds for panel filtering
        self.min_panel_ratio = 0.05   # Minimum 5% of image
        self.max_panel_ratio = 0.98   # Maximum 98% of image (exclude outer margin)
        
        # Grid sealing parameters
        self.seal_kernel_size = (5, 5)
        self.seal_iterations = 2
    
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Segment blueprint layout using directional morphology.
        
        Args:
            image: Input BGR image
            
        Returns:
            SegmentationResult with core diagram and metadata panels
        """
        logger.info("pipeline.segmentation.started", extra={
            "image_shape": image.shape
        })
        
        try:
            height, width = image.shape[:2]
            total_area = width * height
            
            logger.debug(f"segment: image={width}x{height}, total_area={total_area}")
            
            # Step 1: Extract structural grid using directional morphology
            grid_mask = self._extract_structural_grid(image)
            logger.debug(f"segment: grid_mask extracted, nonzero={cv2.countNonZero(grid_mask)}")
            
            # Step 2: Find contour bounding boxes from the grid
            regions = self._find_grid_panels(grid_mask, width, height)
            logger.debug(f"segment: found {len(regions)} raw regions")
            
            # Step 3: Filter by area and select core diagram
            valid_regions = self._filter_panels_by_area(regions, total_area)
            logger.debug(f"segment: {len(valid_regions)} valid regions after filtering")
            for i, r in enumerate(valid_regions[:5]):
                logger.debug(f"  region {i}: {r.width}x{r.height} at ({r.x},{r.y})")
            
            # Step 4: Classify and select core diagram using content density
            classified = self._classify_and_select_core(valid_regions, image, width, height)
            logger.debug(f"segment: selected region {classified.width}x{classified.height}")
            
            if classified is None:
                logger.warning("pipeline.segmentation.no_valid_core")
                # Fallback to full image
                classified = LayoutRegion(
                    x=0, y=0, width=width, height=height,
                    region_type="core_diagram", confidence=0.5
                )
            
            # Step 5: Extract core diagram
            core_image = self._extract_region(image, classified)
            
            logger.info("pipeline.segmentation.completed", extra={
                "core_shape": core_image.shape,
                "regions_found": len(valid_regions)
            })
            
            return SegmentationResult(
                core_diagram=core_image,
                metadata_panels=[],  # Simplified for now
                regions=[classified],
                scale_roi=None
            )
            
        except Exception as e:
            logger.error("pipeline.segmentation.failed", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Fallback
            height, width = image.shape[:2]
            return SegmentationResult(
                core_diagram=image.copy(),
                metadata_panels=[],
                regions=[LayoutRegion(
                    x=0, y=0, width=width, height=height,
                    region_type="core_diagram", confidence=0.5
                )],
                scale_roi=None
            )
    
    def _extract_structural_grid(self, image: np.ndarray) -> np.ndarray:
        """
        Extract structural grid using Directional Morphological Opening.
        
        This destroys text and angled ductwork while preserving horizontal/vertical borders.
        """
        height, width = image.shape[:2]
        
        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: Adaptive Binarization (invert so lines are white)
        binary = cv2.adaptiveThreshold(
            cv2.bitwise_not(gray), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, -2
        )
        
        # Step 3: Morphological opening for thick grid lines
        h_size = max(width // 60, 80)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        
        v_size = max(height // 60, 80)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        
        grid_mask = cv2.add(h_lines, v_lines)
        
        # Step 4: HoughLinesP for thinner internal grid dividers
        # These are harder to catch with morphology alone
        hough_lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=max(width, height) // 8,  # Shorter lines
            maxLineGap=100
        )
        
        if hough_lines is not None:
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Only near-horizontal or near-vertical (grid dividers)
                is_grid = angle < 20 or angle > 160 or (70 < angle < 110)
                if is_grid:
                    cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 7)
        
        # Step 5: Dilate to seal gaps
        seal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        grid_mask = cv2.dilate(grid_mask, seal_kernel, iterations=3)
        
        return grid_mask
    
    def _find_grid_panels(
        self,
        grid_mask: np.ndarray,
        img_width: int,
        img_height: int
    ) -> List[LayoutRegion]:
        """
        Find panel regions bounded by the structural grid.
        Uses contour detection on the inverted grid mask.
        """
        # Invert: grid lines become black (0), panels become white (255)
        inverted = cv2.bitwise_not(grid_mask)
        
        # Find contours of enclosed regions
        contours, _ = cv2.findContours(
            inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        
        for cnt in contours:
            # Get bounding rectangle (do NOT use approxPolyDP - too fragile)
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            regions.append(LayoutRegion(
                x=x, y=y, width=w, height=h,
                region_type="unknown",
                confidence=0.5
            ))
        
        return regions
    
    def _filter_panels_by_area(
        self,
        regions: List[LayoutRegion],
        total_area: int
    ) -> List[LayoutRegion]:
        """
        Filter panels by area constraints.
        - Discard > 98% (outer page margin)
        - Discard < 5% (noise/text blobs)
        """
        valid = []
        
        for r in regions:
            area_ratio = r.area / total_area
            
            # Discard outer margin (> 98%)
            if area_ratio > self.max_panel_ratio:
                continue
            
            # Discard noise (< 5%)
            if area_ratio < self.min_panel_ratio:
                continue
            
            # Update confidence based on area
            r.confidence = min(1.0, area_ratio * 2)
            valid.append(r)
        
        return valid
    
    def _classify_and_select_core(
        self,
        regions: List[LayoutRegion],
        image: np.ndarray,
        img_width: int,
        img_height: int
    ) -> Optional[LayoutRegion]:
        """
        Select the Core Diagram using content density analysis.
        
        The mechanical drawing has high line density (ductwork),
        while title blocks and notes have mostly text/tables.
        """
        if not regions:
            return None
        
        total_area = img_width * img_height
        
        # Score each region by content density
        best_region = None
        best_score = 0
        
        for r in regions:
            area_ratio = r.area / total_area
            
            # Skip regions that are too small or too large
            if area_ratio < 0.15 or area_ratio > 0.95:
                continue
            
            # Extract region
            x1, y1, x2, y2 = r.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)
            region_img = image[y1:y2, x1:x2]
            
            if region_img.size == 0:
                continue
            
            # Calculate content density (non-white pixels)
            gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            content_pixels = cv2.countNonZero(binary)
            content_ratio = content_pixels / r.area
            
            # Position analysis
            cx, cy = r.center
            norm_x = cx / img_width
            norm_y = cy / img_height
            
            # Mechanical diagram scoring:
            # - Must be in upper 70% of page (y < 0.70)
            # - Prefer top-left to center area
            # - Heavily penalize bottom regions (title block, notes)
            
            if norm_y > 0.70:  # Bottom 30% = title block/notes area
                position_score = -50  # Heavy penalty
            else:
                # Reward upper regions, especially top-left quadrant
                position_score = (0.5 - norm_y) * 20 + (0.5 - norm_x) * 10
            
            score = content_ratio * 100 + position_score
            
            logger.debug(f"Region {r.width}x{r.height} at ({r.x},{r.y}): "
                        f"area={area_ratio*100:.1f}%, content={content_ratio*100:.2f}%, score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_region = r
        
        if best_region:
            best_region.region_type = "core_diagram"
            logger.info("pipeline.segmentation.selected_by_content", extra={
                "area_ratio": best_region.area / total_area,
                "score": best_score
            })
            return best_region
        
        # Fallback: position-based crop
        target_width = int(img_width * 0.70)
        target_height = int(img_height * 0.65)
        
        return LayoutRegion(
            x=0, y=0,
            width=target_width,
            height=target_height,
            region_type="core_diagram",
            confidence=0.5
        )
    
    def _extract_region(self, image: np.ndarray, region: LayoutRegion) -> np.ndarray:
        """Extract image region defined by LayoutRegion."""
        x1, y1, x2, y2 = region.bbox
        
        # Ensure bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        return image[y1:y2, x1:x2].copy()
    
    def debug_visualization(
        self,
        image: np.ndarray,
        regions: List[LayoutRegion],
        output_path: str
    ) -> None:
        """Create debug visualization of segmentation."""
        debug_img = image.copy()
        
        colors = {
            "core_diagram": (0, 255, 0),
            "title_block": (255, 0, 0),
            "notes": (0, 0, 255),
            "legend": (255, 255, 0),
            "unknown": (128, 128, 128)
        }
        
        for region in regions:
            color = colors.get(region.region_type, (128, 128, 128))
            x1, y1, x2, y2 = region.bbox
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 3)
            label = f"{region.region_type} ({region.confidence:.2f})"
            cv2.putText(debug_img, label, (x1 + 5, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imwrite(output_path, debug_img)
        logger.debug("pipeline.segmentation.debug_saved", extra={"path": output_path})
