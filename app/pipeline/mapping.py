"""
Spatial mapping module.
Maps OCR-extracted dimension labels to detected duct segments
based on geometric proximity.
"""
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from app.core.config import settings
from app.pipeline.geometric import DuctSegment
from app.pipeline.extraction import DimensionLabel, TextBlock

logger = logging.getLogger("hvac_analyzer.pipeline")


@dataclass
class MappedDuct:
    """A duct segment with associated dimension and metadata."""
    duct: DuctSegment
    dimension: Optional[DimensionLabel]
    pressure_class: Optional[str]
    real_length_ft: float
    id: str


class SpatialMappingService:
    """Maps text labels to duct segments based on spatial proximity."""
    
    def __init__(self):
        self.max_distance = settings.SPATIAL_MAX_DISTANCE_PX
        self.search_radius = settings.SPATIAL_SEARCH_RADIUS_PX
    
    def map_dimensions(
        self,
        ducts: List[DuctSegment],
        dimensions: List[DimensionLabel]
    ) -> List[Tuple[DuctSegment, Optional[DimensionLabel]]]:
        """
        Associate dimension labels with duct segments.
        
        Strategy:
        1. For each dimension label, find nearest duct segment
        2. Associate if within max_distance threshold
        3. Handle conflicts (one dimension claimed by multiple ducts)
        
        Args:
            ducts: Detected duct segments
            dimensions: Parsed dimension labels
            
        Returns:
            List of (duct, dimension) tuples (dimension may be None)
        """
        logger.debug("pipeline.mapping.started", extra={
            "ducts": len(ducts),
            "dimensions": len(dimensions)
        })
        
        if not ducts:
            logger.warning("pipeline.mapping.no_ducts")
            return []
        
        if not dimensions:
            logger.warning("pipeline.mapping.no_dimensions")
            # Return all ducts with None dimensions
            return [(duct, None) for duct in ducts]
        
        # Build distance matrix: dimensions x ducts
        distance_matrix = np.zeros((len(dimensions), len(ducts)))
        
        for i, dim in enumerate(dimensions):
            for j, duct in enumerate(ducts):
                distance = self._point_to_duct_distance(dim.center, duct)
                distance_matrix[i, j] = distance
        
        # Greedy matching: assign closest pairs within threshold
        assignments = {}  # duct_idx -> dim_idx
        assigned_dims = set()
        
        # Sort all possible assignments by distance
        possible_assignments = []
        for i in range(len(dimensions)):
            for j in range(len(ducts)):
                if distance_matrix[i, j] <= self.max_distance:
                    possible_assignments.append((distance_matrix[i, j], i, j))
        
        possible_assignments.sort()  # Sort by distance
        
        for distance, dim_idx, duct_idx in possible_assignments:
            # Skip if dimension already assigned
            if dim_idx in assigned_dims:
                continue
            
            # Skip if duct already has a closer assignment
            if duct_idx in assignments:
                existing_dim_idx = assignments[duct_idx]
                if distance_matrix[existing_dim_idx, duct_idx] <= distance:
                    continue
                # Otherwise, reassign
                assigned_dims.remove(existing_dim_idx)
            
            # Make assignment
            assignments[duct_idx] = dim_idx
            assigned_dims.add(dim_idx)
        
        # Build result list
        results = []
        for j, duct in enumerate(ducts):
            if j in assignments:
                dim_idx = assignments[j]
                results.append((duct, dimensions[dim_idx]))
            else:
                results.append((duct, None))
        
        # Log mapping statistics
        matched = sum(1 for _, dim in results if dim is not None)
        logger.info("pipeline.mapping.completed", extra={
            "ducts": len(ducts),
            "dimensions": len(dimensions),
            "matched": matched,
            "unmatched": len(ducts) - matched
        })
        
        return results
    
    def map_pressure_classes(
        self,
        ducts: List[DuctSegment],
        pressure_labels: List[Tuple[str, TextBlock]]
    ) -> List[Tuple[DuctSegment, Optional[str]]]:
        """
        Associate pressure class labels with duct segments.
        
        Args:
            ducts: Detected duct segments
            pressure_labels: List of (pressure_class, text_block) tuples
            
        Returns:
            List of (duct, pressure_class) tuples
        """
        if not pressure_labels:
            return [(duct, None) for duct in ducts]
        
        results = []
        
        for duct in ducts:
            closest_pressure = None
            min_distance = float('inf')
            
            for pressure, text_block in pressure_labels:
                distance = self._point_to_duct_distance(text_block.center, duct)
                if distance < min_distance and distance <= self.max_distance:
                    min_distance = distance
                    closest_pressure = pressure
            
            results.append((duct, closest_pressure))
        
        return results
    
    def _point_to_duct_distance(
        self, 
        point: Tuple[float, float], 
        duct: DuctSegment
    ) -> float:
        """
        Calculate minimum distance from point to duct segment.
        
        Strategy: Calculate distance to both walls and take minimum.
        """
        # Distance to wall1
        dist1 = self._point_to_line_distance(
            point, 
            (duct.wall1.x1, duct.wall1.y1),
            (duct.wall1.x2, duct.wall1.y2)
        )
        
        # Distance to wall2
        dist2 = self._point_to_line_distance(
            point,
            (duct.wall2.x1, duct.wall2.y1),
            (duct.wall2.x2, duct.wall2.y2)
        )
        
        # Also check distance to midpoint
        mid_dist = np.sqrt(
            (point[0] - duct.center[0])**2 + 
            (point[1] - duct.center[1])**2
        )
        
        return min(dist1, dist2, mid_dist)
    
    @staticmethod
    def _point_to_line_distance(
        point: Tuple[float, float],
        line_start: Tuple[int, int],
        line_end: Tuple[int, int]
    ) -> float:
        """
        Calculate perpendicular distance from point to line segment.
        
        Uses vector projection for efficient calculation.
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line vector
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Degenerate line (point)
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Project point onto line
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        
        # Clamp to segment
        t = max(0, min(1, t))
        
        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def calculate_real_length(
        self,
        duct: DuctSegment,
        scale_px_per_foot: float
    ) -> float:
        """
        Calculate real-world length of duct segment.
        
        Args:
            duct: Duct segment
            scale_px_per_foot: Scale factor (pixels per foot)
            
        Returns:
            Length in feet
        """
        if scale_px_per_foot <= 0:
            return 0.0
        
        return duct.length_px / scale_px_per_foot
