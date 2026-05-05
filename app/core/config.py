from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """12-Factor compliant configuration. All CV parameters loaded from environment."""
    
    # File handling
    HVAC_MAX_FILE_MB: int = 10
    HVAC_PDF_DPI: int = 400  # High DPI for small text detection
    
    # Preprocessing parameters
    CV_OTSU_GAUSSIAN_BLUR_KERNEL: int = 5  # Must be odd
    CV_MORPH_KERNEL_WIDTH: int = 5  # Kernel width for morphological ops
    CV_MORPH_KERNEL_HEIGHT: int = 5  # Kernel height for morphological ops
    CV_MORPH_ITERATIONS: int = 2  # Number of morphological iterations
    
    # Hough Line Detection parameters (used for layout segmentation only now)
    CV_HOUGH_RHO_RESOLUTION: float = 1.0  # Pixel resolution
    CV_HOUGH_THETA_RESOLUTION: float = 1.0  # Degree resolution (divided by 180)
    CV_HOUGH_THRESHOLD: int = 50  # Minimum votes for line detection
    CV_HOUGH_MIN_LINE_LENGTH: int = 200  # Minimum line length in pixels (raised to drop noise)
    CV_HOUGH_MAX_LINE_GAP: int = 20  # Maximum gap between line segments
    
    # Feature flag: use new region-based pipeline vs old Hough line-pair pipeline
    USE_REGION_BASED_PIPELINE: bool = True

    # --- Vector pipeline (PyMuPDF direct PDF parsing) ---
    USE_VECTOR_PIPELINE: bool = True       # take precedence over raster pipelines for PDFs
    VEC_MIN_LINE_LENGTH_PTS: float = 50.0   # min length (PDF points) to consider a line a duct wall
    VEC_PARALLEL_ANGLE_TOL_DEG: float = 2.0  # max angle delta to call two lines parallel
    VEC_MIN_DUCT_WIDTH_PTS: float = 5.0      # min gap between parallel walls (PDF points)
    VEC_MAX_DUCT_WIDTH_PTS: float = 40.0     # max gap between parallel walls (PDF points)
    VEC_MIN_OVERLAP_RATIO: float = 0.7       # min projection overlap between a wall pair
    VEC_MAX_LENGTH_RATIO: float = 1.6        # walls must be similar in length (max/min)
    VEC_MAX_END_OFFSET_PTS: float = 15.0     # walls' endpoints must align (along the run direction)
    VEC_STROKE_WIDTH_TOL: float = 0.2        # max stroke-width delta between paired walls (PDF points)
    VEC_MERGE_COLLINEAR_TOL_PTS: float = 6.0  # distance tolerance for merging collinear duct segments
    VEC_MIN_PRIMITIVES_FOR_VECTOR: int = 100  # below this, treat as scanned PDF and fall back

    # Legacy Hough / morphological fusion params (kept for fallback)
    CV_PARALLEL_ANGLE_TOLERANCE: float = 10.0
    CV_DUCT_MIN_WIDTH_PX: int = 20
    CV_DUCT_MAX_WIDTH_PX: int = 600
    CV_DUCT_WALL_DISTANCE_TOLERANCE: float = 0.3
    CV_DUCT_MIN_LINE_THICKNESS: int = 5
    CV_MORPH_H_KERNEL_LENGTH: int = 200
    CV_MORPH_V_KERNEL_LENGTH: int = 200
    CV_DUCT_MIN_ASPECT_RATIO: float = 3.0
    CV_DUCT_MIN_AREA_PX: int = 5000

    # --- Region-based pipeline params ---
    # Stage 1: OCR mask padding
    RB_TEXT_MASK_PADDING: int = 8          # px to expand each text box before erasing

    # Stage 2a: Pre-bridge cleanup — remove tiny CCs before morphology
    RB_PRE_CLEAN_MIN_AREA: int = 200       # remove CCs smaller than this (symbols/text fragments)
    RB_PRE_CLEAN_MIN_ASPECT: float = 3.0   # keep only elongated CCs (ducts are elongated)

    # Stage 2b: Hough prefilter — extract only long-line pixels before bridging
    RB_HOUGH_PREFILTER_ENABLED: bool = False  # DISABLED: kills short duct segments, keeps borders
    RB_HOUGH_PREFILTER_MIN_LEN: int = 200  # min line length for Hough prefilter
    RB_HOUGH_PREFILTER_THRESH: int = 30    # Hough vote threshold

    # Stage 2c: Morphological bridging (reconstruction-based)
    RB_BRIDGE_KERNEL_H: int = 1            # kernel height for horizontal close
    RB_BRIDGE_KERNEL_W: int = 80           # kernel width for horizontal close (~0.2in @ 400DPI)
    RB_BRIDGE_ITERATIONS: int = 3          # close iterations

    # Stage 3: Connected components
    RB_CC_MIN_AREA: int = 3000             # minimum component area in pixels
    RB_CC_MIN_LONG_SIDE: int = 200         # minimum long-side length in pixels
    RB_CC_MIN_ASPECT: float = 2.5          # min aspect ratio (long/short)

    # Stage 4: Contour validation confidence
    RB_SCORE_THRESHOLD: float = 0.4        # keep components scoring above this

    # Stage 5–6: Skeletonization / graph
    RB_SKELETON_ENABLED: bool = True       # toggle skeletonisation step
    
    # OCR parameters
    OCR_MIN_CONFIDENCE: float = 0.30  # Tesseract confidence threshold
    OCR_PSM_MODE: int = 11  # Page segmentation mode (11 = sparse text)
    
    # Spatial mapping (text to duct association)
    SPATIAL_MAX_DISTANCE_PX: int = 200  # Maximum distance to associate text with duct
    SPATIAL_SEARCH_RADIUS_PX: int = 300  # Search radius around text for ducts
    
    # Annotation rendering
    RENDER_DUCT_LINE_THICKNESS: int = 10  # Thickness of duct highlight lines
    RENDER_OVERLAY_ALPHA: float = 0.7  # Transparency of duct overlays (higher = more visible)
    RENDER_FONT_SCALE: float = 0.7  # OpenCV font scale
    
    GOOGLE_API_KEY:str="AIzaSyBLBtJfMM5GSG5PBRwitSea2opwbU7XDB8"

    # Logging
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False
    )

settings = Settings()
