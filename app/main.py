import uuid
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router
from app.core.exceptions import HVACDomainError
from app.core.logging import trace_id_var

app = FastAPI(
    title="HVAC Duct Analyzer API",
    version="2.0.0",
    description="Geometry-first HVAC duct detection using OpenCV heuristics"
)

@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    trace_id_var.set(trace_id)
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response

@app.exception_handler(HVACDomainError)
async def domain_error_handler(request: Request, exc: HVACDomainError):
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "status": "error",
            "trace_id": getattr(request.state, "trace_id", "unknown"),
            "error_code": exc.error_code,
            "message": str(exc)
        }
    )

# Include API routes
app.include_router(api_router, prefix="/api")

# Mount static files for frontend - must be LAST as it catches all unmatched routes
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    # Fallback for Docker container structure
    static_dir = "/app/static"
    if os.path.exists(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
