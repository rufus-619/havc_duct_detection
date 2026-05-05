from unittest.mock import patch
import numpy as np

def test_unsupported_file_type_returns_415(client):
    response = client.post("/api/v1/analyze-blueprint",
                           files={"file": ("drawing.txt", b"not-an-image", "text/plain")})
    assert response.status_code == 415
    assert response.json()["error_code"] == "UNSUPPORTED_MEDIA_TYPE"

def test_oversized_file_returns_413(client):
    big_file = b"0" * (11 * 1024 * 1024)
    response = client.post("/api/v1/analyze-blueprint",
                           files={"file": ("big.pdf", big_file, "application/pdf")})
    assert response.status_code == 413
    assert response.json()["error_code"] == "PAYLOAD_TOO_LARGE"

@patch('app.api.v1.routes.ocr_adapter')
def test_valid_upload_response_schema_matches_contract(mock_ocr_adapter, client, mock_ocr):
    # Use mock ocr instead of real tesseract
    mock_ocr_adapter.extract.side_effect = mock_ocr.extract
    
    # We need a small valid image
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.png', img)
    
    response = client.post("/api/v1/analyze-blueprint",
                           files={"file": ("sample.png", buf.tobytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "trace_id" in body
    assert "pipeline_telemetry" in body
    
    # Check if data adheres to schema implicitly by status 200 and schema validations in fastapi
    assert "data" in body

def test_error_response_never_contains_traceback(client):
    # Corrupt file → 400
    response = client.post("/api/v1/analyze-blueprint",
                           files={"file": ("bad.png", b"not-an-image", "image/png")})
    assert response.status_code == 400
    assert "Traceback" not in response.text
    assert "error_code" in response.json()
