import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.ocr_service import MockOCRAdapter

@pytest.fixture
def mock_ocr():
    return MockOCRAdapter(fixture_path="tests/mocks/ocr_fixtures.json")

@pytest.fixture
def client():
    return TestClient(app)
