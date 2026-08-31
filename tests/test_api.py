import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gemini_model" in data

def test_query_endpoint():
    response = client.post("/query", json={"query": "What was NVIDIA's revenue in 2025?"})
    assert response.status_code == 200
    data = response.json()
    assert "final_answer" in data
    assert len(data["execution_steps"]) > 0

def test_list_documents_endpoint():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "extracted_images" in data
