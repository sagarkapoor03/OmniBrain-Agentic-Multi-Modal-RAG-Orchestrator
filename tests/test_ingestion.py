import os
import pytest
from app.config import settings
from scripts.generate_sample_pdf import generate_sample_nvda_pdf
from app.ingestion.pdf_parser import parse_pdf
from app.retrieval.vector_store import vector_store

def test_pdf_parsing_and_vector_indexing():
    """Verify PyMuPDF text & image parsing and Qdrant vector indexing."""
    sample_pdf = os.path.join(settings.DOCUMENTS_DIR, "test_sample_nvda.pdf")
    generate_sample_nvda_pdf(sample_pdf)

    assert os.path.exists(sample_pdf)

    parsed = parse_pdf(sample_pdf)
    assert parsed["total_pages"] >= 2
    assert len(parsed["text_chunks"]) > 0
    assert len(parsed["extracted_images"]) > 0

    # Test Qdrant Vector Store indexing
    added = vector_store.add_chunks(parsed["text_chunks"])
    assert added > 0

    # Test Vector Search
    search_res = vector_store.search("revenue in fiscal year 2025", top_k=2)
    assert len(search_res) > 0
