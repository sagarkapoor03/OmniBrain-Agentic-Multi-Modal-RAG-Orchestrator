import os
import fitz  # PyMuPDF
from typing import List, Dict, Any
from app.config import settings

def parse_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
    """
    Parses a PDF file using PyMuPDF:
    1. Extracts text with page numbers and splits into chunks.
    2. Extracts embedded images/charts and saves them to data/documents/extracted_images/
    
    Returns a dict containing text_chunks list and extracted_images list.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    doc = fitz.open(pdf_path)
    doc_name = os.path.basename(pdf_path)

    text_chunks: List[Dict[str, Any]] = []
    extracted_images: List[Dict[str, Any]] = []

    global_chunk_idx = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")

        # Clean text
        page_text_clean = " ".join(page_text.split())

        # Simple sliding window text chunking
        if page_text_clean:
            words = page_text_clean.split()
            step = chunk_size - chunk_overlap
            if step <= 0:
                step = chunk_size

            for i in range(0, len(words), step):
                chunk_words = words[i : i + chunk_size]
                chunk_text = " ".join(chunk_words)

                if len(chunk_text.strip()) > 20:  # ignore tiny noise chunks
                    text_chunks.append({
                        "chunk_id": f"{doc_name}_p{page_num + 1}_c{global_chunk_idx}",
                        "doc_name": doc_name,
                        "page_number": page_num + 1,
                        "text": chunk_text
                    })
                    global_chunk_idx += 1

        # Extract images / figures from page
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            img_filename = f"{os.path.splitext(doc_name)[0]}_p{page_num + 1}_img{img_idx + 1}.{image_ext}"
            img_save_path = os.path.join(settings.EXTRACTED_IMAGES_DIR, img_filename)

            with open(img_save_path, "wb") as f:
                f.write(image_bytes)

            extracted_images.append({
                "image_id": f"{doc_name}_p{page_num + 1}_img{img_idx + 1}",
                "doc_name": doc_name,
                "page_number": page_num + 1,
                "image_path": img_save_path,
                "context": f"Extracted figure from page {page_num + 1} of {doc_name}"
            })

    total_pages = len(doc)
    doc.close()

    return {
        "doc_name": doc_name,
        "total_pages": total_pages,
        "text_chunks": text_chunks,
        "extracted_images": extracted_images
    }
