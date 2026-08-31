import os
from typing import List, Dict, Any
from app.agents.state import AgentState
from app.agents.llm import llm_client
from app.config import settings

def vision_agent_node(state: AgentState) -> AgentState:
    """Vision Agent: Interprets financial charts, figures, and visual graphs via Gemini Flash multimodal mode."""
    query = state["query"]
    image_dir = settings.EXTRACTED_IMAGES_DIR

    image_paths: List[str] = []
    if os.path.exists(image_dir):
        files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        files.sort()
        image_paths = [os.path.join(image_dir, f) for f in files[:2]]

    retrieved_images: List[Dict[str, Any]] = []

    if image_paths:
        prompt = f"""You are the Vision Agent for OmniBrain. Analyze the attached chart/image from the NVIDIA financial document and answer the user query with numerical precision.
Describe chart values, legend items, trends, and specific data points relevant to:

User Query: "{query}" """

        vision_analysis = llm_client.generate_multimodal(prompt=prompt, image_paths=image_paths)

        for img_path in image_paths:
            retrieved_images.append({
                "image_path": img_path,
                "image_name": os.path.basename(img_path),
                "analysis": vision_analysis
            })

        citations = state.get("citations", [])
        for img_path in image_paths:
            citations.append({
                "type": "chart_image",
                "doc_name": "NVIDIA Financial Report",
                "image_path": img_path,
                "snippet": f"Visual Chart Reference: {os.path.basename(img_path)}"
            })
        state["citations"] = citations
    else:
        vision_analysis = "No extracted document chart images found in the repository. Operating on document text context."

    state["retrieved_images"] = retrieved_images

    steps = state.get("execution_steps", [])
    steps.append(f"Vision Agent processed {len(image_paths)} chart image(s) using Gemini Flash Multimodal Vision.")
    state["execution_steps"] = steps

    return state
