import os
import sys
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st

# Page setup (MUST be the first Streamlit command executed)
st.set_page_config(
    page_title="OmniBrain — Multi-Modal RAG Orchestrator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

import requests
import pandas as pd
from PIL import Image

# Import app modules
from app.database.stock_db import run_sql_query
from app.ingestion.pdf_parser import parse_pdf
from app.retrieval.vector_store import vector_store
from app.agents.graph import run_agentic_pipeline
from scripts.generate_sample_pdf import generate_sample_nvda_pdf

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Custom CSS for polished aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #76B900;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    .citation-card {
        background-color: #F8F9FA;
        border-left: 4px solid #76B900;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .step-badge {
        background-color: #E9ECEF;
        color: #333;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Welcome to **OmniBrain**! I am an Agentic Multi-Modal RAG system for NVIDIA (NVDA) corporate reports and stock analytics. How can I assist your quantitative analysis today?"
        }
    ]

# Sidebar Configuration
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg", width=160)
    st.markdown("### OmniBrain Settings")
    st.info("⚡ Powered by **Google Gemini Flash** (Text + Multimodal Vision)")

    st.markdown("---")
    st.markdown("### 📄 Document Ingestion")

    uploaded_file = st.file_uploader("Upload NVIDIA PDF Report (10-K / Annual)", type=["pdf"], key="file_uploader_nvda")
    if uploaded_file is not None:
        if st.button("Process & Index Document", type="primary", key="btn_process_doc"):
            with st.spinner("Parsing text, extracting figures & indexing into Qdrant..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{API_BASE_URL}/upload", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Indexed {data['indexed_chunks']} text chunks & {data['extracted_images_count']} figure images!")
                    else:
                        st.error(f"Error uploading file: {res.text}")
                except Exception as e:
                    # Fallback to direct Python ingestion
                    save_path = os.path.join("data", "documents", uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    parse_res = parse_pdf(save_path)
                    added = vector_store.add_chunks(parse_res["text_chunks"])
                    st.success(f"Direct Ingestion: Indexed {added} chunks & {len(parse_res['extracted_images'])} chart images!")

    # Quick sample PDF button
    if st.button("⚡ Generate & Index Sample NVDA PDF", key="btn_sample_pdf"):
        with st.spinner("Generating sample NVIDIA quarterly report..."):
            sample_pdf_path = os.path.join("data", "documents", "sample_nvda_report.pdf")
            generate_sample_nvda_pdf(sample_pdf_path)
            parse_res = parse_pdf(sample_pdf_path)
            added = vector_store.add_chunks(parse_res["text_chunks"])
            st.success(f"Sample NVDA Report indexed! ({added} text chunks, {len(parse_res['extracted_images'])} charts)")

    st.markdown("---")
    st.markdown("### 📊 Database Explorer")
    with st.expander("Preview NVIDIA Stock & Financial DB", expanded=False):
        tab1, tab2 = st.tabs(["Stock Prices", "Financials"])
        with tab1:
            stock_data = run_sql_query("SELECT symbol, date, open, high, low, close, volume FROM stock_prices WHERE symbol='NVDA' ORDER BY date DESC LIMIT 10;")
            if stock_data and "error" not in stock_data[0]:
                st.dataframe(pd.DataFrame(stock_data), height=200)
            else:
                st.write("No stock price data found.")
        with tab2:
            fin_data = run_sql_query("SELECT symbol, fiscal_year, quarter, revenue_millions, net_income_millions, eps FROM company_financials WHERE symbol='NVDA' ORDER BY fiscal_year DESC;")
            if fin_data and "error" not in fin_data[0]:
                st.dataframe(pd.DataFrame(fin_data), height=200)
            else:
                st.write("No financial data found.")

# Main Header
st.markdown('<div class="main-title">OmniBrain — Agentic Multi-Modal RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">NVIDIA (NVDA) Intelligent Orchestrator for Unstructured Reports & Structured Stock Analytics</div>', unsafe_allow_html=True)

# Sample Prompt Chips
st.markdown("##### 💡 Try Example Prompts:")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📈 Compare revenue growth with stock price", use_container_width=True, key="chip_1"):
        st.session_state["user_prompt"] = "Compare NVIDIA's revenue growth with its historical stock performance."
with col2:
    if st.button("📊 Highest closing price in 2024", use_container_width=True, key="chip_2"):
        st.session_state["user_prompt"] = "What was NVIDIA's highest stock closing price in 2024?"
with col3:
    if st.button("🔍 FY2025 Revenue in report", use_container_width=True, key="chip_3"):
        st.session_state["user_prompt"] = "What was NVIDIA's total revenue in FY2025 according to the report?"
with col4:
    if st.button("🖼️ What does revenue chart indicate?", use_container_width=True, key="chip_4"):
        st.session_state["user_prompt"] = "What does the NVIDIA revenue growth chart indicate?"

st.markdown("---")

# Render Chat History with unique expander keys per message index
for idx, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📚 View Supporting References & Citations", key=f"hist_cit_exp_{idx}"):
                for cit in message["citations"]:
                    st.markdown(f"- **[{cit.get('type', 'reference').upper()}]** {cit.get('snippet', '')}")

        if "retrieved_images" in message and message["retrieved_images"]:
            with st.expander("🖼️ View Referenced Chart Images", key=f"hist_img_exp_{idx}"):
                for img_info in message["retrieved_images"]:
                    path = img_info.get("image_path")
                    if os.path.exists(path):
                        st.image(path, caption=img_info.get("image_name"), width=400)

# Chat Input Handler
prompt = st.chat_input("Ask a question about NVIDIA financial reports or stock performance...")
if "user_prompt" in st.session_state:
    prompt = st.session_state.pop("user_prompt")

if prompt:
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.chat_message("assistant"):
        status_box = st.status("🧠 OmniBrain Supervisor routing query...", expanded=True)
        
        try:
            # Call backend API or local pipeline fallback
            try:
                res = requests.post(f"{API_BASE_URL}/query", json={"query": prompt}, timeout=15)
                data = res.json()
            except Exception:
                data = run_agentic_pipeline(prompt)

            # Update status indicator with completed safe stages
            steps = data.get("execution_steps", [])
            for step in steps:
                status_box.write(f"✓ {step}")
            
            selected_route = data.get("route_selected") or data.get("route", "agent")
            status_box.update(label=f"Completed via `{selected_route}` in {data.get('latency_ms', 0)}ms", state="complete")

            # Final Answer
            final_ans = data.get("final_answer", "No response generated.")
            st.markdown(final_ans)

            citations = data.get("citations", [])
            images = data.get("retrieved_images", [])

            current_idx = len(st.session_state["messages"])
            if citations:
                with st.expander("📚 Supporting References & Citations", key=f"curr_cit_exp_{current_idx}"):
                    for cit in citations:
                        st.markdown(f"- **[{cit.get('type', 'reference').upper()}]** {cit.get('snippet', '')}")

            if images:
                with st.expander("🖼️ Referenced Visual Charts", key=f"curr_img_exp_{current_idx}"):
                    for img_info in images:
                        path = img_info.get("image_path")
                        if os.path.exists(path):
                            st.image(path, caption=img_info.get("image_name"), width=400)

            # Append to session state
            st.session_state["messages"].append({
                "role": "assistant",
                "content": final_ans,
                "citations": citations,
                "retrieved_images": images
            })

        except Exception as e:
            status_box.update(label="Error executing workflow", state="error")
            st.error(f"Execution Error: {e}")
