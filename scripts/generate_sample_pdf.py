import os
import sys
from pathlib import Path

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from app.config import settings

def build_sample_chart(chart_path: str):
    """Generates an NVIDIA Revenue Growth bar chart."""
    years = ["FY2021", "FY2022", "FY2023", "FY2024", "FY2025"]
    revenue = [16675, 26914, 26974, 60922, 130400]

    plt.figure(figsize=(6, 3.5))
    bars = plt.bar(years, revenue, color="#76B900", edgecolor="#1A1A1A", width=0.55)
    plt.title("NVIDIA (NVDA) Annual Revenue ($ Millions)", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Revenue ($M)", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1500,
                 f"${height:,}M", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()

def generate_sample_nvda_pdf(output_path: str):
    """Creates a sample multi-page NVIDIA financial report PDF."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chart_temp_path = os.path.join(settings.EXTRACTED_IMAGES_DIR, "nvda_revenue_chart_temp.png")
    build_sample_chart(chart_temp_path)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "NvidiaTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#76B900"),
        alignment=0
    )
    heading_style = ParagraphStyle(
        "NvidiaHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1A1A1A"),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "NvidiaBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#333333"),
        spaceAfter=8
    )

    elements = []

    # Title & Header
    elements.append(Paragraph("NVIDIA Corporation (NVDA) Financial Report", title_style))
    elements.append(Paragraph("Fiscal Year 2025 Executive Overview & Performance Analysis", ParagraphStyle("Sub", fontName="Helvetica-Oblique", fontSize=11, textColor=colors.gray)))
    elements.append(Spacer(1, 12))

    # Section 1: Executive Overview
    elements.append(Paragraph("1. Executive Overview & Data Center Highlights", heading_style))
    p1 = ("NVIDIA Corporation recorded unprecedented financial performance in Fiscal Year 2025, driven by explosive global demand for AI compute infrastructure. "
          "Full-year revenue reached $130,400 million, representing a 114% year-over-year surge compared to $60,922 million in FY2024. "
          "The Data Center division remained the primary growth engine, supported by widespread enterprise adoption of Hopper HGX platforms and early shipments of the next-generation Blackwell architecture.")
    elements.append(Paragraph(p1, body_style))

    p2 = ("Gross margin reached an all-time high of 75.4%, driven by strong software attach rates including NVIDIA AI Enterprise and Quantum Networking hardware. "
          "Operating income for the full year rose to $81,500 million, while net income scaled to $72,800 million. "
          "Diluted earnings per share (EPS) post-split stood at $2.95 per share.")
    elements.append(Paragraph(p2, body_style))

    elements.append(Spacer(1, 10))

    # Section 2: Financial Performance Table
    elements.append(Paragraph("2. NVIDIA Annual Financial Performance (FY2021 – FY2025)", heading_style))
    
    table_data = [
        ["Fiscal Year", "Revenue ($M)", "Operating Income ($M)", "Net Income ($M)", "Diluted EPS ($)"],
        ["FY 2021", "$16,675", "$4,532", "$4,332", "$1.73"],
        ["FY 2022", "$26,914", "$10,041", "$9,752", "$3.85"],
        ["FY 2023", "$26,974", "$4,224", "$4,368", "$1.74"],
        ["FY 2024", "$60,922", "$32,972", "$29,760", "$11.93"],
        ["FY 2025", "$130,400", "$81,500", "$72,800", "$2.95"],
    ]

    t = Table(table_data, colWidths=[90, 110, 120, 110, 90])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#76B900")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F9F9F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
    ]))
    elements.append(t)

    # Page Break for Visual Chart Section
    elements.append(PageBreak())

    elements.append(Paragraph("3. Multi-Modal Revenue Growth Visual Analysis", heading_style))
    p3 = ("The chart below illustrates NVIDIA's revenue growth trajectory from FY2021 through FY2025. "
          "Notice the accelerated exponential steepness beginning in FY2024 with the launch of generative AI enterprise deployments.")
    elements.append(Paragraph(p3, body_style))
    elements.append(Spacer(1, 10))

    img = Image(chart_temp_path, width=480, height=280)
    elements.append(img)
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Figure 1: NVIDIA (NVDA) 5-Year Revenue Growth Trajectory ($ Millions)", ParagraphStyle("Caption", fontName="Helvetica-Oblique", fontSize=9, alignment=1, textColor=colors.gray)))

    doc.build(elements)
    print(f"Generated sample NVIDIA report at: {output_path}")

if __name__ == "__main__":
    out_pdf = os.path.join(settings.DOCUMENTS_DIR, "sample_nvda_report.pdf")
    generate_sample_nvda_pdf(out_pdf)
