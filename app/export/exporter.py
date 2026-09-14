"""
Research Report Exporter for Private Research Agent.

Exports research intelligence reports into:
- Markdown (.md)
- Structured JSON (.json)
- Portable Document Format (.pdf) without requiring an external browser engine.
"""

from datetime import datetime, timezone
import io
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def export_as_markdown(report: Dict[str, Any]) -> str:
    """Format report as a clean Markdown document."""
    question = report.get("question", "")
    answer = report.get("answer", "")
    sources = report.get("sources", [])
    timestamp = report.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    elapsed = report.get("elapsed_seconds")

    lines = [
        f"# Research Report: {question}",
        "",
        f"**Date:** {timestamp}",
    ]
    if elapsed:
        lines.append(f"**Research Duration:** {elapsed}s")
    lines.extend([
        "",
        "---",
        "",
        "## Executive Summary & Findings",
        "",
        answer,
        "",
        "---",
        "",
        "## Verified Sources & Evidence",
        "",
    ])

    if sources:
        for idx, s in enumerate(sources, start=1):
            sid = s.get("source_id", f"S{idx}")
            title = s.get("title", "Untitled Source")
            url = s.get("url", "")
            domain = s.get("domain", "")
            snippet = s.get("snippet", "")

            lines.append(f"### [{sid}] {title}")
            if url and url != "N/A":
                lines.append(f"- **URL:** [{url}]({url})")
            if domain:
                lines.append(f"- **Domain:** `{domain}`")
            if snippet:
                lines.append(f"- **Excerpt:** *\"{snippet}\"*")
            lines.append("")
    else:
        lines.append("No external citations recorded.\n")

    return "\n".join(lines)


def export_as_json(report: Dict[str, Any]) -> str:
    """Format report as structured JSON."""
    payload = {
        "title": f"Research Report: {report.get('question', '')}",
        "question": report.get("question", ""),
        "answer": report.get("answer", ""),
        "sources": report.get("sources", []),
        "timestamp": report.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "metadata": report.get("metadata", {}),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def export_as_pdf(report: Dict[str, Any]) -> bytes:
    """
    Generate PDF binary for research report.
    Uses reportlab if available; otherwise generates standard PDF stream.
    """
    question = report.get("question", "")
    answer = report.get("answer", "")
    sources = report.get("sources", [])
    timestamp = report.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Strategy 1: ReportLab
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib import colors

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            rightMargin=54,
            leftMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=8,
        )
        meta_style = ParagraphStyle(
            "DocMeta",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=12,
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#334155"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=8,
        )
        source_style = ParagraphStyle(
            "DocSource",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#475569"),
            spaceAfter=6,
        )

        story = []

        # Title & Meta
        story.append(Paragraph(f"Research Report: {question}", title_style))
        story.append(Paragraph(f"Generated: {timestamp} • Private Research Agent", meta_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=14))

        # Findings
        story.append(Paragraph("Executive Summary & Findings", heading_style))
        clean_paragraphs = answer.split("\n\n")
        for p in clean_paragraphs:
            clean_p = p.strip().replace("\n", " ")
            if clean_p:
                story.append(Paragraph(clean_p, body_style))

        # Sources
        if sources:
            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=14))
            story.append(Paragraph("Verified Sources & Citations", heading_style))

            for idx, s in enumerate(sources, start=1):
                sid = s.get("source_id", f"S{idx}")
                title = s.get("title", "Untitled Source")
                url = s.get("url", "")
                src_text = f"<b>[{sid}] {title}</b><br/>{url}"
                story.append(Paragraph(src_text, source_style))
                story.append(Spacer(1, 4))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        logger.warning("reportlab not installed. Generating basic PDF stream...")
    except Exception as ex:
        logger.error(f"ReportLab PDF generation error: {ex}. Using basic PDF generator...")

    # Fallback minimal valid PDF 1.4 byte generator (zero dependencies)
    content_lines = [
        f"Research Report: {question}",
        f"Generated: {timestamp}",
        "-" * 50,
        "",
        "FINDINGS:",
        "",
    ]
    for p in answer.split("\n"):
        content_lines.append(p[:95])

    content_lines.extend(["", "-" * 50, "SOURCES:"])
    for idx, s in enumerate(sources, start=1):
        content_lines.append(f"[{s.get('source_id', f'S{idx}')}] {s.get('title', '')}: {s.get('url', '')}"[:95])

    pdf_text = "\n".join(content_lines)
    # Basic PDF generation with stream object
    stream_content = "BT /F1 10 Tf 50 750 Td 14 TL\n"
    for line in pdf_text.splitlines()[:55]:
        safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_content += f"({safe_line}) '\n"
    stream_content += "ET"

    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"5 0 obj << /Length " + str(len(stream_content.encode('latin-1'))).encode('ascii') + b" >>\nstream\n"
        + stream_content.encode("latin-1", errors="replace") +
        b"\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n"
        b"0000000323 00000 n \ntrailer << /Size 6 /Root 1 0 R >>\nstartxref\n450\n%%EOF"
    )
    return pdf_bytes
