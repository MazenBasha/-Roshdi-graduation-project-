"""
Convert any markdown file in the project into a styled PDF.

Default: PROJECT_REPORT.md -> PROJECT_REPORT.pdf

Uses reportlab's Platypus flowables. Markdown -> HTML -> Paragraph
preserves headings, tables, code blocks, bold/italic, and inline code.

Usage:
    python src/report_to_pdf.py
    python src/report_to_pdf.py FILE_GUIDE.md
    python src/report_to_pdf.py path/to/in.md path/to/out.pdf
"""

import argparse
import os
import re
import sys
from html import escape

import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak,
    KeepTogether, Preformatted,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MD = os.path.join(PROJECT_ROOT, "PROJECT_REPORT.md")


# ─── Styles ──────────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=base["Title"], fontSize=22, leading=28,
        spaceAfter=14, textColor=colors.HexColor("#1a3a6e"),
    )
    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontSize=16, leading=20,
        spaceBefore=18, spaceAfter=8, textColor=colors.HexColor("#1a3a6e"),
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontSize=13, leading=17,
        spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#2a5fa0"),
    )
    s["h3"] = ParagraphStyle(
        "h3", parent=base["Heading3"], fontSize=11, leading=14,
        spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#444"),
    )
    s["body"] = ParagraphStyle(
        "body", parent=base["BodyText"], fontSize=10, leading=14,
        spaceAfter=6, alignment=TA_LEFT,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=base["BodyText"], fontSize=10, leading=14,
        leftIndent=14, bulletIndent=2, spaceAfter=2,
    )
    s["code"] = ParagraphStyle(
        "code", parent=base["Code"], fontSize=8.5, leading=11,
        backColor=colors.HexColor("#f4f4f4"),
        borderColor=colors.HexColor("#ddd"), borderWidth=0.5,
        borderPadding=4, leftIndent=2, rightIndent=2,
        spaceBefore=4, spaceAfter=8,
    )
    return s


# ─── Inline markdown -> reportlab markup ─────────────────────────────────────

def md_inline(text: str) -> str:
    """Convert inline markdown (bold, italic, code, links) to ReportLab tags.

    Order: extract inline code into placeholders FIRST, so its contents
    (which often contain `*`, `_`, etc.) are not mangled by italic/bold
    substitutions. Reinsert at the end.
    """
    code_segments = []

    def stash_code(m):
        code_segments.append(m.group(1))
        return f"\x00CODE{len(code_segments) - 1}\x00"

    # 1) Pull out `inline code` first (placeholders are non-ASCII so they
    #    cannot collide with markdown markers).
    text = re.sub(r"`([^`]+)`", stash_code, text)
    # 2) Escape what remains.
    text = escape(text, quote=False)
    # 3) Bold (** **) before italic (* *).
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    # 4) Links.
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  r'<link href="\2" color="blue">\1</link>', text)

    # 5) Reinsert code segments, escaping their inner content.
    def restore(m):
        idx = int(m.group(1))
        body = escape(code_segments[idx], quote=False)
        return f'<font face="Courier" backColor="#f0f0f0">{body}</font>'

    text = re.sub(r"\x00CODE(\d+)\x00", restore, text)
    return text


# ─── Table parsing ──────────────────────────────────────────────────────────

def split_table_row(line: str):
    parts = line.strip().strip("|").split("|")
    return [p.strip() for p in parts]


def is_table_separator(line: str) -> bool:
    s = line.strip().strip("|").strip()
    if not s:
        return False
    cells = [c.strip() for c in s.split("|")]
    return all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c)


def make_table(rows, styles):
    """Build a Platypus Table from a list-of-lists of raw strings."""
    rendered = []
    for r_idx, row in enumerate(rows):
        rendered.append([
            Paragraph(md_inline(cell), styles["body"]) for cell in row
        ])

    # Use an A4-friendly available width; reportlab will distribute columns.
    n_cols = max(len(r) for r in rows)
    avail_w = A4[0] - 30 * mm
    col_w = [avail_w / n_cols] * n_cols

    tbl = Table(rendered, colWidths=col_w, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a6e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fc")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


# ─── Main parser: walk the markdown line by line ─────────────────────────────

def md_to_flowables(md_text: str, styles):
    flow = []
    lines = md_text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Blank line
        if not stripped:
            i += 1
            continue

        # Horizontal rule -> spacer (skip "---")
        if re.fullmatch(r"-{3,}", stripped):
            flow.append(Spacer(1, 6))
            i += 1
            continue

        # Headings
        if stripped.startswith("# "):
            flow.append(Paragraph(md_inline(stripped[2:]), styles["title"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flow.append(Paragraph(md_inline(stripped[3:]), styles["h1"]))
            i += 1
            continue
        if stripped.startswith("### "):
            flow.append(Paragraph(md_inline(stripped[4:]), styles["h2"]))
            i += 1
            continue
        if stripped.startswith("#### "):
            flow.append(Paragraph(md_inline(stripped[5:]), styles["h3"]))
            i += 1
            continue

        # Code fence
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code_text = "\n".join(buf)
            flow.append(Preformatted(code_text, styles["code"]))
            continue

        # Table (current line + next is separator)
        if "|" in line and i + 1 < n and is_table_separator(lines[i + 1]):
            header = split_table_row(line)
            i += 2  # skip header + separator
            rows = [header]
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(split_table_row(lines[i]))
                i += 1
            flow.append(KeepTogether(make_table(rows, styles)))
            flow.append(Spacer(1, 6))
            continue

        # Bullets
        if re.match(r"^\s*[-*]\s+", line):
            bullet_text = re.sub(r"^\s*[-*]\s+", "", line)
            flow.append(Paragraph(md_inline(bullet_text), styles["bullet"],
                                   bulletText="•"))
            i += 1
            continue

        # Numbered list
        m = re.match(r"^\s*\d+\.\s+(.*)", line)
        if m:
            flow.append(Paragraph(md_inline(m.group(1)), styles["bullet"],
                                   bulletText="–"))
            i += 1
            continue

        # Plain paragraph: collect until blank line
        para_lines = [line]
        i += 1
        while i < n and lines[i].strip() and not (
            lines[i].lstrip().startswith(("#", "- ", "* ", "```"))
            or re.match(r"^\s*\d+\.\s+", lines[i])
            or "|" in lines[i]
        ):
            para_lines.append(lines[i])
            i += 1
        text = " ".join(s.strip() for s in para_lines)
        flow.append(Paragraph(md_inline(text), styles["body"]))

    return flow


# ─── Footer with page numbers ────────────────────────────────────────────────

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888"))
    canvas.drawRightString(
        A4[0] - 15 * mm, 10 * mm,
        f"Egyptian Currency Detection — Project Report   |   Page {doc.page}",
    )
    canvas.restoreState()


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Markdown -> PDF for project docs")
    p.add_argument("input", nargs="?", default=DEFAULT_MD,
                   help="Markdown file (default: PROJECT_REPORT.md)")
    p.add_argument("output", nargs="?", default=None,
                   help="Output PDF (default: same name, .pdf)")
    return p.parse_args()


def main():
    args = parse_args()
    src_md = os.path.abspath(args.input)
    if not os.path.exists(src_md):
        raise FileNotFoundError(f"Source not found: {src_md}")
    out_pdf = os.path.abspath(args.output) if args.output \
        else os.path.splitext(src_md)[0] + ".pdf"

    with open(src_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    styles = build_styles()
    flow = md_to_flowables(md_text, styles)

    title = os.path.splitext(os.path.basename(src_md))[0]
    doc = SimpleDocTemplate(
        out_pdf, pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=18 * mm,
        title=title,
        author="MazenBasha",
    )
    doc.build(flow, onFirstPage=on_page, onLaterPages=on_page)

    size_kb = os.path.getsize(out_pdf) / 1024
    print(f"PDF written: {out_pdf} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
