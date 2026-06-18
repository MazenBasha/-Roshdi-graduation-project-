"""Convert MODEL_DESCRIPTION.md -> styled HTML -> PDF (via Chrome headless)."""
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "MODEL_DESCRIPTION.md"
HTML_PATH = ROOT / "MODEL_DESCRIPTION.html"
PDF_PATH = ROOT / "MODEL_DESCRIPTION.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
html, body { font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
             color: #1a1a1a; line-height: 1.45; font-size: 10.5pt; }
body { max-width: 100%; margin: 0; padding: 0; }
h1 { font-size: 22pt; border-bottom: 3px solid #1f3a93; padding-bottom: 4px;
     color: #1f3a93; margin-top: 0.4em; }
h2 { font-size: 15pt; color: #1f3a93; border-bottom: 1px solid #ccd; padding-bottom: 2px;
     margin-top: 1.4em; page-break-after: avoid; }
h3 { font-size: 12.5pt; color: #1f3a93; margin-top: 1.2em; page-break-after: avoid; }
h4 { font-size: 11pt; color: #34495e; margin-top: 1em; page-break-after: avoid; }
p, li { font-size: 10.5pt; }
table { border-collapse: collapse; width: 100%; margin: 0.6em 0 1.2em 0;
        font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #c0c5cc; padding: 5px 7px; text-align: left; vertical-align: top; }
th { background: #eef2f9; color: #1f3a93; font-weight: 600; }
tr:nth-child(even) td { background: #f8fafd; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9.5pt;
       background: #f3f5f9; padding: 1px 4px; border-radius: 3px; color: #c0392b; }
pre { background: #f3f5f9; padding: 10px 12px; border-left: 3px solid #1f3a93;
      border-radius: 3px; overflow-x: auto; page-break-inside: avoid; font-size: 9pt; }
pre code { background: transparent; padding: 0; color: #1a1a1a; }
blockquote { border-left: 3px solid #888; color: #444; padding: 4px 12px;
             background: #fafafa; margin: 0.6em 0; font-style: italic; }
ul, ol { margin: 0.4em 0 0.8em 1.4em; }
hr { border: none; border-top: 1px solid #ccd; margin: 1.4em 0; }
strong { color: #111; }
@media print { h2, h3, h4 { page-break-after: avoid; } }
"""

def main() -> int:
    md_text = MD_PATH.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )
    html_doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Face Recognition Model - Description and Flutter Integration</title>
<style>{CSS}</style>
</head><body>{html_body}</body></html>
"""
    HTML_PATH.write_text(html_doc, encoding="utf-8")
    print(f"[build] wrote {HTML_PATH.name} ({len(html_doc)} chars)")

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not Path(chrome).exists():
        print("[build] Chrome not found at standard path", file=sys.stderr)
        return 1

    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        f"file://{HTML_PATH}",
    ]
    print("[build] running Chrome headless to render PDF...")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print("[build] Chrome stderr:", res.stderr, file=sys.stderr)
        return res.returncode
    print(f"[build] PDF -> {PDF_PATH} ({PDF_PATH.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
