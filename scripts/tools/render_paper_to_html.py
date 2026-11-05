#!/usr/bin/env python3
"""
Paper LaTeX to Standalone Interactive HTML Renderer.
Compiles docs/paper/main.tex and docs/paper/figures/ into a single
IEEE TVCG formatted HTML document with KaTeX math rendering,
clean two-column layout, and responsive figures.
"""

import os
import re

PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "paper"))
TEX_FILE = os.path.join(PAPER_DIR, "main.tex")
OUTPUT_HTML = os.path.join(PAPER_DIR, "EdgeStereoINR_Paper.html")
FIGURES_DIR = os.path.join(PAPER_DIR, "figures")

def render_html():
    if not os.path.exists(TEX_FILE):
        print(f"[!] Error: {TEX_FILE} not found.")
        return

    with open(TEX_FILE, "r", encoding="utf-8") as f:
        tex = f.read()

    # Extract title, abstract, keywords
    title_m = re.search(r"\\title\{([^}]+)\}", tex)
    title = title_m.group(1) if title_m else "EdgeStereoINR"

    abstract_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    abstract = abstract_m.group(1).strip() if abstract_m else ""

    keywords_m = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", tex, re.DOTALL)
    keywords = keywords_m.group(1).strip() if keywords_m else ""

    # Clean body
    body_m = re.search(r"\\maketitle.*?\n(.*)\\bibliographystyle", tex, re.DOTALL)
    body_tex = body_m.group(1) if body_m else tex

    # Pre-process LaTeX into HTML elements
    # Sections
    body_tex = re.sub(r"\\section\{([^}]+)\}", r"<h2>\1</h2>", body_tex)
    body_tex = re.sub(r"\\subsection\{([^}]+)\}", r"<h3>\1</h3>", body_tex)
    body_tex = re.sub(r"\\subsubsection\{([^}]+)\}", r"<h4>\1</h4>", body_tex)

    # Text formatting
    body_tex = re.sub(r"\\textbf\{([^}]+)\}", r"<strong>\1</strong>", body_tex)
    body_tex = re.sub(r"\\textit\{([^}]+)\}", r"<em>\1</em>", body_tex)
    body_tex = re.sub(r"\\IEEEPARstart\{([^}]+)\}\{([^}]+)\}", r"<span class='dropcap'>\1</span>\2", body_tex)

    # Figures
    def fig_repl(m):
        content = m.group(1)
        img_m = re.search(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}", content)
        cap_m = re.search(r"\\caption\{([^}]+)\}", content)
        lab_m = re.search(r"\\label\{([^}]+)\}", content)
        
        fig_id = lab_m.group(1) if lab_m else ""
        caption = cap_m.group(1) if cap_m else ""
        img_file = img_m.group(1) if img_m else ""
        
        # Prefer PNG over PDF for web rendering
        img_png = img_file.replace(".pdf", ".png")
        return f"""
        <div class="figure-container" id="{fig_id}">
            <img src="figures/{img_png}" alt="{caption}" class="paper-img" />
            <div class="caption"><strong>Figure:</strong> {caption}</div>
        </div>
        """

    body_tex = re.sub(r"\\begin\{figure\}(?:\[[^\]]*\])?(.*?)\\end\{figure\}", fig_repl, body_tex, flags=re.DOTALL)

    # Math equations
    body_tex = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", r"<div class='eq'>$$\1$$</div>", body_tex, flags=re.DOTALL)

    # Lists
    body_tex = re.sub(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", r"<ul>\1</ul>", body_tex, flags=re.DOTALL)
    body_tex = re.sub(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", r"<ol>\1</ol>", body_tex, flags=re.DOTALL)
    body_tex = re.sub(r"\\item\s+", r"<li>", body_tex)
    body_tex = re.sub(r"<li>(.*?)(\n|$)", r"<li>\1</li>\n", body_tex)

    # Tables
    body_tex = re.sub(r"\\begin\{table\}(?:\[[^\]]*\])?(.*?)\\end\{table\}", r"<div class='table-box'>\1</div>", body_tex, flags=re.DOTALL)
    body_tex = re.sub(r"\\caption\{([^}]+)\}", r"<div class='table-caption'><strong>Table:</strong> \1</div>", body_tex)
    
    # Clean citations
    body_tex = re.sub(r"\\cite\{([^}]+)\}", r"[\1]", body_tex)
    body_tex = re.sub(r"\\ref\{([^}]+)\}", r"<a href='#\1'>\1</a>", body_tex)

    # Paragraphs
    paras = body_tex.split("\n\n")
    cleaned_paras = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if p.startswith("<h") or p.startswith("<div") or p.startswith("<ul") or p.startswith("<ol"):
            cleaned_paras.append(p)
        else:
            cleaned_paras.append(f"<p>{p}</p>")
    formatted_body = "\n".join(cleaned_paras)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- KaTeX for high-fidelity LaTeX Math -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {{delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}}
            ]}});"></script>
    <style>
        @page {{
            size: A4;
            margin: 1.5cm;
        }}
        body {{
            font-family: 'Times New Roman', Times, serif;
            color: #111;
            line-height: 1.35;
            background-color: #f7f9fa;
            margin: 0;
            padding: 20px;
        }}
        .paper-container {{
            max-width: 1050px;
            margin: 0 auto;
            background: #fff;
            padding: 40px 60px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border-radius: 4px;
        }}
        h1.title {{
            font-size: 24px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 8px;
            letter-spacing: -0.2px;
        }}
        .authors {{
            text-align: center;
            font-size: 13px;
            font-style: italic;
            color: #333;
            margin-bottom: 25px;
        }}
        .abstract-box {{
            max-width: 900px;
            margin: 0 auto 30px auto;
            padding: 15px 25px;
            background: #fdfdfd;
            border-left: 3px solid #1a56db;
            font-size: 12px;
            text-align: justify;
        }}
        .abstract-box strong {{
            color: #0d2870;
        }}
        .keywords {{
            margin-top: 10px;
            font-size: 11.5px;
            color: #444;
        }}
        .two-column {{
            column-count: 2;
            column-gap: 28px;
            column-rule: 1px solid #e5e7eb;
            text-align: justify;
            font-size: 12.5px;
        }}
        h2 {{
            font-size: 14px;
            text-transform: uppercase;
            font-weight: bold;
            border-bottom: 1px solid #333;
            padding-bottom: 3px;
            margin-top: 20px;
            margin-bottom: 8px;
            break-after: avoid;
        }}
        h3 {{
            font-size: 13px;
            font-style: italic;
            margin-top: 14px;
            margin-bottom: 6px;
            break-after: avoid;
        }}
        p {{
            margin-bottom: 10px;
            text-indent: 14px;
        }}
        .dropcap {{
            float: left;
            font-size: 38px;
            line-height: 32px;
            padding-top: 2px;
            padding-right: 6px;
            font-weight: bold;
            color: #1a56db;
        }}
        .figure-container {{
            break-inside: avoid;
            margin: 15px 0;
            text-align: center;
            background: #fafafa;
            padding: 8px;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        }}
        .paper-img {{
            max-width: 100%;
            height: auto;
            border-radius: 2px;
        }}
        .caption {{
            font-size: 11px;
            color: #333;
            margin-top: 6px;
            text-align: justify;
        }}
        .eq {{
            text-align: center;
            margin: 10px 0;
            font-size: 12px;
        }}
        .table-box {{
            break-inside: avoid;
            margin: 15px 0;
            font-size: 11px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 6px 0;
        }}
        th, td {{
            border-top: 1px solid #333;
            border-bottom: 1px solid #333;
            padding: 4px 6px;
            text-align: center;
        }}
        .table-caption {{
            font-size: 11px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 4px;
        }}
        ul, ol {{
            padding-left: 20px;
            margin-bottom: 10px;
        }}
        li {{
            margin-bottom: 4px;
        }}
        @media print {{
            body {{
                background: #fff;
                padding: 0;
            }}
            .paper-container {{
                box-shadow: none;
                padding: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="paper-container">
        <h1 class="title">{title}</h1>
        <div class="authors">Active Stereo Research Consortium &bull; IEEE Transactions on Visualization and Computer Graphics (TVCG) Submission</div>
        
        <div class="abstract-box">
            <strong>ABSTRACT&mdash;</strong> {abstract}
            <div class="keywords"><strong>Index Terms&mdash;</strong> {keywords}</div>
        </div>

        <div class="two-column">
            {formatted_body}
        </div>
    </div>
</body>
</html>
"""
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("=" * 70)
    print(f"[+] Successfully rendered paper to interactive HTML:")
    print(f"    {OUTPUT_HTML}")
    print(f"[+] File Size: {os.path.getsize(OUTPUT_HTML) / 1024.0:.2f} KB")
    print("=" * 70)

if __name__ == "__main__":
    render_html()
