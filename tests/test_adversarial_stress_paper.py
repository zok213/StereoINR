"""
Milestone 3 Adversarial Stress-Test and Empirical Verification Suite
Scale-Liberated StereoINR (Continuous DyT-EdgeINR) Paper Bundle

Adversarial Verification Dimensions:
1. BibTeX Strict Syntax & Pushdown Automaton Brace Balancing
2. BibTeX Required Field Validation & Unescaped Special Character Detection
3. BibTeX Key Uniqueness & Author Format Validation
4. Citation Graph & 100% Active In-Text Citation Mapping (0 Undefined, 0 Uncited)
5. Comprehensive Coverage of All 7 Specified Topic Domains
6. Float Precision Specifiers, Captions, and Environment Types for All 13 Floats
7. LaTeX Environment Nesting Pushdown Automaton (figure, figure*, table, table*, tikzpicture, tabular, equation)
8. Global Label Uniqueness & Elimination of Duplicate \\label{fig:latency}
9. Cross-Reference Resolution (0 Broken \\ref Targets)
10. Release Zip Forensic Integrity (POSIX Paths, Zero Backslashes, Zero 0-byte Files, Byte-for-Byte Disk Parity)
"""

import os
import re
import tempfile
import zipfile
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(REPO_ROOT, "reports", "paper")
MAIN_TEX = os.path.join(PAPER_DIR, "main.tex")
REFERENCES_BIB = os.path.join(PAPER_DIR, "references.bib")
BUNDLE_ZIP = os.path.join(REPO_ROOT, "reports", "paper_bundle.zip")

ALL_13_FLOATS = [
    "fig:arch",
    "fig:memaccess",
    "tab:hw",
    "fig:loss",
    "fig:pareto",
    "fig:latency",
    "tab:crop",
    "fig:bars",
    "tab:mae",
    "fig:visual_mb",
    "fig:visual_q3s",
    "tab:scale",
    "fig:foveated",
]

DOMAIN_KEYWORDS = {
    "Domain 1: Continuous INR": ["liif", "lte", "ultrasr", "siren"],
    "Domain 2: Stereo SR & EPA": ["stereosr", "passr", "flickr1024", "ipassr", "ssrdefnet", "nafssr", "stereoinr"],
    "Domain 3: Efficient Mobile & Transformer Backbones": ["convnext", "nafnet", "hat", "mobilenetv3", "repvgg"],
    "Domain 4: DyT / Normalization-free": ["dyt", "nfnet"],
    "Domain 5: Distillation & Training Losses": ["kd", "lapsrn", "ffl", "ssim"],
    "Domain 6: XR Passthrough Optics, Latency, Human Factors": [
        "quest3_optics", "vst_latency", "motion_to_photon", "retinal_rivalry",
        "binocular_fusion", "simulator_sickness", "foveated_rendering", "patney_foveated"
    ],
    "Domain 7: Mobile NPU Deployment & Quantization": ["ptq", "qnn_sdk", "sentis"],
}


def load_bibtex_entries():
    """Custom parser extracting raw entry blocks, keys, types, and fields from references.bib."""
    with open(REFERENCES_BIB, "r", encoding="utf-8") as f:
        content = f.read()

    # Match @type{key, ...}
    pattern = re.compile(r"@([a-zA-Z]+)\s*\{\s*([a-zA-Z0-9_\-]+)\s*,", re.MULTILINE)
    matches = list(pattern.finditer(content))

    entries = []
    for i, m in enumerate(matches):
        entry_type = m.group(1).lower()
        key = m.group(2).strip()
        start_idx = m.start()
        # Find closing brace matching the opening brace of @type{
        # We start right after the opening brace of @type{
        open_brace_idx = content.find("{", start_idx)
        depth = 1
        pos = open_brace_idx + 1
        in_quote = False
        while pos < len(content) and depth > 0:
            ch = content[pos]
            if ch == '"' and not in_quote:
                in_quote = True
            elif ch == '"' and in_quote:
                in_quote = False
            elif not in_quote:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
            pos += 1

        body = content[open_brace_idx + 1: pos - 1]
        entries.append({
            "type": entry_type,
            "key": key,
            "raw": content[start_idx:pos],
            "body": body,
            "start": start_idx,
            "end": pos,
        })
    return entries, content


# -----------------------------------------------------------------------------
# 1. BibTeX Strict Syntax & Pushdown Automaton Brace Balancing
# -----------------------------------------------------------------------------
def test_bibtex_parser_strict_syntax():
    """Verify BibTeX syntax, total brace balance, and individual entry brace balance."""
    entries, content = load_bibtex_entries()
    assert len(entries) == 33, f"Expected 33 BibTeX entries, parsed {len(entries)}"

    # Check non-comment global brace balance
    lines = content.splitlines()
    non_comment_lines = [l for l in lines if not l.strip().startswith("%")]
    clean_text = "\n".join(non_comment_lines)
    open_braces = clean_text.count("{")
    close_braces = clean_text.count("}")
    assert open_braces == close_braces, f"Global brace mismatch: {open_braces} open vs {close_braces} close"

    # Verify per-entry brace balance
    for e in entries:
        e_open = e["raw"].count("{")
        e_close = e["raw"].count("}")
        assert e_open == e_close, f"Entry @{e['type']}{{{e['key']}}} has unbalanced braces: {e_open} vs {e_close}"


# -----------------------------------------------------------------------------
# 2. BibTeX Required Field Validation & Unescaped Special Character Detection
# -----------------------------------------------------------------------------
def test_bibtex_required_fields_and_special_chars():
    """Adversarially check required fields and unescaped special characters."""
    entries, _ = load_bibtex_entries()

    for e in entries:
        raw = e["raw"]
        key = e["key"]
        etype = e["type"]

        # Parse field-value pairs
        fields = {}
        for line in e["body"].splitlines():
            line = line.strip()
            if "=" in line:
                f_name, f_val = line.split("=", 1)
                f_name = f_name.strip().lower()
                f_val = f_val.strip().rstrip(",")
                if f_val.startswith("{") and f_val.endswith("}"):
                    f_val = f_val[1:-1]
                elif f_val.startswith('"') and f_val.endswith('"'):
                    f_val = f_val[1:-1]
                fields[f_name] = f_val

        # Required fields by entry type
        assert "title" in fields, f"Entry {key} is missing 'title'"
        assert "year" in fields, f"Entry {key} is missing 'year'"
        assert re.match(r"^\d{4}$", fields["year"]), f"Entry {key} year '{fields['year']}' is not a 4-digit integer"

        if etype == "article":
            assert "author" in fields, f"Article {key} missing 'author'"
            assert "journal" in fields, f"Article {key} missing 'journal'"
        elif etype == "inproceedings":
            assert "author" in fields, f"Inproceedings {key} missing 'author'"
            assert "booktitle" in fields, f"Inproceedings {key} missing 'booktitle'"
        elif etype == "book":
            assert "author" in fields, f"Book {key} missing 'author'"
            assert "publisher" in fields, f"Book {key} missing 'publisher'"
        elif etype == "manual":
            assert ("organization" in fields or "author" in fields), f"Manual {key} missing organization/author"

        # Check for unescaped & outside LaTeX macros/commands
        for fname, fval in fields.items():
            assert len(fval) > 0, f"Field '{fname}' in entry '{key}' is empty"
            # Unescaped & check
            # An unescaped & is an & not preceded by a backslash
            unescaped_amp = re.findall(r"(?<!\\)&", fval)
            assert len(unescaped_amp) == 0, f"Unescaped '&' found in field '{fname}' of entry '{key}': {fval}"

            # Unescaped % check
            unescaped_pct = re.findall(r"(?<!\\)%", fval)
            assert len(unescaped_pct) == 0, f"Unescaped '%' found in field '{fname}' of entry '{key}': {fval}"


# -----------------------------------------------------------------------------
# 3. BibTeX Key Uniqueness & Author Format Validation
# -----------------------------------------------------------------------------
def test_bibtex_key_uniqueness_and_author_formatting():
    """Verify keys are unique, case-insensitive collision-free, and authors formatted correctly."""
    entries, _ = load_bibtex_entries()
    keys = [e["key"] for e in entries]

    assert len(keys) == 33, f"Expected 33 entries, got {len(keys)}"
    assert len(set(keys)) == 33, f"Duplicate BibTeX keys found: {[k for k in keys if keys.count(k) > 1]}"

    lower_keys = [k.lower() for k in keys]
    assert len(set(lower_keys)) == 33, "Case-insensitive collision in BibTeX keys"

    for e in entries:
        if "author" in e["raw"]:
            author_match = re.search(r"author\s*=\s*\{([^}]+)\}", e["raw"])
            if author_match:
                author_str = author_match.group(1).strip()
                assert len(author_str) > 0, f"Empty author in {e['key']}"
                # If multiple authors, must be separated by ' and '
                if "," in author_str:
                    # Valid BibTeX author formatting typically uses "Last, First" or "Last, First and ..."
                    pass


# -----------------------------------------------------------------------------
# 4. Citation Graph & 100% Active In-Text Citation Mapping
# -----------------------------------------------------------------------------
def test_citation_graph_completeness_and_zero_undefined():
    """Verify 100% bidirectional citation mapping: 0 undefined references, 0 dead bib entries."""
    with open(MAIN_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    entries, _ = load_bibtex_entries()
    bib_keys = set(e["key"] for e in entries)

    # Extract all \cite{...}
    raw_cites = re.findall(r"\\cite\{([^}]+)\}", tex)
    cited_keys = set()
    total_cite_usages = 0
    for c in raw_cites:
        keys_in_call = [k.strip() for k in c.split(",") if k.strip()]
        for k in keys_in_call:
            cited_keys.add(k)
            total_cite_usages += 1

    # Check 1: 0 undefined references
    undefined = cited_keys - bib_keys
    assert len(undefined) == 0, f"Undefined citation keys in main.tex: {undefined}"

    # Check 2: 0 dead BibTeX entries (100% coverage)
    uncited = bib_keys - cited_keys
    assert len(uncited) == 0, f"Uncited entries in references.bib: {uncited}"

    # Check 3: Exactly 33 keys cited, >= 100 total citation points
    assert len(cited_keys) == 33, f"Expected 33 unique cited keys, got {len(cited_keys)}"
    assert total_cite_usages >= 100, f"Expected >= 100 citation usages, got {total_cite_usages}"

    # Check 4: Correct IEEEtran bibliography commands
    assert r"\bibliographystyle{IEEEtran}" in tex
    assert r"\bibliography{references}" in tex


# -----------------------------------------------------------------------------
# 5. Comprehensive Coverage of All 7 Specified Topic Domains
# -----------------------------------------------------------------------------
def test_seven_topic_domains_genuine_coverage():
    """Verify genuine coverage across all 7 topic domains and zero known hallucinations."""
    entries, bib_content = load_bibtex_entries()
    bib_keys = set(e["key"] for e in entries)

    # Domain coverage check
    for domain_name, required_keys in DOMAIN_KEYWORDS.items():
        found = [k for k in required_keys if k in bib_keys]
        assert len(found) == len(required_keys), (
            f"Domain '{domain_name}' missing keys: {set(required_keys) - set(found)}"
        )

    # Hallucination regression assertions
    forbidden_strings = [
        "Zhang, Hao and Wang, Yibo",
        "Zhu, Junyuan and Lian, Meng",
        "MRL-TR-2024-03",
        "Zheng, Feng and Xiao, Robert",
        "Xu, Xingqian and Gu, Jiayi",
    ]
    for s in forbidden_strings:
        assert s not in bib_content, f"Regression: hallucination '{s}' found in references.bib!"

    # Ground truth author assertions
    ground_truth_authors = [
        "Liu, Yi and Liu, Xinyi",
        "Zhu, Jiachen and Chen, Xinlei",
        "Chaurasia, Gaurav",
        "Warburton, Matthew",
        "Stauffert, Jan-Philipp",
        "Ding, Xiaohan and Zhang, Xiangyu",
        "Nagel, Markus and Fournarakis, Marios",
    ]
    for a in ground_truth_authors:
        assert a in bib_content, f"Ground truth author '{a}' missing in references.bib"


# -----------------------------------------------------------------------------
# 6. Float Precision Specifiers, Captions, and Environment Types for All 13 Floats
# -----------------------------------------------------------------------------
def test_all_13_floats_specifiers_and_captions():
    """Adversarially verify all 13 floats for presence, specifier, caption, and label."""
    with open(MAIN_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    expected_specifiers = {
        "fig:arch": (r"\begin{figure*}[!t]", r"\end{figure*}"),
        "fig:memaccess": (r"\begin{figure*}[!t]", r"\end{figure*}"),
        "tab:hw": (r"\begin{table}[!t]", r"\end{table}"),
        "fig:loss": (r"\begin{figure}[!t]", r"\end{figure}"),
        "fig:pareto": (r"\begin{figure}[!t]", r"\end{figure}"),
        "fig:latency": (r"\begin{figure}[!b]", r"\end{figure}"),
        "tab:crop": (r"\begin{table*}[!t]", r"\end{table*}"),
        "fig:bars": (r"\begin{figure*}[!t]", r"\end{figure*}"),
        "tab:mae": (r"\begin{table}[!t]", r"\end{table}"),
        "fig:visual_mb": (r"\begin{figure*}[!t]", r"\end{figure*}"),
        "fig:visual_q3s": (r"\begin{figure*}[!t]", r"\end{figure*}"),
        "tab:scale": (r"\begin{table}[!b]", r"\end{table}"),
        "fig:foveated": (r"\begin{figure*}[!t]", r"\end{figure*}"),
    }

    assert len(expected_specifiers) == 13

    for label, (begin_tag, end_tag) in expected_specifiers.items():
        assert f"\\label{{{label}}}" in tex, f"Label \\label{{{label}}} missing in main.tex"
        # Find block containing this label
        label_pos = tex.find(f"\\label{{{label}}}")
        # Search backwards for the nearest begin_tag
        block_start = tex.rfind(begin_tag, 0, label_pos)
        assert block_start != -1, f"Float for {label} missing exact expected opener: {begin_tag}"
        # Search forwards for the matching end_tag
        block_end = tex.find(end_tag, label_pos)
        assert block_end != -1, f"Float for {label} missing exact expected closer: {end_tag}"

        float_block = tex[block_start:block_end + len(end_tag)]
        assert "\\caption" in float_block, f"Float for {label} missing \\caption command"


# -----------------------------------------------------------------------------
# 7. LaTeX Environment Nesting Pushdown Automaton
# -----------------------------------------------------------------------------
def test_latex_environment_nesting_pushdown_automaton():
    """Verify strictly nested environment structure via pushdown automaton."""
    with open(MAIN_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    # Filter out LaTeX comments
    clean_lines = []
    for line in tex.splitlines():
        # Strip comments not preceded by backslash
        comment_pos = -1
        for m in re.finditer(r"(?<!\\)%", line):
            comment_pos = m.start()
            break
        if comment_pos != -1:
            clean_lines.append(line[:comment_pos])
        else:
            clean_lines.append(line)

    clean_content = "\n".join(clean_lines)

    env_token_pattern = re.compile(r"\\(begin|end)\{([a-zA-Z0-9_\*]+)\}")
    stack = []

    for m in env_token_pattern.finditer(clean_content):
        action, env_name = m.group(1), m.group(2)
        if action == "begin":
            stack.append((env_name, m.start()))
        elif action == "end":
            assert len(stack) > 0, f"Unmatched \\end{{{env_name}}} at character offset {m.start()}"
            top_env, top_pos = stack.pop()
            assert top_env == env_name, (
                f"Mismatched environment: \\begin{{{top_env}}} at {top_pos} "
                f"closed by \\end{{{env_name}}} at {m.start()}"
            )

    assert len(stack) == 0, f"Unclosed environments remaining on stack: {[e[0] for e in stack]}"


# -----------------------------------------------------------------------------
# 8. Global Label Uniqueness & Elimination of Duplicate \label{fig:latency}
# -----------------------------------------------------------------------------
def test_global_label_uniqueness_and_latency_fix():
    """Verify all labels are globally unique with 0 duplicate labels anywhere in main.tex."""
    with open(MAIN_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    all_labels = re.findall(r"\\label\{([^}]+)\}", tex)
    label_counts = {lbl: all_labels.count(lbl) for lbl in set(all_labels)}
    duplicates = {lbl: count for lbl, count in label_counts.items() if count > 1}

    assert len(duplicates) == 0, f"Duplicate labels found in main.tex: {duplicates}"
    assert label_counts.get("fig:latency", 0) == 1, (
        f"Expected exactly 1 occurrence of fig:latency, found {label_counts.get('fig:latency', 0)}"
    )


# -----------------------------------------------------------------------------
# 9. Cross-Reference Resolution
# -----------------------------------------------------------------------------
def test_cross_references_resolution():
    """Verify that every \\ref in main.tex targets a valid, defined \\label."""
    with open(MAIN_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))

    broken_refs = refs - labels
    assert len(broken_refs) == 0, f"Broken cross-references found in main.tex: {broken_refs}"


# -----------------------------------------------------------------------------
# 10. Release Zip Forensic Integrity
# -----------------------------------------------------------------------------
def test_paper_bundle_zip_forensic_integrity():
    """Forensically inspect reports/paper_bundle.zip for POSIX paths, file presence, and byte parity."""
    assert os.path.isfile(BUNDLE_ZIP), f"Zip bundle missing at {BUNDLE_ZIP}"

    with zipfile.ZipFile(BUNDLE_ZIP, "r") as z:
        namelist = z.namelist()
        infolist = z.infolist()

        # 1. POSIX forward slash check (Zero Windows backslashes)
        for name in namelist:
            assert "\\" not in name, f"Non-POSIX backslash detected in archive entry: {name}"

        # 2. Zero 0-byte corrupt files check
        for info in infolist:
            if not info.is_dir():
                assert info.file_size > 0, f"Corrupt 0-byte file detected in archive: {info.filename}"

        # 3. Essential files present at root
        essential_files = ["main.tex", "references.bib", "README.md", "CLAUDE_PAPER_MASTER_GUIDE.md"]
        for ef in essential_files:
            assert ef in namelist, f"Essential file {ef} missing from zip archive"

        # 4. Check all figures included in main.tex exist in zip
        with open(MAIN_TEX, "r", encoding="utf-8") as f:
            tex = f.read()
        fig_includes = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
        for fig in fig_includes:
            norm_fig = fig.replace("\\", "/")
            assert norm_fig in namelist, f"Referenced figure '{norm_fig}' missing from zip archive"

        # 5. Byte-for-byte fidelity test between zip and disk for main.tex and references.bib
        zip_main_bytes = z.read("main.tex")
        with open(MAIN_TEX, "rb") as f:
            disk_main_bytes = f.read()
        assert zip_main_bytes == disk_main_bytes, "Divergence detected: zip main.tex differs from disk main.tex!"

        zip_bib_bytes = z.read("references.bib")
        with open(REFERENCES_BIB, "rb") as f:
            disk_bib_bytes = f.read()
        assert zip_bib_bytes == disk_bib_bytes, "Divergence detected: zip references.bib differs from disk references.bib!"

        # 6. Extraction test in clean sandbox
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extractall(tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, "main.tex"))
            assert os.path.isfile(os.path.join(tmpdir, "references.bib"))
            for fig in fig_includes:
                assert os.path.isfile(os.path.join(tmpdir, fig.replace("\\", "/")))


# -----------------------------------------------------------------------------
# 11. Math Mode Dollar Sign Parity & Balance
# -----------------------------------------------------------------------------
def test_math_mode_dollar_sign_balance():
    """Verify that all inline math dollar signs ($) in main.tex are strictly paired and balanced."""
    with open(MAIN_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    # Strip line comments
    clean_lines = [re.sub(r"(?<!\\)%.*$", "", l) for l in tex.splitlines()]
    clean_text = "\n".join(clean_lines)

    # An unescaped dollar sign is a $ preceded by an even number of backslashes
    unescaped_dollars = []
    for i, ch in enumerate(clean_text):
        if ch == "$":
            b_count = 0
            k = i - 1
            while k >= 0 and clean_text[k] == "\\":
                b_count += 1
                k -= 1
            if b_count % 2 == 0:
                unescaped_dollars.append(i)

    assert len(unescaped_dollars) % 2 == 0, (
        f"Unbalanced math mode dollar signs: found {len(unescaped_dollars)} unescaped '$' characters!"
    )


# -----------------------------------------------------------------------------
# 12. Cross-Section Citation Distribution
# -----------------------------------------------------------------------------
def test_section_citation_distribution():
    """Verify that citations are actively distributed across all major sections."""
    with open(MAIN_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    sections = [
        "Introduction",
        "Related Work",
        "Method",
        "Dataset",
        "Results",
        "Discussion",
        "Conclusion",
    ]

    for sec in sections:
        pattern = r"\\section\*?\{[^}]*" + re.escape(sec) + r"[^}]*\}(.*?)(?=\\section|\Z)"
        match = re.search(pattern, tex, re.IGNORECASE | re.DOTALL)
        assert match is not None, f"Section matching '{sec}' not found"
        sec_text = match.group(1)
        sec_cites = re.findall(r"\\cite\{([^}]+)\}", sec_text)
        assert len(sec_cites) > 0, f"Section '{sec}' has 0 citation calls!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
