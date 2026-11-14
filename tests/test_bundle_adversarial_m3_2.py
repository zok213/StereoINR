r"""
Adversarial Release Bundle Stress-Testing, Layout Validation, and Unpack-Compilation Verification Harness.
Milestone 3 - Challenger 2 (teamwork_challenger_m3_2)

Validates:
1. Binary-level ZIP integrity of reports/paper_bundle.zip (POSIX slashes, no backslashes, CRC check, root files).
2. Clean isolated unpack in temp directory and resolution of 100% of \includegraphics targets on disk.
3. Complete audit of all 13 LaTeX floats:
   - Presence of labels: fig:arch, fig:memaccess, tab:hw, fig:loss, fig:pareto, fig:latency, tab:crop,
     fig:bars, tab:mae, fig:visual_mb, fig:visual_q3s, tab:scale, fig:foveated.
   - Placement constraints: [!b] for fig:latency and tab:scale; [!t] for others.
   - Margin compliance: single-column floats <= \columnwidth, two-column floats <= \textwidth.
   - No duplicate labels or dangling environments.
4. IEEEtran preamble float fraction and counter parameters.
5. Bidirectional bibliography mapping and Overleaf readiness.
6. Binary validation of all 23 bundle PNG assets (magic bytes, dimensions).
7. Float sequence ordering and LaTeX float specifier safety (no [b] on wide figure* / table*).
"""

import os
import re
import struct
import tempfile
import zipfile
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_ZIP = os.path.join(REPO_ROOT, "reports", "paper_bundle.zip")

ALL_13_FLOAT_LABELS = [
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

# Chronological narrative order corresponding to Sections III-A, III-B, V-A, V-B, V-B, V-B, V-C, V-C, V-D, V-E, V-E, V-F, VI
EXPECTED_NARRATIVE_ORDER = [
    "fig:arch",        # Sec III-A
    "fig:memaccess",   # Sec III-B
    "fig:loss",        # Sec V-A
    "tab:hw",          # Sec V-B
    "fig:pareto",      # Sec V-B
    "fig:latency",     # Sec V-B
    "tab:crop",        # Sec V-C
    "fig:bars",        # Sec V-C
    "tab:mae",         # Sec V-D
    "fig:visual_mb",   # Sec V-E
    "fig:visual_q3s",  # Sec V-E
    "tab:scale",       # Sec V-F
    "fig:foveated",    # Sec VI
]


def test_zip_binary_integrity_and_posix_paths():
    """Adversarial check: Assert 100% POSIX paths, zero backslashes, CRC test, and root files."""
    assert os.path.isfile(BUNDLE_ZIP), f"Bundle does not exist at {BUNDLE_ZIP}"
    file_size = os.path.getsize(BUNDLE_ZIP)
    assert file_size > 1024 * 1024, f"Bundle suspiciously small: {file_size} bytes"

    with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
        # 1. CRC test
        corrupt = zf.testzip()
        assert corrupt is None, f"Zip CRC test failed on entry: {corrupt}"

        namelist = zf.namelist()
        assert len(namelist) >= 25, f"Expected >= 25 entries, found {len(namelist)}"

        # 2. Strict POSIX path assertion
        backslash_entries = [name for name in namelist if "\\" in name]
        assert len(backslash_entries) == 0, f"Found Windows backslashes in zip: {backslash_entries}"

        # 3. Path traversal attack check
        traversal_entries = [name for name in namelist if name.startswith("/") or "../" in name or "..\\" in name]
        assert len(traversal_entries) == 0, f"Zip slip / path traversal entries detected: {traversal_entries}"

        # 4. Mandatory root files
        assert "main.tex" in namelist, "main.tex missing from archive root"
        assert "references.bib" in namelist, "references.bib missing from archive root"

        # 5. Non-zero entry sizes
        for info in zf.infolist():
            assert info.file_size > 0, f"Empty entry found in zip: {info.filename}"


def test_all_bundle_image_assets_binary_validity():
    """Adversarial check: Assert all 23 figures/ PNG assets have valid PNG signatures and positive dimensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
            zf.extractall(tmpdir)

        figures_dir = os.path.join(tmpdir, "figures")
        assert os.path.isdir(figures_dir), "figures directory missing in unpacked bundle"

        png_files = [f for f in os.listdir(figures_dir) if f.endswith(".png")]
        assert len(png_files) == 23, f"Expected exactly 23 PNG files in figures/, found {len(png_files)}"

        for png_name in png_files:
            png_path = os.path.join(figures_dir, png_name)
            assert os.path.getsize(png_path) > 1000, f"Image {png_name} too small (<1KB)"

            with open(png_path, "rb") as f:
                header = f.read(24)
                assert header[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG magic bytes for {png_name}"
                assert header[12:16] == b"IHDR", f"Missing IHDR chunk in {png_name}"
                width, height = struct.unpack(">II", header[16:24])
                assert width > 0 and height > 0, f"Invalid dimensions {width}x{height} for {png_name}"


def test_isolated_unpack_and_graphics_resolution():
    """Adversarial check: Unpack to clean isolated temp dir and verify all \\includegraphics targets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
            zf.extractall(tmpdir)

        unpacked_main = os.path.join(tmpdir, "main.tex")
        unpacked_bib = os.path.join(tmpdir, "references.bib")
        assert os.path.isfile(unpacked_main), "main.tex not extracted"
        assert os.path.isfile(unpacked_bib), "references.bib not extracted"

        with open(unpacked_main, "r", encoding="utf-8") as f:
            tex_content = f.read()

        graphics_matches = re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", tex_content)
        assert len(graphics_matches) > 0, "No \\includegraphics calls found in main.tex"

        for graphic_rel in graphics_matches:
            assert "\\" not in graphic_rel, f"Non-POSIX path in \\includegraphics: {graphic_rel}"
            graphic_full = os.path.join(tmpdir, graphic_rel)
            assert os.path.isfile(graphic_full), f"Referenced graphic does not exist on disk: {graphic_rel}"
            assert os.path.getsize(graphic_full) > 1000, f"Referenced graphic is truncated/empty: {graphic_rel}"


def test_ieeetran_preamble_float_parameters():
    """Adversarial check: Verify that IEEEtran float fractions and counters are correctly configured in preamble."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
            zf.extractall(tmpdir)

        with open(os.path.join(tmpdir, "main.tex"), "r", encoding="utf-8") as f:
            tex_content = f.read()

        doc_begin = tex_content.find(r"\begin{document}")
        assert doc_begin != -1, "Missing \\begin{document}"
        preamble = tex_content[:doc_begin]

        expected_params = [
            (r"\renewcommand{\topfraction}{0.90}", r"\\renewcommand\{\\topfraction\}\{0\.90\}"),
            (r"\renewcommand{\bottomfraction}{0.80}", r"\\renewcommand\{\\bottomfraction\}\{0\.80\}"),
            (r"\renewcommand{\textfraction}{0.05}", r"\\renewcommand\{\\textfraction\}\{0\.05\}"),
            (r"\renewcommand{\floatpagefraction}{0.85}", r"\\renewcommand\{\\floatpagefraction\}\{0\.85\}"),
            (r"\renewcommand{\dbltopfraction}{0.90}", r"\\renewcommand\{\\dbltopfraction\}\{0\.90\}"),
            (r"\renewcommand{\dblfloatpagefraction}{0.85}", r"\\renewcommand\{\\dblfloatpagefraction\}\{0\.85\}"),
            (r"\setcounter{topnumber}{3}", r"\\setcounter\{topnumber\}\{3\}"),
            (r"\setcounter{bottomnumber}{2}", r"\\setcounter\{bottomnumber\}\{2\}"),
            (r"\setcounter{totalnumber}{5}", r"\\setcounter\{totalnumber\}\{5\}"),
            (r"\setcounter{dbltopnumber}{3}", r"\\setcounter\{dbltopnumber\}\{3\}"),
        ]

        for name, pattern in expected_params:
            assert re.search(pattern, preamble) is not None, f"Missing or incorrect IEEEtran preamble parameter: {name}"


def _extract_float_blocks(tex_content):
    """Helper to extract all figure, figure*, table, and table* environments with metadata."""
    pattern = re.compile(
        r"(\\begin\{(figure\*?|table\*?)\}(?:\[([^\]]*)\])?(.*?)\\end\{\2\})",
        re.DOTALL
    )
    blocks = []
    for match in pattern.finditer(tex_content):
        full_block = match.group(1)
        env_name = match.group(2)
        specifier = match.group(3) or ""
        body = match.group(4)

        labels = re.findall(r"\\label\{([^}]+)\}", body)
        blocks.append({
            "full_block": full_block,
            "env_name": env_name,
            "specifier": specifier,
            "body": body,
            "labels": labels,
            "start": match.start(),
            "end": match.end(),
        })
    return blocks


def test_all_13_floats_presence_and_uniqueness():
    """Adversarial check: Assert all 13 required floats exist exactly once without label duplication."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
            zf.extractall(tmpdir)

        with open(os.path.join(tmpdir, "main.tex"), "r", encoding="utf-8") as f:
            tex_content = f.read()

        blocks = _extract_float_blocks(tex_content)
        all_labels_in_floats = [lbl for b in blocks for lbl in b["labels"]]

        for required_label in ALL_13_FLOAT_LABELS:
            count_in_floats = all_labels_in_floats.count(required_label)
            assert count_in_floats == 1, (
                f"Float label '{required_label}' expected exactly 1 occurrence in float environments, "
                f"found {count_in_floats}"
            )
            count_in_entire_doc = len(re.findall(r"\\label\{" + re.escape(required_label) + r"\}", tex_content))
            assert count_in_entire_doc == 1, (
                f"Float label '{required_label}' duplicated in document: {count_in_entire_doc} occurrences"
            )


def test_float_placement_specifiers_and_bottom_anchors():
    """
    Adversarial check:
    Verify that bottom-placement ([!b]) is strictly used for fig:latency and tab:scale
    to prevent top-of-column congestion, and verify placement for all 13 floats.
    Crucially asserts that wide floats (figure*, table*) do NOT use [b] (which is invalid in LaTeX).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
            zf.extractall(tmpdir)

        with open(os.path.join(tmpdir, "main.tex"), "r", encoding="utf-8") as f:
            tex_content = f.read()

        blocks = _extract_float_blocks(tex_content)
        label_to_block = {}
        for b in blocks:
            for lbl in b["labels"]:
                label_to_block[lbl] = b

        # 1. Mandatory bottom-placement floats
        for bottom_label in ["fig:latency", "tab:scale"]:
            assert bottom_label in label_to_block, f"Missing float {bottom_label}"
            blk = label_to_block[bottom_label]
            assert blk["specifier"] == "!b", (
                f"Float '{bottom_label}' MUST use [!b] bottom placement to relieve top congestion, "
                f"got '[{blk['specifier']}]'"
            )
            # Must be single-column
            assert blk["env_name"] in ["figure", "table"], (
                f"Float '{bottom_label}' using [!b] must be single-column (LaTeX disallows figure*[b]), "
                f"got {blk['env_name']}"
            )

        # 2. Top-placement floats
        top_labels = [lbl for lbl in ALL_13_FLOAT_LABELS if lbl not in ["fig:latency", "tab:scale"]]
        for top_label in top_labels:
            assert top_label in label_to_block, f"Missing float {top_label}"
            blk = label_to_block[top_label]
            assert blk["specifier"] == "!t", (
                f"Float '{top_label}' expected [!t] placement, got '[{blk['specifier']}]'"
            )


def test_float_dimensions_and_margin_confinement():
    """
    Adversarial check:
    Verify that no float exceeds columnwidth (for 1-column floats) or textwidth (for 2-column floats).
    Asserts absence of unconstrained wide dimensions in single-column floats.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
            zf.extractall(tmpdir)

        with open(os.path.join(tmpdir, "main.tex"), "r", encoding="utf-8") as f:
            tex_content = f.read()

        blocks = _extract_float_blocks(tex_content)
        label_to_block = {}
        for b in blocks:
            for lbl in b["labels"]:
                label_to_block[lbl] = b

        single_col_floats = ["fig:loss", "tab:hw", "fig:pareto", "fig:latency", "tab:mae", "tab:scale"]
        two_col_floats = ["fig:arch", "fig:memaccess", "tab:crop", "fig:bars", "fig:visual_mb", "fig:visual_q3s", "fig:foveated"]

        # Check single-column floats
        for label in single_col_floats:
            blk = label_to_block[label]
            assert blk["env_name"] in ["figure", "table"], f"Float {label} should be single-column (figure/table), got {blk['env_name']}"
            body = blk["body"]

            # Single-column floats must NOT contain width=\textwidth or resizebox{\textwidth}
            assert r"\textwidth" not in body, f"Float {label} mistakenly references \\textwidth in a single-column environment!"

            # If resizebox is used, it must be \columnwidth
            resizes = re.findall(r"\\resizebox\{([^}]+)\}", body)
            for r_dim in resizes:
                assert r_dim == r"\columnwidth" or r_dim.endswith(r"\columnwidth"), (
                    f"Float {label} resizebox dimension exceeds columnwidth: {r_dim}"
                )

        # Check two-column floats
        for label in two_col_floats:
            blk = label_to_block[label]
            assert blk["env_name"] in ["figure*", "table*"], f"Float {label} should be two-column (figure*/table*), got {blk['env_name']}"
            body = blk["body"]

            # Check minipages in fig:foveated
            if label == "fig:foveated":
                minipages = re.findall(r"\\begin\{minipage\}(?:\[[^\]]*\])?\{([^}]+)\}", body)
                assert len(minipages) == 2, f"fig:foveated expected 2 minipages, found {len(minipages)}"
                total_frac = 0.0
                for mp in minipages:
                    m = re.match(r"([0-9\.]+)\\textwidth", mp)
                    assert m is not None, f"fig:foveated minipage width not in \\textwidth: {mp}"
                    total_frac += float(m.group(1))
                assert total_frac <= 1.0, f"fig:foveated minipages sum to {total_frac} > 1.0\\textwidth"

            # Check resizebox or includegraphics
            resizes = re.findall(r"\\resizebox\{([^}]+)\}", body)
            for r_dim in resizes:
                m = re.match(r"([0-9\.]*)?\\textwidth", r_dim)
                assert m is not None, f"Float {label} resizebox dimension not within \\textwidth: {r_dim}"
                frac = float(m.group(1)) if m.group(1) else 1.0
                assert frac <= 1.0, f"Float {label} resizebox exceeds textwidth: {frac}"

            includes = re.findall(r"\\includegraphics\[([^\]]+)\]", body)
            for opt in includes:
                m = re.search(r"width=([0-9\.]*)?\\textwidth", opt)
                if m:
                    frac = float(m.group(1)) if m.group(1) else 1.0
                    assert frac <= 1.0, f"Float {label} includegraphics exceeds textwidth: {frac}"


def test_float_logical_sequence_order():
    """Adversarial check: Assert that floats appear in document source in logical sequential section order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
            zf.extractall(tmpdir)

        with open(os.path.join(tmpdir, "main.tex"), "r", encoding="utf-8") as f:
            tex_content = f.read()

        positions = {}
        for lbl in EXPECTED_NARRATIVE_ORDER:
            match = re.search(r"\\label\{" + re.escape(lbl) + r"\}", tex_content)
            assert match is not None, f"Label {lbl} missing"
            positions[lbl] = match.start()

        # Check order matches expected narrative flow
        sorted_labels = sorted(EXPECTED_NARRATIVE_ORDER, key=lambda l: positions[l])
        assert sorted_labels == EXPECTED_NARRATIVE_ORDER, (
            f"Floats declared out of narrative order. Observed order: {sorted_labels}"
        )


def test_latex_environment_structural_balance():
    """Adversarial check: Assert all environments have exact matching begins and ends."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
            zf.extractall(tmpdir)

        with open(os.path.join(tmpdir, "main.tex"), "r", encoding="utf-8") as f:
            tex_content = f.read()

        monitored_envs = [
            "document",
            "figure",
            "figure*",
            "table",
            "table*",
            "tikzpicture",
            "axis",
            "tabular",
            "minipage",
            "abstract",
            "IEEEkeywords",
        ]

        for env in monitored_envs:
            begins = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", tex_content))
            ends = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", tex_content))
            assert begins == ends, f"Environment '{env}' unbalanced: {begins} begins vs {ends} ends"
            assert begins > 0, f"Expected environment '{env}' to be present"


def test_bibliography_and_citation_bidirectional_integrity():
    """Adversarial check: Assert 100% mutual coverage between references.bib and main.tex in unpacked bundle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
            zf.extractall(tmpdir)

        with open(os.path.join(tmpdir, "references.bib"), "r", encoding="utf-8") as f:
            bib_content = f.read()
        with open(os.path.join(tmpdir, "main.tex"), "r", encoding="utf-8") as f:
            tex_content = f.read()

        bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_content))
        raw_cites = re.findall(r"\\cite\{([^}]+)\}", tex_content)
        cited_keys = set()
        for group in raw_cites:
            for k in group.split(","):
                k_clean = k.strip()
                if k_clean:
                    cited_keys.add(k_clean)

        uncited = bib_keys - cited_keys
        undefined = cited_keys - bib_keys

        assert len(uncited) == 0, f"Found uncited BibTeX entries in bundle: {uncited}"
        assert len(undefined) == 0, f"Found undefined citations in main.tex: {undefined}"
        assert len(cited_keys) == 33, f"Expected exactly 33 verified citations, got {len(cited_keys)}"

        # Verify IEEEtran commands
        assert r"\bibliographystyle{IEEEtran}" in tex_content
        assert r"\bibliography{references}" in tex_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
