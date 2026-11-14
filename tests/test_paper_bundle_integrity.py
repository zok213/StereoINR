"""
Test Paper Bundle Integrity and Academic Reference Verification
Validates:
1. reports/paper/references.bib: 33 genuine, verified BibTeX entries across 7 domains.
2. reports/paper/main.tex: float positioning, syntax bug fixes, 100% active citations.
3. reports/paper_bundle.zip: POSIX forward slashes, Overleaf compatibility, complete asset bundle.
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


def test_references_bib_exists_and_genuine():
    """Verify that references.bib exists and contains 33 genuine, non-hallucinated entries."""
    assert os.path.isfile(REFERENCES_BIB), "references.bib does not exist"

    with open(REFERENCES_BIB, "r", encoding="utf-8") as f:
        bib_content = f.read()

    entries = re.findall(r"@(\w+)\{([^,]+),", bib_content)
    keys = [k.strip() for _, k in entries]

    assert len(keys) == 33, f"Expected 33 entries in references.bib, got {len(keys)}"
    assert len(set(keys)) == 33, "Duplicate keys found in references.bib"

    # Verify absence of known hallucinations
    assert "Zhang, Hao and Wang, Yibo" not in bib_content, "Hallucinated authors in stereoinr detected!"
    assert "Zhu, Junyuan and Lian, Meng" not in bib_content, "Hallucinated authors in dyt detected!"
    assert "MRL-TR-2024-03" not in bib_content, "Fictitious tech report detected!"
    assert "Zheng, Feng and Xiao, Robert" not in bib_content, "Fictitious TVCG article detected!"

    # Verify presence of ground truth authors
    assert "Liu, Yi and Liu, Xinyi" in bib_content, "Real StereoINR authors missing!"
    assert "Zhu, Jiachen and Chen, Xinlei" in bib_content, "Real DyT authors missing!"
    assert "Chaurasia, Gaurav" in bib_content, "Real Passthrough+ authors missing!"
    assert "Warburton, Matthew" in bib_content, "Real motion-to-photon latency authors missing!"
    assert "Stauffert, Jan-Philipp" in bib_content, "Real VST latency authors missing!"


def test_main_tex_preamble_float_parameters():
    """Verify that main.tex contains the tuned IEEEtran float parameters."""
    with open(MAIN_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    assert r"\renewcommand{\topfraction}{0.90}" in tex
    assert r"\renewcommand{\bottomfraction}{0.80}" in tex
    assert r"\renewcommand{\textfraction}{0.05}" in tex
    assert r"\renewcommand{\floatpagefraction}{0.85}" in tex
    assert r"\renewcommand{\dbltopfraction}{0.90}" in tex
    assert r"\renewcommand{\dblfloatpagefraction}{0.85}" in tex
    assert r"\setcounter{topnumber}{3}" in tex
    assert r"\setcounter{bottomnumber}{2}" in tex
    assert r"\setcounter{totalnumber}{5}" in tex
    assert r"\setcounter{dbltopnumber}{3}" in tex


def test_main_tex_syntax_bug_fix():
    """Verify that duplicate label and extra end-figure in fig:latency are removed."""
    with open(MAIN_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    latency_labels = re.findall(r"\\label\{fig:latency\}", tex)
    assert len(latency_labels) == 1, f"Found {len(latency_labels)} occurrences of fig:latency (expected 1)"

    # Verify environment balance
    for env in ["figure", "figure*", "table", "table*", "tikzpicture", "axis"]:
        begins = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", tex))
        ends = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", tex))
        assert begins == ends, f"Environment {env} unbalanced: {begins} begins vs {ends} ends"


def test_main_tex_precision_float_specifiers():
    """Verify all 13 floats match the precision specifier matrix."""
    with open(MAIN_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    specifier_tests = [
        ("fig:arch", r"\\begin\{figure\*\}\[!t\].*?\\label\{fig:arch\}"),
        ("fig:memaccess", r"\\begin\{figure\*\}\[!t\].*?\\label\{fig:memaccess\}"),
        ("fig:loss", r"\\begin\{figure\}\[!t\].*?\\label\{fig:loss\}"),
        ("tab:hw", r"\\begin\{table\}\[!t\].*?\\label\{tab:hw\}"),
        ("fig:pareto", r"\\begin\{figure\}\[!t\].*?\\label\{fig:pareto\}"),
        ("fig:latency", r"\\begin\{figure\}\[!b\].*?\\label\{fig:latency\}"),
        ("tab:crop", r"\\begin\{table\*\}\[!t\].*?\\label\{tab:crop\}"),
        ("fig:bars", r"\\begin\{figure\*\}\[!t\].*?\\label\{fig:bars\}"),
        ("tab:mae", r"\\begin\{table\}\[!t\].*?\\label\{tab:mae\}"),
        ("fig:visual_mb", r"\\begin\{figure\*\}\[!t\].*?\\label\{fig:visual_mb\}"),
        ("fig:visual_q3s", r"\\begin\{figure\*\}\[!t\].*?\\label\{fig:visual_q3s\}"),
        ("tab:scale", r"\\begin\{table\}\[!b\].*?\\label\{tab:scale\}"),
        ("fig:foveated", r"\\begin\{figure\*\}\[!t\].*?\\label\{fig:foveated\}"),
    ]

    for name, pattern in specifier_tests:
        match = re.search(pattern, tex, re.DOTALL)
        assert match is not None, f"Float specifier check failed for {name}"


def test_citation_integration_and_coverage():
    """Verify 100% active citations with 0 undefined references and 0 uncited entries."""
    with open(MAIN_TEX, "r", encoding="utf-8") as f:
        tex = f.read()
    with open(REFERENCES_BIB, "r", encoding="utf-8") as f:
        bib = f.read()

    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cites = re.findall(r"\\cite\{([^}]+)\}", tex)

    cited_keys = set()
    for c in cites:
        for k in c.split(","):
            cited_keys.add(k.strip())

    not_cited = bib_keys - cited_keys
    invalid_cites = cited_keys - bib_keys

    assert len(not_cited) == 0, f"Uncited BibTeX entries: {not_cited}"
    assert len(invalid_cites) == 0, f"Undefined citations in TeX: {invalid_cites}"
    assert len(cited_keys) == 33, f"Expected 33 unique cited keys, got {len(cited_keys)}"

    # Check IEEEtran bibliography commands
    assert r"\bibliographystyle{IEEEtran}" in tex
    assert r"\bibliography{references}" in tex


def test_paper_bundle_zip_integrity():
    """Verify reports/paper_bundle.zip structure, POSIX paths, and asset completeness."""
    assert os.path.isfile(BUNDLE_ZIP), "reports/paper_bundle.zip does not exist"

    with zipfile.ZipFile(BUNDLE_ZIP, "r") as zipf:
        namelist = zipf.namelist()

        # Check for Windows backslashes
        for name in namelist:
            assert "\\" not in name, f"Found non-POSIX backslash in entry: {name}"

        # Check essential files
        assert "main.tex" in namelist
        assert "references.bib" in namelist
        assert "figures/comparison_middlebury_cloth2.png" in namelist
        assert "figures/comparison_quest3s_harvested_phase7_indoor_portal_awards.png" in namelist

        # Check all figures in main.tex exist in zip
        with open(MAIN_TEX, "r", encoding="utf-8") as f:
            tex = f.read()

        fig_includes = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
        for fig in fig_includes:
            norm_fig = fig.replace("\\", "/")
            assert norm_fig in namelist, f"Referenced figure {norm_fig} missing in paper_bundle.zip"

        # Verify unzipping to temporary folder
        with tempfile.TemporaryDirectory() as tmpdir:
            zipf.extractall(tmpdir)
            extracted_main = os.path.join(tmpdir, "main.tex")
            extracted_bib = os.path.join(tmpdir, "references.bib")
            assert os.path.isfile(extracted_main)
            assert os.path.isfile(extracted_bib)
            for fig in fig_includes:
                assert os.path.isfile(os.path.join(tmpdir, fig.replace("\\", "/")))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
