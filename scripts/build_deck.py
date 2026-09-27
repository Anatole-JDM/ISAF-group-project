"""Build the presentation end to end: numbers -> figures -> .pptx and .pdf.

    python3 scripts/build_deck.py

Both deliverables are rendered from scripts/deck_content.py, which reads only
outputs/deck_stats.json, which is regenerated here from the artifacts. No figure
on any slide is typed in by hand.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mpl")


def run(script: str) -> None:
    print(f"\n→ {script}")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                       capture_output=True, text=True, cwd=ROOT)
    for line in (r.stdout or "").splitlines():
        if "Matplotlib" not in line and "matplotlib" not in line:
            print("   " + line)
    if r.returncode:
        print(r.stderr[-2000:])
        raise SystemExit(f"{script} failed")


if __name__ == "__main__":
    run("deck_stats.py")
    run("deck_figures.py")
    run("render_pptx.py")
    run("render_pdf.py")
    print("\nDeliverables:")
    for f in ("ISAF_presentation.pptx", "ISAF_presentation.pdf"):
        p = ROOT / "reports" / f
        print(f"   {p.relative_to(ROOT)}   {p.stat().st_size/1024:.0f} KB")
