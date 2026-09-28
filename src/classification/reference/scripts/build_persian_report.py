"""Render the Persian Markdown report to a self-contained RTL PDF."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "FINAL_REPORT_FA.md"
STYLE = ROOT / "reports" / "report_fa.css"
OUTPUT = ROOT / "reports" / "FINAL_REPORT_FA.pdf"


def _find_executable(name: str, candidates: tuple[Path, ...]) -> Path:
    discovered = shutil.which(name)
    if discovered:
        return Path(discovered)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"required executable not found: {name}")


def main() -> int:
    pandoc = _find_executable("pandoc", ())
    browser = _find_executable(
        "msedge",
        (
            Path(r"SOURCE_ROOT"),
            Path(r"SOURCE_ROOT"),
            Path(r"SOURCE_ROOT"),
        ),
    )
    if not SOURCE.is_file() or not STYLE.is_file():
        raise FileNotFoundError("Persian report source/style is missing")

    with tempfile.TemporaryDirectory(
        prefix="hmda_report_", dir=ROOT, ignore_cleanup_errors=True
    ) as temporary:
        temp_dir = Path(temporary)
        html = temp_dir / "FINAL_REPORT_FA.html"
        profile = temp_dir / "browser-profile"
        subprocess.run(
            [
                str(pandoc),
                str(SOURCE),
                "--from=gfm+raw_html",
                "--to=html5",
                "--standalone",
                "--embed-resources",
                "--resource-path",
                str(ROOT / "reports"),
                "--css",
                str(STYLE),
                "--metadata",
                "pagetitle=گزارش جامع پروژه HMDA",
                "--output",
                str(html),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-crash-reporter",
                "--disable-breakpad",
                "--disable-extensions",
                "--run-all-compositor-stages-before-draw",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile}",
                f"--print-to-pdf={OUTPUT}",
                html.as_uri(),
            ],
            cwd=ROOT,
            check=True,
        )

    if not OUTPUT.is_file() or OUTPUT.stat().st_size < 10_000:
        raise RuntimeError("PDF generation did not create a valid-sized file")
    if OUTPUT.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("generated output does not have a PDF header")
    print(f"Created {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
