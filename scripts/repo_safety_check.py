"""Repository-level safety checks for MarketFusion AI.

The script intentionally uses only the Python standard library so it can run
before project dependencies are installed. It checks tracked-file hygiene and
protects the current read-only research stage from accidental trading code.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py", ".java", ".xml", ".md", ".txt", ".json", ".yml", ".yaml",
    ".toml", ".ini", ".cfg", ".properties", ".ps1", ".sh", ".csv",
}

# Match executable/API-use shapes rather than generic words. Several read-only
# Java classes deliberately mention "IEngine" in comments to document that they
# never obtain it; matching the bare word would make the safety check fail on the
# very comments that explain the safety boundary.
FORBIDDEN_TRADING_PATTERNS = {
    "MetaTrader5 order_send": re.compile(r"\b(?:mt5\s*\.\s*)?order_send\s*\(", re.IGNORECASE),
    "JForex IEngine import": re.compile(r"^\s*import\s+com\.dukascopy\.api\.IEngine\s*;", re.MULTILINE),
    "JForex getEngine": re.compile(r"\bgetEngine\s*\("),
    "JForex submitOrder": re.compile(r"\bsubmitOrder\s*\("),
    "direct Dukascopy BI5 scraping": re.compile(r"https?://[^\s\"']+\.bi5\b", re.IGNORECASE),
}

# High-signal credential assignment forms. Horizontal whitespace is used around
# '=' deliberately: '\\s*' would cross a newline after an empty value such as
# BLS_API_KEY= and incorrectly consume the next Markdown/code-fence line.
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?im)^[ \t]*(FRED_API_KEY|BLS_API_KEY|OANDA_API_TOKEN|OANDA_ACCOUNT_ID|"
    r"DUKASCOPY_USER|DUKASCOPY_PASSWORD)[ \t]*=[ \t]*([^#\r\n]*)[ \t]*$"
)
ALLOWED_PLACEHOLDER_FRAGMENTS = (
    "replace_with_",
    "your_private_",
    "your_personal_",
    "your_v20_",
    "...",
)

REQUIRED_GITIGNORE_LINES = {
    ".env",
    ".env.*",
    "!.env.example",
    "*.parquet",
    "data/dukascopy/",
    "data/mt5/ticks/",
    "data/mt5/history/",
    "jforex-event-exporter/target/",
}


def run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def tracked_files() -> list[Path]:
    files: list[Path] = []
    for raw in run_git("ls-files").splitlines():
        raw = raw.strip()
        if raw:
            files.append(ROOT / raw)
    return files


def is_text_candidate(path: Path) -> bool:
    if path.name in {".gitignore", ".env.example"}:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def check_tracked_secret_files(files: list[Path], errors: list[str]) -> None:
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        name = path.name.lower()
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            errors.append(f"tracked secret environment file: {rel}")
        if path.suffix.lower() in {".pem", ".p12", ".pfx"}:
            errors.append(f"tracked credential/key container: {rel}")


def check_credential_literals(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if not path.is_file() or not is_text_candidate(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT).as_posix()
        for match in CREDENTIAL_ASSIGNMENT.finditer(text):
            key = match.group(1)
            value = match.group(2).strip().strip("'\"")
            lowered = value.lower()
            if not value or any(fragment in lowered for fragment in ALLOWED_PLACEHOLDER_FRAGMENTS):
                continue
            errors.append(f"possible hard-coded credential {key} in {rel}")


def check_read_only_policy(files: list[Path], errors: list[str]) -> None:
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if not (
            rel.startswith("src/")
            or rel.startswith("jforex-event-exporter/src/main/java/")
        ):
            continue
        if not path.is_file() or not is_text_candidate(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in FORBIDDEN_TRADING_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"read-only policy violation ({label}) in {rel}")


def check_gitignore(errors: list[str]) -> None:
    path = ROOT / ".gitignore"
    if not path.is_file():
        errors.append("missing .gitignore")
        return
    lines = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(REQUIRED_GITIGNORE_LINES.difference(lines))
    if missing:
        errors.append(".gitignore missing required entries: " + ", ".join(missing))


def main() -> int:
    try:
        files = tracked_files()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    errors: list[str] = []
    check_tracked_secret_files(files, errors)
    check_credential_literals(files, errors)
    check_read_only_policy(files, errors)
    check_gitignore(errors)

    print("MARKETFUSION REPOSITORY PREFLIGHT")
    print(f"Tracked files checked: {len(files)}")
    print("Policy: no trading execution code in the current research stage")

    if errors:
        print(f"STATUS: FAIL ({len(errors)} issue(s))")
        for error in errors:
            print(f"- {error}")
        return 1

    print("STATUS: PASS")
    print("- secret-file tracking guard: PASS")
    print("- hard-coded credential guard: PASS")
    print("- read-only trading guard: PASS")
    print("- required .gitignore entries: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
