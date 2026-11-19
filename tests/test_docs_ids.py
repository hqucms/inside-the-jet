"""The question IDs cited outside TASKS.md (README, docs, CHANGELOG, notebooks, examples, src) exist in TASKS.md.

Release 0.2.0 renumbered the questions (``T1.3`` became ``W1.2.4`` and others; docs/id_mapping_0.2.md). An ID of
0.1.x left in a document or a help string points a student to the wrong question, so none may remain outside the
mapping and the CHANGELOG entries of 0.1.x.
"""

import json
import re
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "TASKS.md"
MAPPING = ROOT / "docs" / "id_mapping_0.2.md"
GLOSSARY = ROOT / "docs" / "glossary.md"

pytestmark = pytest.mark.skipif(not TASKS.exists(), reason="needs the repository checkout")

#: A question (``W1.2.4``) or task (``W1.2``) of 0.2.0; a bonus question ``B.1`` to ``B.5``.
NEW_ID = re.compile(r"(?<![\w.])(W[1-4]\.\d(?:\.\d{1,2})?|B\.\d{1,2})(?![\w]|\.\d)")
#: A question or task of 0.1.x (``T1.3``, ``T6``) when it is cited as one: ``T1.3``, ``T1.3(a)``, "task T4".
OLD_ID = re.compile(r"(?<![\w.$\\{])T[0-7]\.\d\b|\btasks? T[0-7]\b|\bT[0-7] (?:to|and|or) T[0-7]\b")
#: The one way to name the old IDs outside the mapping: their range, in a sentence that points to the mapping.
OLD_RANGE = re.compile(r"\(?T0\.1(?: to |-)T7\.4\)?")


@cache
def _known() -> set[str]:
    """Every question and task ID of TASKS.md."""
    text = TASKS.read_text(encoding="utf-8")
    questions = set(re.findall(r"^\*\*(W[1-4]\.\d\.\d{1,2}|B\.\d)\*\*", text, flags=re.MULTILINE))
    return questions | {q.rsplit(".", 1)[0] for q in questions if q.startswith("W")}


def _changelog_current() -> str:
    """The CHANGELOG text above the first entry of 0.1.x (the entries of 0.1.x keep their IDs)."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## \[0\.1\.", text, flags=re.MULTILINE)
    return text[: match.start()] if match else text


def _notebook_text(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells)


@cache
def _documents() -> dict[str, str]:
    """The texts that cite question IDs, by their path relative to the repository root."""
    files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted((ROOT / "examples").rglob("*.md"))]
    files += sorted((ROOT / "examples").rglob("*.py")) + sorted((ROOT / "src" / "itj").rglob("*.py"))
    out = {
        p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8")
        for p in files
        if p.exists() and p != MAPPING and "__pycache__" not in p.parts
    }
    out.update(
        {p.relative_to(ROOT).as_posix(): _notebook_text(p) for p in sorted((ROOT / "notebooks").glob("*.ipynb"))}
    )
    return out


def _where(text: str, start: int) -> int:
    return text.count("\n", 0, start) + 1


def test_cited_ids_exist():
    unknown = []
    documents = {**_documents(), "CHANGELOG.md": _changelog_current()}
    for name, text in documents.items():
        for match in NEW_ID.finditer(text):
            if match.group(1) not in _known():
                unknown.append(f"{name}:{_where(text, match.start())}: {match.group(1)}")
    assert not unknown, "IDs that TASKS.md does not have:\n" + "\n".join(unknown)


def test_no_old_ids_left():
    left = []
    for name, text in _documents().items():
        text = OLD_RANGE.sub("", text)
        for match in OLD_ID.finditer(text):
            left.append(f"{name}:{_where(text, match.start())}: {match.group(0)}")
    assert not left, f"IDs of 0.1.x (map them with {MAPPING.name}):\n" + "\n".join(left)


def test_glossary_names_the_first_use():
    """Every glossary entry names the question or task of its first use, and that ID exists."""
    if not GLOSSARY.exists():
        pytest.skip("no glossary")
    text = GLOSSARY.read_text(encoding="utf-8")
    entries = re.findall(r"^- \*\*(.+?)\*\*(.*?)(?=^- \*\*|^#|\Z)", text, flags=re.MULTILINE | re.DOTALL)
    assert entries
    without = [term for term, body in entries if not NEW_ID.search(body)]
    assert not without, f"glossary entries without the ID of their first use: {without}"
