"""TASKS.md and the documents: question IDs, task structure, data files, links, anchors and no TBC markers."""

import fnmatch
import importlib.util
import re
from pathlib import Path

import pytest

from itj.jetclass import MANIFEST, class_names

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "TASKS.md"
SCRIPT = ROOT / "examples" / "make_task_data.py"

pytestmark = pytest.mark.skipif(not (TASKS.exists() and SCRIPT.exists()), reason="needs the repository checkout")

#: The questions of every task (section 6.3 of the design): T0.1-T0.4, T1.1-T1.7, ...
QUESTIONS = {0: 4, 1: 7, 2: 7, 3: 7, 4: 6, 5: 7, 6: 7, 7: 4}

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|<img[^>]*\ssrc=\"([^\"]+)\"")
FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)


def _markdown_files() -> list[Path]:
    files = [ROOT / "README.md", TASKS, ROOT / "CHANGELOG.md", ROOT / "NOTICE.md"]
    files += sorted((ROOT / "docs").glob("*.md"))
    files += sorted((ROOT / "examples").glob("*/README.md"))
    files += [ROOT / "tests" / "data" / "README.md"]
    return [f for f in files if f.exists()]


def _tasks_text() -> str:
    return TASKS.read_text(encoding="utf-8")


def _sections(text: str) -> dict[int, str]:
    """The text of every task ``## T<k> ...`` up to the next level-2 heading."""
    parts = re.split(r"^(## .*)$", text, flags=re.MULTILINE)
    out = {}
    for heading, body in zip(parts[1::2], parts[2::2], strict=True):
        match = re.match(r"## T(\d) ", heading)
        if match:
            out[int(match.group(1))] = body
    return out


def test_question_ids_unique_and_ordered():
    ids = re.findall(r"^\*\*T(\d)\.(\d+)\*\*", _tasks_text(), flags=re.MULTILINE)
    pairs = [(int(t), int(q)) for t, q in ids]
    assert len(set(pairs)) == len(pairs), "a question ID appears twice"
    assert pairs == sorted(pairs), "question IDs out of order"
    expected = [(t, q) for t, n in QUESTIONS.items() for q in range(1, n + 1)]
    assert pairs == expected


def test_every_task_has_goal_deliverables_hints():
    sections = _sections(_tasks_text())
    assert sorted(sections) == list(QUESTIONS)
    for task, body in sections.items():
        for marker in ("**Goal:**", "**Deliverables:**", "**Hints**"):
            assert marker in body, f"T{task} has no {marker}"
        assert body.index("**Goal:**") < body.index("**Deliverables:**") < body.index("**Hints**"), f"T{task}"
        assert re.search(rf"^\*\*T{task}\.1\*\*", body, flags=re.MULTILINE), f"T{task}"


def _produced() -> set[str]:
    """Every file the two data blocks make (generate, merge, split, fetch jetclass)."""
    spec = importlib.util.spec_from_file_location("make_task_data", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    produced: set[str] = set()
    for command in module.task_commands(TASKS):
        if command[0] != "fetch":
            produced.update(p.as_posix() for p in module.output_of(command))
        elif command[1] == "jetclass":
            split = command[command.index("--split") + 1] if "--split" in command else "val"
            if split == "example":
                produced.add(f"data/jetclass/example/{MANIFEST['example', 'all'].filename}")
                continue
            classes = command[command.index("--classes") + 1] if "--classes" in command else "QCD,Wqq,Tbqq"
            for short in class_names(classes.split(",")):
                produced.add(f"data/jetclass/{split}/{MANIFEST[split, short].filename}")
    return produced


def test_every_data_file_is_produced_by_a_block():
    produced = _produced()
    directories = {parent.as_posix() for p in produced for parent in Path(p).parents}
    text = _tasks_text()
    missing = []
    for name in sorted(set(re.findall(r"(?<![\w/.])(?:\.\./)?(data/[\w./*-]*\w)", text))):
        name = name.removeprefix("../")
        if Path(name).name.startswith("toy_"):  # T0: the default names of jetgen generate
            continue
        if name in produced or name.rstrip("/") in directories:
            continue
        if any(fnmatch.fnmatch(p, name) for p in produced):
            continue
        missing.append(name)
    assert not missing, f"used in TASKS.md but made by no data block: {missing}"


def _slug(heading: str) -> str:
    """The anchor GitHub gives a heading."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading.strip())
    text = re.sub(r"<[^>]+>", "", text).lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6} +(.+?) *#*$", text, flags=re.MULTILINE):
        slug = _slug(heading)
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    anchors.update(re.findall(r"<a +(?:id|name)=\"([^\"]+)\"", text))
    return anchors


def test_slug():
    assert _slug("T1 Jets and their observables (about 4 hours)") == "t1-jets-and-their-observables-about-4-hours"
    assert _slug("`jetgen fetch`: downloads") == "jetgen-fetch-downloads"


def test_relative_links_and_anchors_resolve():
    broken = []
    for path in _markdown_files():
        text = FENCE.sub("", path.read_text(encoding="utf-8"))
        for match in LINK.finditer(text):
            target = match.group(1) or match.group(2)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto:, ...
                continue
            file_part, _, anchor = target.partition("#")
            resolved = (path.parent / file_part).resolve() if file_part else path
            where = f"{path.relative_to(ROOT)}: {target}"
            if not resolved.exists():
                broken.append(where)
            elif anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved):
                broken.append(where + " (anchor)")
    assert not broken, "broken links:\n" + "\n".join(broken)


def test_no_tbc_markers():
    left = [
        f"{path.relative_to(ROOT)}:{k}"
        for path in _markdown_files()
        for k, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "TBC" in line
    ]
    assert not left, f"numbers still to be confirmed: {left}"
