"""TASKS.md and the documents: question IDs, tiers, times, the week structure, data files, links and anchors.

The question IDs of release 0.2.0 are ``W<week>.<task>.<question>`` and ``B.1`` to ``B.5``; a question starts a line
as ``**W1.1.3** *Core, 90 min.* Title``. docs/id_mapping_0.2.md maps the IDs of 0.1.x (tests/test_question_ids.py).
"""

import fnmatch
import importlib.util
import re
import types
from functools import cache
from pathlib import Path

import pytest

from itj.jetclass import MANIFEST, class_names

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "TASKS.md"
SCRIPT = ROOT / "examples" / "make_task_data.py"
MAPPING = "docs/id_mapping_0.2.md"

pytestmark = pytest.mark.skipif(not (TASKS.exists() and SCRIPT.exists()), reason="needs the repository checkout")

#: The questions of every task: (warm-up, core, stretch).
QUESTIONS = {
    "W1.0": (1, 1, 2),
    "W1.1": (2, 4, 4),
    "W1.2": (1, 1, 9),
    "W1.3": (1, 1, 4),
    "W1.4": (1, 1, 5),
    "W2.1": (1, 1, 1),
    "W2.2": (1, 1, 2),
    "W2.3": (2, 1, 1),
    "W2.4": (1, 1, 2),
    "W2.5": (1, 1, 4),
    "W3.1": (1, 1, 4),
    "W3.2": (1, 1, 6),
    "W3.3": (1, 2, 5),
    "W3.4": (1, 3, 3),
    "W4.1": (1, 1, 5),
    "W4.2": (1, 2, 6),
    "W4.3": (1, 1, 2),
    "W4.4": (1, 2, 2),
    "W4.5": (1, 1, 7),
}
BONUS = 5
TIERS = ("Warm-up", "Core", "Stretch")
#: The hours of a week's core path outside its questions: the start-of-week box, the summary, the hand-in, the
#: discussion and (week 4) the final summary.
OVERHEAD = {1: 1.0, 2: 1.25, 3: 1.25, 4: 1.75}
#: The band of a week's core path in hours: the guard against creeping load (the pilot sets the final band).
WEEK_BAND = (12.5, 14.0)

QUESTION = re.compile(
    r"^\*\*W([1-4])\.(\d)\.(\d{1,2})\*\* \*(Warm-up|Core|Stretch), (\d+ min|\d+(?:\.\d+)? h)(?:\s\+\s[^*]+)?\.\*",
    re.MULTILINE,
)
BONUS_LINE = re.compile(r"^\*\*B\.([1-9])\*\* \*Bonus, (\d+ min|\d+(?:\.\d+)? h)\.\*", re.MULTILINE)
HOURS = r"\(core path about (\d+(?:\.\d+)?) h, stretch about (\d+(?:\.\d+)?) h\)"
WEEK_HEADING = re.compile(rf"^## Week ([1-4]): (.+) {HOURS}$", re.MULTILINE)
TASK_HEADING = re.compile(rf"^### (W[1-4]\.\d) (.+) {HOURS}$", re.MULTILINE)
#: An ID in the text of TASKS (bold) and in file names.
ID = re.compile(r"\b(W[1-4]\.\d\.\d{1,2}|B\.[1-9])\b")
#: An ID of release 0.1.x, which TASKS names only in the sentence that points to the mapping.
OLD_ID = re.compile(r"\bT[0-7]\.\d\b")
OLD_ID_SENTENCE = re.compile(
    r"The IDs of release 0\.1 \(T0\.1 to T7\.4\) are mapped to these in\s+\[" + re.escape(MAPPING) + r"\]"
)

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|<img[^>]*\ssrc=\"([^\"]+)\"")
FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)


def _markdown_files() -> list[Path]:
    files = [ROOT / "README.md", TASKS, ROOT / "CHANGELOG.md", ROOT / "NOTICE.md"]
    files += sorted((ROOT / "docs").glob("*.md"))
    files += sorted((ROOT / "examples").glob("*/README.md"))
    files += [ROOT / "tests" / "data" / "README.md"]
    return [f for f in files if f.exists()]


@cache
def _tasks_text() -> str:
    return TASKS.read_text(encoding="utf-8")


def _hours(time: str) -> float:
    """``"45 min"`` or ``"2.25 h"`` in hours."""
    number, unit = time.split()
    return float(number) / 60 if unit == "min" else float(number)


@cache
def _questions() -> list[tuple[str, str, str, float]]:
    """(task, ID, tier, hours) of every question of TASKS, in the order of the text."""
    out = []
    for match in QUESTION.finditer(_tasks_text()):
        week, task, number, tier, time = match.groups()
        out.append((f"W{week}.{task}", f"W{week}.{task}.{number}", tier, _hours(time)))
    return out


def _all_ids() -> set[str]:
    return {q[1] for q in _questions()} | {f"B.{k}" for k in range(1, BONUS + 1)}


def _sections(text: str, level: int) -> list[tuple[str, str]]:
    """(heading, body) of every heading of exactly ``level``; a body ends at the next heading of that level or above."""
    parts = re.split(rf"^(#{{1,{level}}} .*)$", FENCE.sub("", text), flags=re.MULTILINE)
    return [(h, b) for h, b in zip(parts[1::2], parts[2::2], strict=True) if h.startswith("#" * level + " ")]


def _weeks() -> dict[int, tuple[str, str]]:
    """The heading and body of every ``## Week k: ...``."""
    out = {}
    for heading, body in _sections(_tasks_text(), 2):
        match = re.match(r"## Week (\d):", heading)
        if match:
            out[int(match.group(1))] = (heading, body)
    return out


def _tasks() -> dict[str, tuple[str, str]]:
    """The heading and body of every task ``### Wk.t ...`` (up to the next heading of level 2 to 4)."""
    parts = re.split(r"^(#{2,4} .*)$", FENCE.sub("", _tasks_text()), flags=re.MULTILINE)
    out = {}
    for heading, body in zip(parts[1::2], parts[2::2], strict=True):
        match = re.match(r"### (W[1-4]\.\d|B) ", heading)
        if match:
            out[match.group(1)] = (heading, body)
    return out


# --- IDs, tiers and times -----------------------------------------------------------------------------------------


def test_every_question_line_is_well_formed():
    """Every line that starts with a bold ID has the tier and the time: a typo cannot hide a question."""
    starts = re.findall(r"^\*\*(?:W\d|B\.)[^*]*\*\*.*$", _tasks_text(), flags=re.MULTILINE)
    bad = [line for line in starts if not (QUESTION.match(line) or BONUS_LINE.match(line))]
    assert not bad, f"question lines without '*Tier, time.*': {bad}"


def test_question_ids_unique_and_ordered():
    questions = _questions()
    ids = [q[1] for q in questions]
    assert len(set(ids)) == len(ids), "a question ID appears twice"
    expected = [
        f"{task}.{n}" for task, counts in QUESTIONS.items() for n in range(1, sum(counts) + 1)
    ]  # numbered from 1 in every task, tasks in order
    assert ids == expected
    bonus = [int(n) for n, _ in BONUS_LINE.findall(_tasks_text())]
    assert bonus == list(range(1, BONUS + 1))


def test_tiers_ordered_and_complete():
    for task, counts in QUESTIONS.items():
        tiers = [q[2] for q in _questions() if q[0] == task]
        assert tiers == sorted(tiers, key=TIERS.index), f"{task}: tiers out of order {tiers}"
        assert tuple(tiers.count(t) for t in TIERS) == counts, f"{task}: (warm-up, core, stretch) changed"
        assert all(counts), f"{task} needs at least one question of every tier"


def test_task_and_week_hours():
    """The task headings sum their questions; the week headings sum their tasks plus the overhead, within the band."""
    tasks = {task: heading for task, (heading, _) in _tasks().items() if task != "B"}
    assert list(tasks) == list(QUESTIONS)
    core: dict[str, float] = {}
    stretch: dict[str, float] = {}
    for task, heading in tasks.items():
        match = TASK_HEADING.match(heading)
        assert match, f"task heading without '(core path about N h, stretch about M h)': {heading}"
        hours = [q for q in _questions() if q[0] == task]
        core[task] = sum(h for _, _, tier, h in hours if tier != "Stretch")
        stretch[task] = sum(h for _, _, tier, h in hours if tier == "Stretch")
        assert float(match.group(3)) == pytest.approx(core[task], abs=0.125), f"{task}: core path"
        assert float(match.group(4)) == pytest.approx(stretch[task], abs=0.125), f"{task}: stretch"
    headings = WEEK_HEADING.findall(_tasks_text())
    assert [int(w) for w, *_ in headings] == [1, 2, 3, 4]
    table = _tasks_text().split("\n## ", 1)[0]  # the hours table of the introduction
    for week, title, week_core, week_stretch in headings:
        k = int(week)
        mine = [t for t in QUESTIONS if t.startswith(f"W{k}.")]
        total = sum(core[t] for t in mine) + OVERHEAD[k]
        assert float(week_core) == pytest.approx(total, abs=0.125), f"week {k}: core path"
        assert WEEK_BAND[0] <= float(week_core) <= WEEK_BAND[1], f"week {k}: core path outside {WEEK_BAND} h"
        assert float(week_stretch) == pytest.approx(sum(stretch[t] for t in mine), abs=0.125), f"week {k}: stretch"
        row = rf"^\| {k} \| {re.escape(title)} \| about {week_core} h \| about {week_stretch} h \|$"
        assert re.search(row, table, flags=re.MULTILINE), f"the hours table of the introduction disagrees on week {k}"
    bonus = re.search(r"^## Bonus: .* \(about (\d+(?:\.\d+)?) h\)$", _tasks_text(), flags=re.MULTILINE)
    assert bonus, "no '## Bonus: ... (about N h)' heading"
    assert float(bonus.group(1)) == pytest.approx(sum(_hours(t) for _, t in BONUS_LINE.findall(_tasks_text())))


# --- structure ----------------------------------------------------------------------------------------------------


def test_every_week_has_its_parts_in_order():
    weeks = _weeks()
    assert sorted(weeks) == [1, 2, 3, 4]
    for k, (_, body) in weeks.items():
        summary = "#### Final summary" if k == 4 else "#### Weekly summary"
        parts = [
            "**Why this week.**",
            "**Thread.**",
            "**Concepts this week**",
            "**Start of week**",
            f"### Data for week {k}",
            f"### W{k}.",
            summary,
        ]
        where = []
        for part in parts:
            assert part in body, f"week {k} has no {part}"
            where.append(body.index(part))
        assert where == sorted(where), f"week {k}: its parts are out of order"
        for part in ("**If you are behind**", "**First stretch picks"):
            assert part in body, f"week {k} has no {part}"
            assert body.index(part) < body.index(f"### W{k}."), f"week {k}: {part} after the first task"
        tasks = re.findall(rf"^### (W{k}\.\d) ", body, flags=re.MULTILINE)
        assert tasks == [t for t in QUESTIONS if t.startswith(f"W{k}.")], f"week {k}: tasks"


def test_every_task_has_goal_questions_deliverables_hints():
    tasks = _tasks()
    assert list(tasks) == [*QUESTIONS, "B"]
    for task, (_, body) in tasks.items():
        for marker in ("**Goal:**", "**Deliverables:**", "**Hints**"):
            assert marker in body, f"{task} has no {marker}"
        first = re.search(r"^\*\*(W\d\.\d\.\d+|B\.\d)\*\*", body, flags=re.MULTILINE)
        assert first, f"{task} has no question"
        assert first.group(1) in (f"{task}.1", "B.1"), f"{task}: the first question is {first.group(1)}"
        last = max(m.start() for m in re.finditer(r"^\*\*(W\d\.\d\.\d+|B\.\d)\*\*", body, flags=re.MULTILINE))
        goal, deliverables, hints = (body.index(m) for m in ("**Goal:**", "**Deliverables:**", "**Hints**"))
        assert goal < first.start(), f"{task}: the Goal comes before the questions"
        assert last < deliverables < hints, f"{task}: the questions, then Deliverables, then Hints"


def test_figure_names_start_with_their_id():
    """Every figure is named ``<ID>_<name>.(pdf|png)``; in a task's Deliverables, by an ID of that task."""
    text = FENCE.sub("", _tasks_text())
    names = set(re.findall(r"`([^`\s]+\.(?:pdf|png))`", text))
    ids = _all_ids()
    for name in names:
        match = re.match(r"^(W[1-4]\.\d\.\d{1,2}|B\.\d)_[\w-]+\.(?:pdf|png)$", name)
        assert match, f"figure name without its ID: {name}"
        assert match.group(1) in ids, f"figure of an ID that does not exist: {name}"
    for task, (_, body) in _tasks().items():
        deliverables = body[body.index("**Deliverables:**") : body.index("**Hints**")]
        for name in re.findall(r"`([^`\s]+\.(?:pdf|png))`", deliverables):
            assert name.startswith(f"{task}." if task != "B" else "B."), f"{task} hands in {name}"


def test_no_old_ids_and_no_course_fix_codes():
    text = _tasks_text()
    assert len(OLD_ID_SENTENCE.findall(text)) == 1, f"the sentence that points to {MAPPING} is missing"
    assert not OLD_ID.findall(OLD_ID_SENTENCE.sub("", text)), "an ID of release 0.1 in TASKS.md"
    assert not re.findall(r"\bCF-\d+\b", text), "a course-fix code in TASKS.md"
    assert f"]({MAPPING})" in text


#: Phrases that give a result away; TASKS gives inputs, never answers, reference values or truth values.
ANSWER_PHRASES = re.compile(
    r"answer is|answer:|solution:|reference value|you should (?:find|get|see) (?:about|around|that)"
    r"|should (?:be|come out|give) (?:about|around|close to|near|roughly)|expected (?:value|answer|result)"
    r"|only Sudakov safe|changes under a soft particle|are IRC safe;",
    re.IGNORECASE,
)
METRIC_VALUE = re.compile(
    r"\b(?:AUC|R_\{?50\}?|R50|accuracy|JSD|selectivity|R\^2)\b[^.\n]{0,25}?"
    r"(?:=|\\approx|≈|\bof about\b|\bis about\b|\bnear\b|\bat about\b|\baround\b)\s*\$?\d"
)
#: Expected-behaviour sentences that check the set-up, not an answer (the rendering rule allows them).
SET_UP_CHECKS = ("after 30 steps expect an accuracy of about", "an\naccuracy near 1/3, chance for three classes")


def test_no_answers_in_tasks():
    text = _tasks_text()
    for allowed in SET_UP_CHECKS:
        text = text.replace(allowed, "")
    assert not ANSWER_PHRASES.findall(text), "TASKS.md gives a result away"
    found = [m.group(0) for m in METRIC_VALUE.finditer(text)]
    assert not found, f"a value of a metric in TASKS.md (answers stay in the reference solutions): {found}"


def test_python_snippets_import_what_they_use():
    """A snippet that calls ``itj.<module>.`` imports that module, in the snippet or an earlier one of its question."""
    lines = _tasks_text().splitlines()
    imported: set[str] = set()
    k, missing = 0, []
    while k < len(lines):
        line = lines[k]
        if re.match(r"^(#{1,4} |\*\*(W\d\.\d\.\d+|B\.\d)\*\*)", line):  # a new heading or question
            imported = set()
        if line.strip() == "```python":
            end = lines.index("```", k + 1)
            body = lines[k + 1 : end]
            for text in body:
                imported.update(re.findall(r"^\s*import (itj[\w.]*)", text))
                imported.update(re.findall(r"^\s*from (itj[\w.]*) import", text))
            for text in body:
                if re.match(r"^\s*(import|from) ", text):
                    continue
                for use in re.findall(r"\b(itj(?:\.\w+)+)", text.split("#")[0]):
                    module = _module_of(use)
                    if not any(m == module or m.startswith(module + ".") for m in imported):
                        missing.append(f"line {k + 1 + body.index(text) + 1}: {use} without import {module}")
            k = end
        k += 1
    assert not missing, "\n".join(missing)


def _module_of(dotted: str) -> str:
    """The module that ``itj.a.b.c`` needs: the longest prefix whose attributes are modules (``itj.toy.generate`` is
    the function that ``itj.toy`` exports, not the submodule of the same name)."""
    name, obj = "itj", importlib.import_module("itj")
    for part in dotted.split(".")[1:]:
        attr = getattr(obj, part, None)
        if attr is None:
            try:
                attr = importlib.import_module(f"{name}.{part}")
            except ImportError:
                return name
        if not isinstance(attr, types.ModuleType):
            return name
        name, obj = f"{name}.{part}", attr
    return name


# --- data ---------------------------------------------------------------------------------------------------------


@cache
def _script():
    spec = importlib.util.spec_from_file_location("make_task_data", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def _produced() -> dict[str, int]:
    """Every file the four data blocks make (generate, merge, split, fetch jetclass), with the week of its block."""
    module = _script()
    produced: dict[str, int] = {}
    for week in (4, 3, 2, 1):  # the earliest block wins
        for command in module.task_commands(TASKS, str(week)):
            if command[0] != "fetch":
                produced.update((p.as_posix(), week) for p in module.output_of(command))
            elif command[1] == "jetclass":
                split = command[command.index("--split") + 1] if "--split" in command else "val"
                if split == "example":
                    produced[f"data/jetclass/example/{MANIFEST['example', 'all'].filename}"] = week
                    continue
                classes = command[command.index("--classes") + 1] if "--classes" in command else "QCD,Wqq,Tbqq"
                for short in class_names(classes.split(",")):
                    produced[f"data/jetclass/{split}/{MANIFEST[split, short].filename}"] = week
    return produced


def _week_of(name: str) -> int | None:
    """The week whose block makes ``name`` (a file, a folder or a pattern); None if no block makes it."""
    produced = _produced()
    if name in produced:
        return produced[name]
    matches = [w for p, w in produced.items() if fnmatch.fnmatch(p, name) or p.startswith(name.rstrip("/") + "/")]
    return min(matches) if matches else None


def _data_paths(text: str) -> list[str]:
    return [
        name.removeprefix("../")
        for name in re.findall(r"(?<![\w/.])(?:\.\./)?(data/[\w./*-]*\w)", text)
        if not Path(name).name.startswith("toy_")  # W1.0: the default names of jetgen generate
    ]


def test_every_data_file_is_produced_by_a_block():
    missing = sorted({name for name in _data_paths(_tasks_text()) if _week_of(name) is None})
    assert not missing, f"used in TASKS.md but made by no data block: {missing}"


def test_no_week_needs_the_data_of_a_later_week():
    """A path used in week k is made by the block of week k or an earlier one; the bonus counts as week 4."""
    late = []
    sections = {k: body for k, (_, body) in _weeks().items()}
    bonus = _tasks_text()[_tasks_text().index("## Bonus:") :]
    for week, body in [*sections.items(), (4, bonus)]:
        for name in _data_paths(body):
            made = _week_of(name)
            if made is not None and made > week:
                late.append(f"week {week} uses {name} (made in week {made})")
    assert not late, "\n".join(sorted(set(late)))


# --- links and markers --------------------------------------------------------------------------------------------


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
    assert _slug("Week 2: Taggers, from cuts to transformers (core path about 12.75 h, stretch about 11.5 h)") == (
        "week-2-taggers-from-cuts-to-transformers-core-path-about-1275-h-stretch-about-115-h"
    )


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
