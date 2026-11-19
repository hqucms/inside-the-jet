"""docs/id_mapping_0.2.md: every question ID of 0.1.x maps to IDs of TASKS.md, and every ID of TASKS.md is covered.

The mapping is what the reference solutions, the grading notes, the course fixes and the result keys are remapped
with, so it must stay complete when TASKS.md changes (a renumbering or a move across a tier is recorded there).
"""

import re
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "TASKS.md"
MAPPING = ROOT / "docs" / "id_mapping_0.2.md"

pytestmark = pytest.mark.skipif(not (TASKS.exists() and MAPPING.exists()), reason="needs the repository checkout")

#: The questions of every task of 0.1.x: T0.1-T0.4, T1.1-T1.7, ..., T7.1-T7.4 (49 IDs).
QUESTIONS_0_1 = {0: 4, 1: 7, 2: 7, 3: 7, 4: 6, 5: 7, 6: 7, 7: 4}
OLD_IDS = [f"T{t}.{q}" for t, n in QUESTIONS_0_1.items() for q in range(1, n + 1)]

NEW_ID = r"(?:W[1-4]\.\d\.\d{1,2}|B\.[1-9])"
TIER_OF = {"Warm-up": "warm-up", "Core": "core", "Stretch": "stretch", "Bonus": "bonus"}


@cache
def _tasks() -> dict[str, str]:
    """Every question ID of TASKS.md with its tier (warm-up, core, stretch or bonus), in order."""
    text = TASKS.read_text(encoding="utf-8")
    found = re.findall(rf"^\*\*({NEW_ID})\*\* \*(Warm-up|Core|Stretch|Bonus),", text, flags=re.MULTILINE)
    return {qid: TIER_OF[tier] for qid, tier in found}


def _table(heading: str) -> list[list[str]]:
    """The rows (cells without the outer pipes) of the first table after ``heading``, without its header."""
    lines = MAPPING.read_text(encoding="utf-8").splitlines()
    start = lines.index(heading)
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        elif rows:
            break
    assert len(rows) > 2, f"no table under {heading!r}"
    assert set(rows[1][0]) <= set("-: "), "the second row of a table is its separator"
    return rows[2:]


def _without_brackets(cell: str) -> str:
    """The cell without its remarks in parentheses ("(previewed in ...)", "(toy)")."""
    return re.sub(r"\([^)]*\)", "", cell)


def _expand(cell: str) -> list[str]:
    """The IDs of a cell, with ranges such as ```W1.0.3`-`W1.0.4``` or ```B.2`-`B.5``` expanded."""
    ids: list[str] = []
    for first, last in re.findall(rf"`({NEW_ID})`(?:-`({NEW_ID})`)?", cell):
        if not last:
            ids.append(first)
            continue
        order = list(_tasks())
        assert first in order, first
        assert last in order, last
        ids += order[order.index(first) : order.index(last) + 1]
    return ids


@cache
def _forward() -> list[tuple[str, str, list[str], list[str], str]]:
    """(0.1.x ID or "new", part, the 0.2.0 IDs, the tiers, the raw 0.2.0 cell) of every forward row."""
    rows = []
    for cells in _table("## From 0.1.x to 0.2.0"):
        assert len(cells) == 4, f"a row of the forward table needs four cells: {cells}"
        old, part, new, tiers = cells
        old = old.strip("`")
        rows.append((old, part, re.findall(rf"`({NEW_ID})`", _without_brackets(new)), _tier_list(tiers), new))
    return rows


def _tier_list(cell: str) -> list[str]:
    return [t.strip() for t in cell.split(";") if t.strip()]


def test_tasks_has_its_questions():
    assert len(_tasks()) == 127  # 122 questions in four weeks and 5 bonus questions


def test_every_old_id_maps():
    rows = _forward()
    olds = [old for old, *_ in rows if old != "new"]
    assert set(olds) <= set(OLD_IDS), f"not an ID of 0.1.x: {sorted(set(olds) - set(OLD_IDS))}"
    assert sorted(set(olds), key=OLD_IDS.index) == OLD_IDS, "an ID of 0.1.x has no row"
    assert olds == sorted(olds, key=OLD_IDS.index), "the forward table is ordered by the 0.1.x ID"
    for old in OLD_IDS:
        targets = [ids for o, _, ids, _, _ in rows if o == old]
        assert any(targets), f"{old} maps to no question of 0.2.0"


def test_every_mapped_id_exists_with_its_tier():
    tasks = _tasks()
    for old, part, ids, tiers, raw in _forward():
        every = re.findall(rf"`({NEW_ID})`", raw)  # also the IDs in remarks such as "(previewed in ...)"
        unknown = [i for i in every if i not in tasks]
        assert not unknown, f"{old} ({part}) maps to IDs that TASKS.md does not have: {unknown}"
        if old == "new" or not ids:
            continue
        if not tiers and "*Start of week* box" in raw:  # a part that became a command of a box, used from an ID on
            continue
        assert len(tiers) == len(ids), f"{old} ({part}): one tier per ID, {ids} against {tiers}"
        for qid, tier in zip(ids, tiers, strict=True):
            assert tier == tasks[qid], f"{old} ({part}): {qid} is {tasks[qid]} in TASKS.md, not {tier}"


def test_every_new_id_is_covered():
    """Every ID of TASKS.md appears in the forward table: as the target of an old ID or in the row "new"."""
    covered = {qid for *_, raw in _forward() for qid in re.findall(rf"`({NEW_ID})`", raw)}
    missing = [qid for qid in _tasks() if qid not in covered]
    assert not missing, f"IDs of TASKS.md that the mapping does not name: {missing}"
    assert any(old == "new" for old, *_ in _forward()), "the forward table has no row 'new'"


def test_reverse_table_agrees():
    """The reverse table covers every ID of TASKS.md once, in order, and names the same 0.1.x sources."""
    tasks = list(_tasks())
    reverse: dict[str, set[str]] = {}
    order: list[str] = []
    for cells in _table("## From 0.2.0 back to 0.1.x"):
        assert len(cells) == 2, f"a row of the reverse table needs two cells: {cells}"
        ids = _expand(cells[0])
        assert ids, f"no ID in {cells[0]!r}"
        sources = set(re.findall(r"`(T\d\.\d)`(?:-`(T\d\.\d)`)?", cells[1]))
        olds: set[str] = set()
        for first, last in sources:
            olds.update(OLD_IDS[OLD_IDS.index(first) : OLD_IDS.index(last) + 1] if last else [first])
        if "new" in cells[1]:
            olds.add("new")
        assert olds, f"no source in {cells}"
        ranged = bool(re.search(r"`-`", cells[0]))
        for k, qid in enumerate(ids):
            # a range of 0.2.0 IDs against a range of 0.1.x IDs pairs them one by one (B.2-B.5 from T7.1-T7.4)
            if ranged and len(sources) == 1 and next(iter(sources))[1] and len(olds) == len(ids):
                reverse[qid] = {sorted(olds, key=OLD_IDS.index)[k]}
            else:
                reverse[qid] = set(olds)
        order += ids
    assert order == tasks, "the reverse table must list every ID of TASKS.md once, in order"
    forward: dict[str, set[str]] = {qid: set() for qid in tasks}
    for old, _, ids, _, _ in _forward():
        for qid in ids:
            forward.setdefault(qid, set()).add(old)  # an unknown ID fails test_every_mapped_id_exists
    for qid in tasks:
        assert forward[qid] == reverse[qid], f"{qid}: forward {sorted(forward[qid])}, reverse {sorted(reverse[qid])}"
