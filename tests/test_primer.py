"""docs/primer.md and docs/ml_primer.md: the section anchors that the week files link to, and the Python blocks run."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIMER = ROOT / "docs" / "primer.md"
ML_PRIMER = ROOT / "docs" / "ml_primer.md"

#: The sections of "Jets from zero", in order; each is read just before one question of W1.1.
PRIMER_ANCHORS = [
    "units-and-energies",
    "four-vectors-and-invariant-mass",
    "collider-coordinates",
    "the-particles",
    "what-a-jet-is",
    "boosted-w-and-top-jets",
    "the-parton-shower",
    "hadronisation-and-the-underlying-event",
    "what-a-detector-sees",
    "the-toy-and-its-files",
    "histograms-and-errors",
    "a-first-look-at-the-observables",
    "where-to-read-more",
    "check-yourself-answers",
]
#: The sections of "Machine learning from zero", in order.
ML_PRIMER_ANCHORS = [
    "a-tagger-is-a-function",
    "train-validation-and-test",
    "decision-trees-and-boosting",
    "neural-networks",
    "pytorch-in-one-page",
    "networks-on-particles",
    "embeddings-and-maps",
    "regression-and-probes",
    "learning-without-labels",
    "domain-shift",
    "gradients-of-the-input",
]
FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)


def _anchors_in_order(path: Path) -> list[str]:
    """The GitHub anchors of the headings of ``path``, in order (as tests/test_tasks_md.py computes them)."""
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    out = []
    for heading in re.findall(r"^#{1,6} +(.+?) *#*$", text, flags=re.MULTILINE):
        slug = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading.strip())
        slug = re.sub(r"<[^>]+>", "", slug).lower()
        out.append(re.sub(r"[^\w\- ]", "", slug).replace(" ", "-"))
    out += re.findall(r"<a +(?:id|name)=\"([^\"]+)\"", text)
    return out


@pytest.mark.parametrize(
    ("path", "anchors"), [(PRIMER, PRIMER_ANCHORS), (ML_PRIMER, ML_PRIMER_ANCHORS)], ids=["primer", "ml_primer"]
)
def test_sections_and_their_order(path, anchors):
    assert path.exists(), f"{path.relative_to(ROOT)} is missing"
    found = _anchors_in_order(path)
    missing = [a for a in anchors if a not in found]
    assert not missing, f"{path.name}: sections missing (the week files link to them): {missing}"
    positions = [found.index(a) for a in anchors]
    assert positions == sorted(positions), f"{path.name}: the sections are out of order"


@pytest.mark.parametrize("path", [PRIMER, ML_PRIMER], ids=["primer", "ml_primer"])
def test_python_blocks_run(path, tmp_path, monkeypatch):
    """Every ```python block of the primer runs, in order and in one namespace (as a reader would type them)."""
    if not path.exists():
        pytest.skip(f"no {path.name}")
    blocks = re.findall(r"^```python\n(.*?)^```", path.read_text(encoding="utf-8"), flags=re.MULTILINE | re.DOTALL)
    if not blocks:
        pytest.skip(f"{path.name} has no Python blocks")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace: dict = {"__name__": "__primer__"}
    for k, block in enumerate(blocks, 1):
        try:
            exec(compile(block, f"{path.name} block {k}", "exec"), namespace)
        except Exception as error:  # pragma: no cover - the message names the block
            pytest.fail(f"{path.name}, Python block {k} fails: {error!r}\n{block}")


#: The opening angle of a boosted decay, which `W1.1.3` derives: no page that the primer sends a reader to before that
#: question may state it (``2M/p_T``, ``2m/p_T``, ``2 m_W / p_T``).
OPENING_ANGLE = re.compile(r"2\s*[mM](?:_\{?\w+\}?)?\s*/\s*p_\{?T")


def _section(path: Path, anchor: str) -> str:
    """The text of the section of ``path`` whose heading has ``anchor``, up to the next heading of the same level."""
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    headings = list(re.finditer(r"^(#{1,6}) +(.+?) *#*$", text, flags=re.MULTILINE))
    for k, match in enumerate(headings):
        slug = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", match.group(2).strip())
        slug = re.sub(r"[^\w\- ]", "", re.sub(r"<[^>]+>", "", slug).lower()).replace(" ", "-")
        if slug == anchor:
            level = len(match.group(1))
            end = next((h.start() for h in headings[k + 1 :] if len(h.group(1)) <= level), len(text))
            return text[match.start() : end]
    raise AssertionError(f"{path.name} has no section #{anchor}")


def test_no_opening_angle_before_w113():
    """Sections 1-6 of the primer, and the doc sections they link to, leave the result of `W1.1.3` to the student."""
    text = PRIMER.read_text(encoding="utf-8")
    early = text[: text.index("## The parton shower")]
    assert not OPENING_ANGLE.search(early), "the primer states the opening angle before W1.1.3"
    for target, anchor in set(re.findall(r"\]\(((?!http)[\w./]+\.md)#([\w-]+)\)", early)):
        section = _section(PRIMER.parent / target, anchor)
        assert not OPENING_ANGLE.search(section), f"{target}#{anchor} states the opening angle that W1.1.3 derives"
