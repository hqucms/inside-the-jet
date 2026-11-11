"""Planted shortcuts for the bonus task T7.2 (``jetgen generate --mystery``).

A planted shortcut is one configuration field changed for one process only: the jets of that process then carry a
feature that a tagger can learn although it has nothing to do with the physics of the class. ``--mystery`` picks
one of :data:`SHORTCUTS` from the seed, generates the file with it as a per-process override and stores the choice
only in ``provenance["mystery"]``, scrambled (JSON XORed with a SHA-256 stream of the key, then base64): it is not
in the stored configuration, not in ``provenance["process_overrides"]``, not in ``jetgen info`` (``--provenance``
prints a placeholder) and cannot be read with a base64 decoder. :func:`reveal` unscrambles it
(``jetgen info --reveal``).

The scrambling is not encryption. Reading this module spoils T7.2: the mapping from the seed to the shortcut and
the default key are public. An instructor who wants a real hunt sets the environment variable ``ITJ_MYSTERY_KEY``
to a private integer before generating the students' files; both the mapping and the scrambling then depend on it,
and ``--reveal`` needs the same key (``docs/physics.md#planted-shortcuts``).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..data import JetData

__all__ = ["HIDDEN", "SHORTCUTS", "decode", "encode", "hide", "mystery_overrides", "reveal"]

#: The planted shortcuts: (name, process, configuration field, value). The strengths are calibrated so that a PFN
#: (``kin`` features, 10 epochs on 42 000 jets) gains 1.5-3 % of test accuracy from the shortcut (QCD/W/top accuracy
#: 0.921 without it); ``cell_grid`` is the exception: a per-particle model cannot see a grid in absolute (eta, phi),
#: so it changes the data but not the accuracy (``docs/physics.md#planted-shortcuts``).
SHORTCUTS: tuple[tuple[str, str, str, object], ...] = (
    ("cell_grid", "tbqq", "cell_position", "centre"),
    ("soft_tracks", "qcd", "track_eff_low", 0.0),
    ("busy_w", "wqq", "ue_density", 10.0),
    ("wide_top", "tbqq", "kt_sigma", 1.5),
    ("hard_w", "wqq", "pt_min", 650.0),
)

#: Environment variable with the instructor's key (an integer; default 0, the public mapping).
KEY_VARIABLE = "ITJ_MYSTERY_KEY"

#: What ``jetgen info --provenance`` prints in place of the scrambled entry.
HIDDEN = "<hidden: jetgen info --reveal>"

#: Prefix of a scrambled entry (files of itj 0.1.0 hold plain base64 of JSON, which :func:`decode` still reads).
_SCRAMBLED = "x1:"


def _key() -> int:
    raw = os.environ.get(KEY_VARIABLE, "0").strip() or "0"
    try:
        key = int(raw)
    except ValueError:
        raise ValueError(f"{KEY_VARIABLE} must be a non-negative integer, got {raw!r}") from None
    if key < 0:
        raise ValueError(f"{KEY_VARIABLE} must be a non-negative integer, got {key}")
    return key


def _index(seed: int) -> int:
    return int(np.random.SeedSequence([int(seed), _key()]).generate_state(1)[0]) % len(SHORTCUTS)


def mystery_overrides(seed: int) -> dict[str, dict[str, object]]:
    """Return the per-process override of the shortcut planted by ``jetgen generate --mystery --seed seed``.

    The shortcut is ``SHORTCUTS[SeedSequence([seed, key]).generate_state(1)[0] % 5]`` with ``key`` the integer in
    the environment variable ``ITJ_MYSTERY_KEY`` (default 0).

    Parameters
    ----------
    seed : int
        The run seed.

    Returns
    -------
    dict of str to dict
        ``{process: {field: value}}``, the ``process_overrides`` of :func:`itj.toy.generate`.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    _, process, name, value = SHORTCUTS[_index(int(seed))]
    return {process: {name: value}}


def _scramble(raw: bytes, key: int) -> bytes:
    """XOR ``raw`` with a SHA-256 counter stream of ``key`` (its own inverse)."""
    stream = b"".join(
        hashlib.sha256(f"itj-mystery:{key}:{block}".encode()).digest() for block in range(len(raw) // 32 + 1)
    )
    return bytes(a ^ b for a, b in zip(raw, stream[: len(raw)], strict=True))


def encode(seed: int) -> str:
    """Return the ``provenance["mystery"]`` entry of a mystery file of this seed (scrambled JSON, see the module)."""
    name, process, field, value = SHORTCUTS[_index(int(seed))]
    payload = {"name": name, "process": process, "field": field, "value": value, "key_set": _key() != 0}
    raw = _scramble(json.dumps(payload, sort_keys=True).encode(), _key())
    return _SCRAMBLED + base64.b64encode(raw).decode("ascii")


def decode(text: str) -> dict[str, Any]:
    """Decode a ``provenance["mystery"]`` entry (inverse of :func:`encode`, with the key of ``ITJ_MYSTERY_KEY``)."""
    scrambled = text.startswith(_SCRAMBLED)
    body = text[len(_SCRAMBLED) :] if scrambled else text
    try:
        raw = base64.b64decode(body.encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"cannot decode the mystery entry {text[:20]!r}...: {error}") from None
    try:
        payload = json.loads((_scramble(raw, _key()) if scrambled else raw).decode())
    except (ValueError, UnicodeError):
        if scrambled:
            raise ValueError(
                f"cannot decode the mystery entry: the file was made with another {KEY_VARIABLE}; set the key it "
                "was made with (ask your instructor)"
            ) from None
        raise ValueError(f"cannot decode the mystery entry {text[:20]!r}...: not JSON") from None
    if not isinstance(payload, dict) or not {"name", "process", "field", "value"} <= set(payload):
        raise ValueError("the mystery entry is not a planted shortcut")
    return payload


def hide(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a provenance with every mystery entry (also of merged inputs) replaced by :data:`HIDDEN`."""
    out = dict(provenance)
    if "mystery" in out:
        out["mystery"] = HIDDEN
    if isinstance(out.get("merged"), list):
        out["merged"] = [
            {**item, "provenance": hide(item["provenance"])}
            if isinstance(item, Mapping) and isinstance(item.get("provenance"), Mapping)
            else item
            for item in out["merged"]
        ]
    return out


def _entries(provenance: Mapping[str, Any] | None) -> list[str]:
    """Every mystery entry of a provenance, including those of merged inputs."""
    if not provenance:
        return []
    out = [provenance["mystery"]] if isinstance(provenance.get("mystery"), str) else []
    for item in provenance.get("merged") or []:
        if isinstance(item, Mapping):
            out += _entries(item.get("provenance"))
    return out


def reveal(path_or_data: str | os.PathLike[str] | JetData) -> str:
    """Describe the planted shortcut of a file or data set (``jetgen info --reveal``).

    Parameters
    ----------
    path_or_data : str, os.PathLike or JetData
        A file written by ``jetgen generate --mystery`` (or anything else) or its data.

    Returns
    -------
    str
        One line per planted shortcut, or ``"this file has no planted shortcut"``.
    """
    if isinstance(path_or_data, JetData):
        provenance = path_or_data.provenance
    else:
        from ..io import read_metadata

        provenance = read_metadata(Path(path_or_data))["provenance"]
    entries = _entries(provenance)
    if not entries:
        return "this file has no planted shortcut"
    from .config import SimConfig

    defaults = {key: value for part in SimConfig().to_dict().values() for key, value in part.items()}
    lines = []
    for text in entries:
        p = decode(text)
        default = defaults.get(p["field"], "?")
        lines.append(
            f"planted shortcut {p['name']!r}: the {p['process']} jets were generated with {p['field']} = "
            f"{p['value']!r} (default {default!r}); every other process is unchanged jet by jet"
        )
    return "\n".join(lines)
