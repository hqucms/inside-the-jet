"""A Spray of Hadrons: the jet toy of the course.

QCD, W -> q qbar' and top -> b q qbar' jets with an angular-ordered parton shower (Sudakov veto algorithm, colour
coherence, exact four-momentum of the resonances), independent fragmentation with B hadrons and pi0 decays, an
underlying event and a Delphes-like detector (tracks with impact parameters, ECAL and HCAL cells, particle flow),
written in the JetClass format plus a truth record (prongs, partons, every shower branching, the primary Lund
plane). A ladder of presets switches the physics on step by step (``partons``, ``shower``, ``hadrons``,
``detector``); the same seed gives the same events at every preset (``docs/physics.md``).

Quick start::

    from itj.toy import generate, preset, save, load, check_identities

    data = generate(30_000, seed=11)                            # qcd, wqq, tbqq at the detector level
    data["part_px"], data["jet_sdmass"], data.process_names
    save(data, "data/toy_detector_qcd-wqq-tbqq_n30000_seed11.parquet")

    partons = generate(30_000, config=preset("partons"), seed=11)  # the same hard events
    qg = generate(20_000, config=preset("shower", alphas_mode="fixed", alphas_value=0.12), processes=("q", "g"))
    check_identities(data)                                      # every exact identity: True

The same is available on the command line as ``jetgen generate`` (``jetgen --help``).

Exported names (``__all__``, sorted):

* generation: :func:`generate`, :func:`show_ladder`, :func:`shower_one`, :data:`CHUNK_SIZE`, :data:`PROCESSES`,
  :data:`TOY_NAME`;
* configuration: :class:`SimConfig`, :func:`preset`, :data:`PRESETS`, :func:`describe_presets`,
  :func:`matching_preset`, :func:`config_fields`;
* data: :class:`JetData`, :func:`save`, :func:`load` (from :mod:`itj.data` and :mod:`itj.io`);
* validation: :func:`check_identities`, :func:`validate`, :class:`ValidationRow`, :func:`format_validation`,
  :func:`sudakov_cdf`, :func:`lund_ll`.

The planted shortcuts of the bonus question B.3 (:mod:`itj.toy.mystery`: ``mystery_overrides``, ``SHORTCUTS``,
``reveal``) are deliberately not re-exported: tab completion on ``itj.toy`` should not hint at them, and reading
that module spoils the task.
"""

from __future__ import annotations

from ..data import JetData
from ..io import load, save
from .config import PRESETS, SimConfig, config_fields, describe_presets, matching_preset, preset
from .generate import CHUNK_SIZE, generate, show_ladder
from .hardprocess import PROCESSES
from .shower import shower_one
from .validate import ValidationRow, check_identities, format_validation, lund_ll, sudakov_cdf, validate

#: The name of the toy (docs titles, ``jetgen --help``).
TOY_NAME = "A Spray of Hadrons"

__all__ = [
    "CHUNK_SIZE",
    "PRESETS",
    "PROCESSES",
    "TOY_NAME",
    "JetData",
    "SimConfig",
    "ValidationRow",
    "check_identities",
    "config_fields",
    "describe_presets",
    "format_validation",
    "generate",
    "load",
    "lund_ll",
    "matching_preset",
    "preset",
    "save",
    "show_ladder",
    "shower_one",
    "sudakov_cdf",
    "validate",
]
