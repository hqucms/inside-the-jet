"""Configuration: presets, overrides of every parameter, validation, JSON round trip, descriptions."""

import dataclasses
import json

import numpy as np
import pytest

from itj.toy import PRESETS, SimConfig, config_fields, describe_presets, matching_preset, preset
from itj.toy.config import DetectorPart, HardPart, ShowerPart

_SPECIAL = {
    "qcd_flavour_weights": (0.2, 0.2, 0.2, 0.2, 0.2),
    "jet_pt_min": 400.0,
    "jet_pt_max": 1200.0,
}


def _changed(f: dataclasses.Field):
    """A valid value different from the default of a configuration field."""
    meta, default = f.metadata, f.default
    if f.name in _SPECIAL:
        return _SPECIAL[f.name]
    if "choices" in meta:
        return next(c for c in meta["choices"] if c != default)
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default - 1
    return default * 0.9 if default else 0.5


_FIELDS = list(config_fields())


@pytest.mark.parametrize(("part", "name", "f"), _FIELDS, ids=[name for _, name, _ in _FIELDS])
def test_every_parameter_can_be_overridden(part, name, f):
    value = _changed(f)
    config = SimConfig().replace(**{name: value})
    got = getattr(getattr(config, part), name)
    assert got == value
    assert got != f.default
    for other_part, other, _ in config_fields():
        if other != name:
            assert getattr(getattr(config, other_part), other) == getattr(getattr(SimConfig(), other_part), other)
    assert SimConfig.from_json(config.to_json()) == config
    assert SimConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert f.metadata["help"]
    assert preset("detector", **{name: value}) == config


def test_parameter_names_are_unique_and_documented():
    names = [name for _, name, _ in config_fields()]
    assert len(names) == len(set(names))
    parts = {part for part, _, _ in config_fields()}
    assert parts == set(SimConfig.SECTIONS)
    for _, name, f in config_fields():
        assert f.metadata["help"].strip(), name
        assert "unit" in f.metadata
        assert "—" not in f.metadata["help"]  # no em dashes


def test_ladder():
    assert list(PRESETS) == ["partons", "shower", "hadrons", "detector"]
    assert SimConfig() == PRESETS["detector"] == preset()
    switches = [
        (c.shower.shower, c.hadron.hadronize, c.hadron.b_hadrons, c.ue.ue, c.detector.detector)
        for c in PRESETS.values()
    ]
    assert switches == [
        (False, False, False, False, False),
        (True, False, False, False, False),
        (True, True, True, True, False),
        (True, True, True, True, True),
    ]
    # the presets differ only in the ladder switches
    flat = [{k: v for part in c.to_dict().values() for k, v in part.items()} for c in PRESETS.values()]
    differing = {k for d in flat for k in d if d[k] != flat[-1][k]}
    assert differing == {"shower", "hadronize", "b_hadrons", "ue", "detector"}
    for name, config in PRESETS.items():
        assert matching_preset(config) == name
    assert matching_preset(preset("shower", kt_sigma=0.3)) is None


def test_describe():
    text = describe_presets()
    for name in PRESETS:
        assert name in text
    assert "default configuration" in text
    assert "B hadrons: off" in preset("partons").describe()
    assert "UE:        off" in preset("detector", ue=False).describe()
    assert "fixed 0.12" in preset("shower", alphas_mode="fixed", alphas_value=0.12).describe()


def test_replace_with_a_part_and_original_unchanged():
    config = SimConfig()
    other = config.replace(detector=DetectorPart(cell_position="centre"), ue=False)
    assert other.detector.cell_position == "centre"
    assert other.ue.ue is False
    assert config.ue.ue is True


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"cell_position": "center"}, ValueError, "Did you mean 'centre'"),
        ({"pi0_fractoin": 0.3}, TypeError, "Did you mean 'pi0_fraction'"),
        ({"pt_power": float("nan")}, ValueError, "finite"),
        ({"pt_power": "4"}, TypeError, "number"),
        ({"ue": 1}, TypeError, "True or False"),
        ({"n_flavours": 2.5}, TypeError, "integer"),
        ({"pt_min": 2000.0}, ValueError, "pt_min < pt_max"),
        ({"pi0_fraction": 0.95}, ValueError, "pi0_fraction \\+ neutral_fraction"),
        ({"kt_cut": 0.1}, ValueError, "kt_cut"),
        ({"qcd_flavour_weights": (1.0, 2.0)}, ValueError, "sequence of 5"),
        ({"jet_pt_min": 900.0, "jet_pt_max": 800.0}, ValueError, "jet_pt_min"),
        ({"top_mass": 100.0}, ValueError, "top mass window"),
        ({"w_helicity": None}, TypeError, "must not be None"),
    ],
)
def test_invalid_values(overrides, error, match):
    with pytest.raises(error, match=match):
        SimConfig().replace(**overrides)


def test_unknown_preset_suggests():
    with pytest.raises(ValueError, match="Did you mean 'shower'"):
        preset("showr")


def test_unknown_section_or_parameter_in_dict():
    with pytest.raises(ValueError, match="unknown configuration sections"):
        SimConfig.from_dict({"hadr": {}})
    with pytest.raises(ValueError, match="Did you mean 'w_mass'"):
        SimConfig.from_dict({"hard": {"w_mas": 80.0}})
    assert SimConfig.from_dict({}) == SimConfig()


def test_numpy_values_are_coerced():
    config = SimConfig().replace(
        w_mass=np.float64(81.0),
        n_flavours=np.int64(4),
        ue=np.bool_(False),
        qcd_flavour_weights=np.array([0.5, 0.3, 0.1, 0.05, 0.05]),
        b_decay_multiplicity=6.0,
    )
    assert type(config.hard.w_mass) is float
    assert type(config.shower.n_flavours) is int
    assert type(config.ue.ue) is bool
    assert type(config.hadron.b_decay_multiplicity) is int
    assert config.hard.qcd_flavour_weights == (0.5, 0.3, 0.1, 0.05, 0.05)
    assert all(type(w) is float for w in config.hard.qcd_flavour_weights)
    json.dumps(config.to_dict())


def test_parts_are_frozen_and_hashable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        HardPart().w_mass = 70.0  # type: ignore[misc]
    assert hash(ShowerPart()) == hash(ShowerPart())
    assert len({SimConfig(), SimConfig(), preset("shower")}) == 2


def test_running_coupling_must_stay_below_one_at_the_cutoff() -> None:
    assert preset("shower", alphas_value=0.15).shower.alphas_value == 0.15
    with pytest.raises(ValueError, match=r"alphas_value <= 0\.154"):
        preset("shower", alphas_value=0.2)
    with pytest.raises(ValueError, match="Lambda"):
        preset("shower", alphas_value=0.16, kt_cut=1.0)
    assert preset("shower", alphas_value=0.2, kt_cut=4.0).shower.kt_cut == 4.0
    assert preset("shower", alphas_mode="fixed", alphas_value=0.2).shower.alphas_value == 0.2
