"""Contrastive learning tools (``itj.ssl``): NT-Xent, alignment and uniformity, augmentations, linear evaluation."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from itj.data import JetData
from itj.ssl import (
    JetCLRAugment,
    ProjectionHead,
    alignment,
    linear_eval,
    mask_particles,
    nt_xent,
    two_views,
    uniformity,
)

# -------------------------------------------------------------------------------------------------------------- losses


def _nt_xent_direct(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> float:
    """NT-Xent written out with loops (SimCLR, eq. 1), in float64."""
    z = torch.cat([z1, z2]).double()
    z = z / z.norm(dim=1, keepdim=True)
    n = len(z)
    b = n // 2
    total = 0.0
    for i in range(n):
        positive = (i + b) % n
        num = math.exp(float(z[i] @ z[positive]) / temperature)
        den = sum(math.exp(float(z[i] @ z[k]) / temperature) for k in range(n) if k != i)
        total += -math.log(num / den)
    return total / n


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
def test_nt_xent_equals_the_direct_formula(temperature: float) -> None:
    g = torch.Generator().manual_seed(0)
    z1 = torch.randn(6, 5, generator=g)
    z2 = torch.randn(6, 5, generator=g)
    got = float(nt_xent(z1, z2, temperature=temperature))
    assert got == pytest.approx(_nt_xent_direct(z1, z2, temperature), rel=1e-5)


def test_nt_xent_limits_and_gradient() -> None:
    b = 16
    z = torch.randn(1, 8).expand(b, 8)
    # identical embeddings for all jets: every similarity is 1, the loss is ln(2B - 1)
    assert float(nt_xent(z, z)) == pytest.approx(math.log(2 * b - 1), rel=1e-6)
    # perfectly aligned and orthogonal to all other jets: the loss goes to 0 at small temperature
    eye = torch.eye(b)
    assert float(nt_xent(eye, eye, temperature=0.01)) < 1e-6
    # scale invariance (normalised)
    g = torch.Generator().manual_seed(1)
    z1, z2 = torch.randn(b, 4, generator=g), torch.randn(b, 4, generator=g)
    assert float(nt_xent(3 * z1, 0.5 * z2)) == pytest.approx(float(nt_xent(z1, z2)), rel=1e-5)
    z1.requires_grad_(True)
    nt_xent(z1, z2).backward()
    assert z1.grad is not None
    assert torch.isfinite(z1.grad).all()
    with pytest.raises(ValueError, match="z1 and z2"):
        nt_xent(z1, z2[:3])
    with pytest.raises(ValueError, match="temperature"):
        nt_xent(z1, z2, temperature=0.0)


def test_alignment_and_uniformity() -> None:
    g = torch.Generator().manual_seed(2)
    z = torch.randn(64, 3, generator=g)
    assert float(alignment(z, 2 * z)) == pytest.approx(0.0, abs=1e-6)
    assert float(alignment(z, -z)) == pytest.approx(4.0, rel=1e-5)  # antipodal: ||2u||^2
    collapsed = torch.ones(64, 3)
    assert float(uniformity(collapsed)) == pytest.approx(0.0, abs=1e-6)  # log E exp(0)
    assert float(uniformity(z)) < -1.0  # spread on the sphere
    head = ProjectionHead(10, d_hidden=16, d_out=4)
    assert head(torch.randn(5, 10)).shape == (5, 4)


# -------------------------------------------------------------------------------------------------------- augmentations


def test_augment_views_differ_and_keep_labels(jetclass_sample: JetData) -> None:
    augment = JetCLRAugment()
    v1, v2 = two_views(jetclass_sample, augment, 0)
    for view in (v1, v2):
        assert len(view) == len(jetclass_sample)
        assert np.array_equal(view.labels(), jetclass_sample.labels())
        assert not any(name.startswith("truth_") for name in view.names)
        # collinear splitting with frac 0.5 adds particles
        assert np.all(view.counts >= jetclass_sample.counts)
    assert not np.allclose(v1["part_deta"].values[:50], v2["part_deta"].values[:50])  # type: ignore[union-attr]
    # the scalar pT sum of every jet is kept by rotation, splitting and angular smearing (up to float32)

    def scalar_pt(data: JetData) -> np.ndarray:
        px, py = data["part_px"], data["part_py"]
        pt = np.hypot(px.values.astype(np.float64), py.values.astype(np.float64))  # type: ignore[union-attr]
        return np.bincount(px.row_index(), weights=pt, minlength=len(data))  # type: ignore[union-attr]

    np.testing.assert_allclose(scalar_pt(v1), scalar_pt(jetclass_sample), rtol=1e-5)
    # the fused smearing and rotation equal the separate itj.augment calls with the same random numbers
    from itj import augment as aug

    g = np.random.default_rng(5)
    fused = JetCLRAugment(collinear_frac=0.0)(jetclass_sample, np.random.default_rng(5))
    separate = aug.rotate(aug.smear_soft(jetclass_sample.without_truth(), scale=0.1, rng=g), rng=g)
    for name in ("part_px", "part_deta", "part_dphi"):
        np.testing.assert_allclose(fused[name].values, separate[name].values, atol=2e-3)  # type: ignore[union-attr]
    # reproducible with the same seed
    w1, _ = two_views(jetclass_sample, augment, 0)
    np.testing.assert_array_equal(w1["part_px"].values, v1["part_px"].values)  # type: ignore[union-attr]
    # the switches
    plain = JetCLRAugment(rotate=False, smear_soft=0.0, collinear_frac=0.0)
    same = plain(jetclass_sample, 0)
    np.testing.assert_array_equal(same.counts, jetclass_sample.counts)
    np.testing.assert_allclose(
        same["part_deta"].values,  # type: ignore[union-attr]
        jetclass_sample["part_deta"].values,  # type: ignore[union-attr]
        atol=1e-4,
    )
    more = JetCLRAugment(translate=0.1, drop_soft=1.0)(jetclass_sample, 1)
    assert np.all(more.counts >= 1)
    assert "JetCLRAugment(" in repr(augment)
    with pytest.raises(ValueError, match="collinear_frac"):
        JetCLRAugment(collinear_frac=1.5)
    with pytest.raises(ValueError, match="smear_soft"):
        JetCLRAugment(smear_soft=-1.0)


@pytest.mark.slow
def test_augment_is_fast(jetclass_sample: JetData) -> None:
    jets = jetclass_sample.select(np.arange(256) % len(jetclass_sample))
    augment = JetCLRAugment()
    rng = np.random.default_rng(0)
    augment(jets, rng)  # warm-up
    best = min(_timed(augment, jets, rng) for _ in range(10))  # best of 10: robust against a busy machine
    print(f"augmenting 256 jets: {1000 * best:.1f} ms (target 20 ms on a 2016 six-core CPU)")
    # a generous bound: timing is only a smoke test here (coverage tracing and busy CI runners slow it down)
    assert best < 0.100, f"augmenting 256 jets took {1000 * best:.1f} ms"


def _timed(augment: JetCLRAugment, jets: JetData, rng: np.random.Generator) -> float:
    start = time.perf_counter()
    augment(jets, rng)
    return time.perf_counter() - start


# ---------------------------------------------------------------------------------------------------- masked particles


def test_mask_particles(jetclass_sample: JetData) -> None:
    from itj.features import make_batch

    batch = make_batch(jetclass_sample.head(20), "kin", label_set=None)
    g = torch.Generator().manual_seed(0)
    masked, targets, which = mask_particles(batch, frac=0.3, generator=g)
    b, f, p = batch.features.shape
    assert targets.shape == (b, f, p)
    assert which.shape == (b, p)
    assert which.dtype == torch.bool
    real = batch.particle_mask
    assert not (which & ~real).any()
    n = real.sum(1)
    expected = torch.clamp(torch.round(0.3 * n.double()).long(), min=1)
    assert torch.equal(which.sum(1), expected)
    torch.testing.assert_close(targets, batch.features)
    assert torch.all(masked.features[:, :, :][which.unsqueeze(1).expand(-1, f, -1)] == 0)
    assert torch.equal(masked.mask, batch.mask)
    # unmasked real particles are untouched
    keep = (real & ~which).unsqueeze(1).expand(-1, f, -1)
    torch.testing.assert_close(masked.features[keep], batch.features[keep])
    # the wrap-padded copies of the masked particles are hidden too
    for j in range(b):
        nj = int(n[j])
        for slot in range(nj, p):
            if which[j, slot % nj]:
                assert torch.all(masked.vectors[j, :, slot] == 0)
    with pytest.raises(ValueError, match="frac"):
        mask_particles(batch, frac=0.0)


# ---------------------------------------------------------------------------------------------------- linear evaluation


def test_linear_eval(jetclass_sample: JetData) -> None:
    pytest.importorskip("sklearn")
    from itj.models import build_model

    torch.manual_seed(0)
    encoder = build_model("pfn")
    idx = np.arange(len(jetclass_sample))
    train, test = jetclass_sample.select(idx % 2 == 0), jetclass_sample.select(idx % 2 == 1)
    out = linear_eval(encoder, train, test)
    assert set(out) == {"accuracy", "n_labels", "auc_w", "auc_top"}
    assert out["n_labels"] == len(train)
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["auc_top"] > 0.5  # even a random PFN separates top from QCD a little
    few = linear_eval(encoder, train, test, n_labels=30, seed=1)
    assert few["n_labels"] == 30
    with pytest.raises(ValueError, match="n_labels"):
        linear_eval(encoder, train, test, n_labels=500)
