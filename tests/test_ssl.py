"""Self-supervision tools (``itj.ssl``): NT-Xent, augmentations, masked particles, the JEPA, linear evaluation."""

from __future__ import annotations

import dataclasses
import math
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from itj.data import JetData
from itj.ssl import (
    EMATarget,
    JEPAPredictor,
    JetCLRAugment,
    MaskedParticleDecoder,
    ProjectionHead,
    alignment,
    context_view,
    effective_rank,
    ema_momentum,
    jepa_loss,
    linear_eval,
    mask_particles,
    nt_xent,
    subjet_labels,
    subjet_target_mask,
    two_views,
    uniformity,
)

if TYPE_CHECKING:
    from itj.features import JetBatch
    from itj.models.base import Trace

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


def test_mask_particles_drop(jetclass_sample: JetData) -> None:
    from itj.features import make_batch

    batch = make_batch(jetclass_sample.head(20), "kin", label_set=None)
    real = batch.particle_mask
    masked, targets, which = mask_particles(batch, frac=0.3, generator=torch.Generator().manual_seed(1), drop=True)
    assert torch.equal(masked.particle_mask, real & ~which)
    assert (masked.particle_mask.sum(1) >= 1).all()
    torch.testing.assert_close(targets, batch.features)
    # the same particles as without drop (same generator state)
    _, _, same = mask_particles(batch, frac=0.3, generator=torch.Generator().manual_seed(1))
    assert torch.equal(same, which)
    # every particle of a jet would be masked with frac=1: one stays visible
    _, _, all_but_one = mask_particles(batch, frac=1.0, drop=True)
    assert torch.equal(all_but_one.sum(1), real.sum(1) - 1)


@pytest.mark.parametrize(("num_layers", "drop"), [(0, False), (1, True)])
def test_masked_particle_decoder_knows_the_slot(jetclass_sample: JetData, num_layers: int, drop: bool) -> None:
    from itj.features import make_batch
    from itj.models import build_model

    torch.manual_seed(0)
    model = build_model("partsmall_nopair")
    batch = make_batch(jetclass_sample.head(16), "kin", label_set=None)
    mb, _, which = mask_particles(batch, frac=0.3, generator=torch.Generator().manual_seed(0), drop=drop)
    model.eval()
    h = model.trace(mb).layers["block2"]
    jet = int(torch.argmax(which.sum(1)))
    slots = torch.nonzero(which[jet]).squeeze(1)
    assert len(slots) > 1
    if not drop:  # without position the encoder gives every masked slot of a jet the same output
        assert (h[jet, slots] - h[jet, slots[0]]).abs().amax() < 1e-4
    dec = MaskedParticleDecoder(h.shape[-1], 3, num_layers=num_layers)
    assert (dec.layers is None) == (num_layers == 0)
    pred = dec(h, batch.particle_mask, which)
    assert pred.shape == (*which.shape, 3)
    assert (pred[jet, slots] - pred[jet, slots[0]]).abs().amax() > 1e-4  # the decoder tells them apart
    # gradients reach the encoder
    pred[which].square().mean().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())
    # a decoder alone learns a target that depends only on the slot, which a head without the slot cannot
    torch.manual_seed(1)
    dec = MaskedParticleDecoder(8, 1, dim=16, num_heads=2, num_layers=num_layers)
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    real = torch.ones(32, 12, dtype=torch.bool)
    for _ in range(150):
        which = torch.rand(32, 12) < 0.3
        h = torch.where(which[..., None], torch.zeros(32, 12, 8), torch.randn(32, 12, 8))
        target = torch.linspace(-1, 1, 12).expand(32, 12)
        loss = ((dec(h, real, which)[..., 0] - target)[which] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.05 * float(target.var())
    with pytest.raises(ValueError, match="max_len"):
        dec(torch.zeros(1, 200, 8), torch.ones(1, 200, dtype=torch.bool), torch.zeros(1, 200, dtype=torch.bool))
    with pytest.raises(ValueError, match="num_layers"):
        MaskedParticleDecoder(8, num_layers=-1)


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


# ----------------------------------------------------------------------------------------------------------------- JEPA


@pytest.fixture(scope="module")
def toy_batch(toy_small: JetData) -> JetBatch:
    from itj.features import make_batch

    return make_batch(toy_small.head(48), "kinpid", label_set=None)


def test_subjet_labels_are_the_exclusive_kt_subjets(toy_batch: JetBatch) -> None:
    from itj.cluster import cluster

    real = toy_batch.particle_mask
    lab = subjet_labels(toy_batch, 2)
    assert lab.shape == real.shape
    assert lab.dtype == torch.int64
    assert torch.equal(lab < 0, ~real)
    assert set(lab[real].unique().tolist()) == {0, 1}
    p4 = toy_batch.vectors.permute(0, 2, 1).double().numpy()
    ref = cluster(p4, real.numpy(), algorithm="kt", R=0.8).exclusive_labels(2)
    np.testing.assert_array_equal(lab.numpy(), ref)
    # subjet 0 is the harder one
    pt = torch.hypot(toy_batch.vectors[:, 0], toy_batch.vectors[:, 1]).double()
    sub_pt = torch.stack(
        [torch.hypot(*(toy_batch.vectors[:, :2].double() * (lab == a)[:, None]).sum(-1).T) for a in (0, 1)]
    )
    assert (sub_pt[0] >= sub_pt[1] - 1e-6 * pt.sum(1)).all()
    with pytest.raises(ValueError, match="n_subjets"):
        subjet_labels(toy_batch, 0)


def test_subjet_labels_of_a_dataset_sliced_per_batch(toy_small: JetData) -> None:
    """The caching recipe of the ``subjet_labels`` Notes gives the labels of every batch."""
    from itj.features import JetDataset

    ds = JetDataset(toy_small.head(40), "kin", None)
    all_labels = subjet_labels(ds.batch(np.arange(len(ds)), length=ds.length), 2)
    assert all_labels.shape == (len(ds), ds.length)
    for idx in (np.arange(8), np.array([30, 3, 17, 5])):
        batch = ds.batch(idx)
        labels = all_labels[batch.index, : batch.mask.shape[-1]]
        assert torch.equal(labels, subjet_labels(batch, 2))
        a = subjet_target_mask(batch, 2, generator=torch.Generator().manual_seed(0), labels=labels)
        b = subjet_target_mask(batch, 2, generator=torch.Generator().manual_seed(0))
        assert torch.equal(a[1], b[1])


def test_subjet_target_mask_hides_one_whole_subjet(toy_batch: JetBatch) -> None:
    real = toy_batch.particle_mask
    lab = subjet_labels(toy_batch, 3)
    context, target, position = subjet_target_mask(toy_batch, 3, generator=torch.Generator().manual_seed(0), labels=lab)
    assert context.shape == target.shape == real.shape
    assert position.shape == (len(real), 3)
    assert not (context & target).any()
    assert torch.equal(context | target, real)
    assert target.any(1).all()
    assert context.any(1).all()
    for j in range(len(real)):  # the target is exactly one subjet
        chosen = lab[j][target[j]].unique()
        assert len(chosen) == 1
        assert torch.equal(target[j], lab[j] == chosen[0])
    # every subjet gets drawn, and the same generator gives the same masks; labels=None clusters itself
    draws = [
        subjet_target_mask(toy_batch, 3, generator=torch.Generator().manual_seed(s), labels=lab)[1] for s in range(6)
    ]
    assert len({int(lab[0][d[0]][0]) for d in draws}) == 3
    again = subjet_target_mask(toy_batch, 3, generator=torch.Generator().manual_seed(0))
    assert torch.equal(again[1], target)
    torch.testing.assert_close(again[2], position)
    # the position: the target four-vector sum relative to the context's, ln(pT_target / pT_context)
    v = toy_batch.vectors.double()
    pt_t = torch.hypot(*(v[:, :2] * target[:, None]).sum(-1).T)
    pt_c = torch.hypot(*(v[:, :2] * context[:, None]).sum(-1).T)
    torch.testing.assert_close(position[:, 2].double(), torch.log(pt_t / pt_c), atol=1e-5, rtol=0)
    assert float(position[:, :2].abs().max()) < 2.0
    # a one-particle jet has no target
    one = dataclasses.replace(toy_batch, mask=toy_batch.mask * (torch.arange(real.shape[1]) < 1).float())
    c1, t1, p1 = subjet_target_mask(one, 2)
    assert not t1.any()
    assert torch.equal(c1, one.particle_mask)
    assert float(p1.abs().max()) == 0.0
    with pytest.raises(ValueError, match="n_subjets"):
        subjet_target_mask(toy_batch, 1)
    with pytest.raises(ValueError, match="labels"):
        subjet_target_mask(toy_batch, 2, labels=lab[:, :3])


def test_context_view_is_a_jet_of_its_own(toy_batch: JetBatch) -> None:
    real = toy_batch.particle_mask
    # all particles kept: the batch itself (the toy jet axis is the four-vector sum)
    same = context_view(toy_batch, real)
    for name in ("features", "points", "vectors", "mask"):
        torch.testing.assert_close(getattr(same.batch, name), getattr(toy_batch, name), atol=2e-6, rtol=0)
    torch.testing.assert_close(same.positions[real], toy_batch.points.permute(0, 2, 1)[real], atol=2e-6, rtol=0)
    assert float(same.positions[~real].abs().max()) == 0.0
    # a context: equal to the batch that JetDataset builds from the context particles alone
    context, _, _ = subjet_target_mask(toy_batch, 2, generator=torch.Generator().manual_seed(1))
    view = context_view(toy_batch, context)
    k = context.sum(1)
    assert view.batch.mask.shape[-1] == int(k.max())
    assert torch.equal(view.batch.particle_mask.sum(1), k)
    for j in range(len(k)):
        n = int(k[j])
        slots = torch.nonzero(context[j]).squeeze(1)
        assert torch.equal(view.slots[j, :n], slots)  # in pT order
        # wrap padding repeats the context particles
        p = view.batch.mask.shape[-1]
        torch.testing.assert_close(view.batch.vectors[j], toy_batch.vectors[j][:, slots[torch.arange(p) % n]])
    jets = _context_jets(toy_batch, context)
    from itj.features import make_batch

    ref = make_batch(jets, "kinpid", label_set=None)
    for name in ("features", "points", "vectors", "mask"):
        torch.testing.assert_close(getattr(view.batch, name), getattr(ref, name), atol=2e-5, rtol=0)
    # the positions of every particle (also the hidden ones) in the context frame: the points of the context jet
    for j in range(len(k)):
        n = int(k[j])
        torch.testing.assert_close(view.positions[j, view.slots[j, :n]], ref.points[j, :, :n].T, atol=2e-5, rtol=0)
    with pytest.raises(ValueError, match="context"):
        context_view(toy_batch, context[:, :3])


def _context_jets(batch: JetBatch, context: torch.Tensor) -> JetData:
    """The context particles as JetData, with the jet kinematics recomputed from their four-vector sum."""
    from itj.data import Ragged

    v = batch.vectors.permute(0, 2, 1).double().numpy()
    keep = context.numpy()
    counts = keep.sum(1)
    cols = {name: v[..., i][keep] for i, name in enumerate(("part_px", "part_py", "part_pz", "part_energy"))}
    s = (v * keep[..., None]).sum(1)
    pt = np.hypot(s[:, 0], s[:, 1])
    eta, phi = np.arcsinh(s[:, 2] / pt), np.arctan2(s[:, 1], s[:, 0])
    row = np.repeat(np.arange(len(counts)), counts)
    p_eta = np.arcsinh(cols["part_pz"] / np.hypot(cols["part_px"], cols["part_py"]))
    p_phi = np.arctan2(cols["part_py"], cols["part_px"])
    deta = np.where(eta[row] >= 0, 1.0, -1.0) * (p_eta - eta[row])
    dphi = (p_phi - phi[row] + np.pi) % (2 * np.pi) - np.pi
    pid_names = (
        "part_charge",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
    )
    # the PID features of "kinpid" are raw (channels 5-10): read them back from the batch
    pid = batch.features[:, 5:11].permute(0, 2, 1).double().numpy()[keep]
    data = {name: Ragged.from_counts(x.astype(np.float32), counts) for name, x in cols.items()}
    data.update({name: Ragged.from_counts(pid[:, i].astype(np.float32), counts) for i, name in enumerate(pid_names)})
    data["part_deta"] = Ragged.from_counts(deta.astype(np.float32), counts)
    data["part_dphi"] = Ragged.from_counts(dphi.astype(np.float32), counts)
    data["jet_pt"] = pt.astype(np.float32)
    data["jet_energy"] = s[:, 3].astype(np.float32)
    return JetData(data)


@pytest.mark.parametrize("model_name", ["partsmall_nopair", "partsmall"])
def test_context_never_sees_the_target(toy_batch: JetBatch, model_name: str) -> None:
    """Changing the target particles (features, points, vectors) changes nothing the context encoder sees."""
    from itj.models import build_model

    context, target, _ = subjet_target_mask(toy_batch, 2, generator=torch.Generator().manual_seed(2))
    g = torch.Generator().manual_seed(3)
    hide = target[:, None, :]
    # also the wrap-padded copies of the target particles in the padded slots
    n = toy_batch.particle_mask.sum(1)
    src = torch.arange(target.shape[1])[None, :] % n[:, None]
    hide = hide | torch.gather(target, 1, src)[:, None, :]
    other = dataclasses.replace(
        toy_batch,
        features=torch.where(hide, torch.randn(toy_batch.features.shape, generator=g), toy_batch.features),
        points=torch.where(hide, torch.randn(toy_batch.points.shape, generator=g), toy_batch.points),
        vectors=torch.where(hide, toy_batch.vectors * 2.5, toy_batch.vectors),
    )
    a, b = context_view(toy_batch, context), context_view(other, context)
    for name in ("features", "points", "vectors", "mask"):
        assert torch.equal(getattr(a.batch, name), getattr(b.batch, name)), name
    assert torch.equal(a.slots, b.slots)
    # the positions of the context particles do not move either (those of the target do: they are the queries)
    assert torch.equal(a.positions[context], b.positions[context])
    torch.manual_seed(0)
    model = build_model(model_name, feature_set="kinpid")
    model.train()  # BatchNorm with batch statistics: no target values in them either
    torch.manual_seed(1)
    ha = model.trace(a.batch).layers["block2"]
    torch.manual_seed(1)
    hb = model.trace(b.batch).layers["block2"]
    assert torch.equal(ha, hb)
    if model_name == "partsmall":  # the pair features of the context batch involve context particles only
        tr = model.eval().trace(a.batch)
        assert tr.pair_bias is not None
        pc = a.batch.particle_mask
        assert (tr.pair_bias.abs().sum(1)[~(pc[:, :, None] & pc[:, None, :])] == 0).all()
    # the original batch with the target only masked would leak through the jet-relative features
    from itj.features import FEATURE_SETS

    names = [v.name for v in FEATURE_SETS["kinpid"]]
    var = FEATURE_SETS["kinpid"][names.index("part_logptrel")]
    assert var.center is not None
    center, scale = var.center, var.scale

    def pt_rel(features: torch.Tensor) -> torch.Tensor:  # pT / pT_jet from the standardised part_logptrel
        return torch.exp(features[:, names.index("part_logptrel")] / scale + center)

    masked = dataclasses.replace(toy_batch, mask=toy_batch.mask * context[:, None].float())
    s_masked = (pt_rel(masked.features) * context).sum(1)  # sum of pT / pT_jet over the context
    s_view = (pt_rel(a.batch.features) * a.batch.particle_mask).sum(1)
    # 1 - pT_target / pT_jet: the target pT shows
    assert float(s_masked.mean()) < 0.8
    # the context is all of its jet (the scalar sum of pT is at least the vector sum, and close to it)
    assert float(s_view.min()) > 1.0 - 1e-4
    assert float(s_view.max()) < 1.1


def test_ema_target_gets_no_gradient_and_follows_the_momentum_rule() -> None:
    from itj.models import build_model

    torch.manual_seed(0)
    model = build_model("partsmall_nopair")
    ema = EMATarget(model)
    assert ema.model is not model
    assert all(not p.requires_grad for p in ema.model.parameters())
    assert not ema.model.training
    before = {k: v.clone() for k, v in ema.model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
        dict(model.named_buffers())["mod.embed.input_bn.running_mean"].fill_(0.5)  # a buffer
    online = {k: v.clone() for k, v in model.state_dict().items()}
    ema.update(model, 0.9)
    params = dict(model.named_parameters())
    for name, value in ema.model.state_dict().items():
        if name in params:
            torch.testing.assert_close(value, 0.9 * before[name] + 0.1 * online[name])
        else:  # buffers are copied
            torch.testing.assert_close(value, online[name])
    frozen = {k: v.clone() for k, v in ema.model.state_dict().items()}
    ema.update(model, 1.0)
    for k, v in ema.model.state_dict().items():
        if k in params:
            torch.testing.assert_close(v, frozen[k])
    ema.update(model, 0.0)
    for k, v in ema.model.state_dict().items():
        torch.testing.assert_close(v, online[k])
    with pytest.raises(ValueError, match="momentum"):
        ema.update(model, 1.5)
    assert ema_momentum(0, 100) == pytest.approx(0.996)
    assert ema_momentum(50, 100) == pytest.approx(0.998)
    assert ema_momentum(200, 100) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="total"):
        ema_momentum(0, 0)


def test_jepa_predictor_and_loss(toy_batch: JetBatch) -> None:
    from itj.models import build_model

    torch.manual_seed(0)
    model = build_model("partsmall_nopair", feature_set="kinpid")
    context, target, _ = subjet_target_mask(toy_batch, 2, generator=torch.Generator().manual_seed(0))
    view = context_view(toy_batch, context)
    tr = model.trace(view.batch)
    pred_net = JEPAPredictor(64, 32, dim=48, num_layers=1)
    pred = pred_net(tr.layers["block2"], view, target, cls=tr.embedding)
    assert pred.shape == (*target.shape, 32)
    assert JEPAPredictor(64)(tr.layers["block2"], view, target).shape == (*target.shape, 64)
    # two hidden particles of a jet get different predictions (their positions differ)
    j = int(torch.argmax(target.sum(1)))
    slots = torch.nonzero(target[j]).squeeze(1)
    assert (pred[j, slots] - pred[j, slots[0]]).abs().amax() > 1e-4
    # gradients reach the context encoder, including the class-attention block through the class token
    pred[target].square().mean().backward()
    params = dict(model.named_parameters())
    assert params["mod.blocks.0.fc1.weight"].grad is not None
    assert params["mod.cls_blocks.0.fc1.weight"].grad is not None
    # without context positions, only the positions of the queries reach the predictor
    blind = JEPAPredictor(64, dim=32, num_layers=1, context_positions=False)
    moved = dataclasses.replace(view, positions=torch.where(target[..., None], view.positions, view.positions + 0.3))
    h = tr.layers["block2"].detach()
    torch.testing.assert_close(blind(h, moved, target)[target], blind(h, view, target)[target])
    pred_net.eval()
    assert not torch.allclose(pred_net(h, moved, target)[target], pred_net(h, view, target)[target])
    # the predictor refuses a context that contains a target particle
    with pytest.raises(ValueError, match="context"):
        pred_net(tr.layers["block2"].detach(), view, target | context, cls=None)
    with pytest.raises(ValueError, match="num_layers"):
        JEPAPredictor(64, num_layers=0)
    # the loss: 0 for a perfect prediction, detached targets, only the target slots count
    t = torch.randn(4, 6, 8, requires_grad=True)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    mask[:, 1] = True
    mask[0, 4] = True
    perfect = torch.nn.functional.layer_norm(t.detach(), (8,))
    for kind in ("l2", "smooth_l1"):
        assert float(jepa_loss(perfect, t, mask, kind=kind)) == pytest.approx(0.0, abs=1e-10)
    assert float(jepa_loss(t.detach(), t, mask, normalise=False)) == 0.0
    wrong = perfect.clone()
    wrong[~mask] = 100.0  # outside the target: ignored
    assert float(jepa_loss(wrong, t, mask)) == pytest.approx(0.0, abs=1e-10)
    wrong[0, 1] += 1.0
    assert float(jepa_loss(wrong, t, mask)) == pytest.approx(8 / (5 * 8))
    p = torch.zeros(4, 6, 8, requires_grad=True)
    jepa_loss(p, t, mask).backward()
    assert t.grad is None
    assert p.grad is not None
    assert float(jepa_loss(p, t, torch.zeros_like(mask)).detach()) == 0.0
    with pytest.raises(ValueError, match="same shape"):
        jepa_loss(p[:, :3], t, mask)
    with pytest.raises(ValueError, match="kind"):
        jepa_loss(p, t, mask, kind="l1")


def test_effective_rank() -> None:
    g = torch.Generator().manual_seed(0)
    assert effective_rank(torch.ones(50, 6)) == 1.0  # collapsed
    line = torch.randn(50, 1, generator=g) * torch.tensor([[1.0, 2.0, -1.0]])
    assert effective_rank(line) == pytest.approx(1.0, abs=1e-6)
    iso = torch.randn(4000, 5, generator=g)
    assert 4.8 < effective_rank(iso) <= 5.0
    assert effective_rank(iso.numpy()) == pytest.approx(effective_rank(iso))


def _jepa_run(batch: JetBatch, steps: int, seed: int, total: int = 40) -> tuple[list[float], torch.Tensor, Trace]:
    """Return the losses, the last predictions at the target slots and the context encoder's trace of the batch."""
    from itj.models import build_model
    from itj.train import set_seed

    set_seed(seed)
    model = build_model("partsmall_nopair", feature_set="kinpid")
    predictor = JEPAPredictor(64, dim=32, num_layers=1)
    ema = EMATarget(model)
    opt = torch.optim.AdamW([*model.parameters(), *predictor.parameters()], lr=3e-3)
    labels = subjet_labels(batch, 2)
    losses = []
    for step in range(steps):
        context, target, _ = subjet_target_mask(
            batch, 2, generator=torch.Generator().manual_seed(step % 2), labels=labels
        )
        view = context_view(batch, context)
        tr = model.trace(view.batch)
        pred = predictor(tr.layers["block2"], view, target, cls=tr.embedding)
        loss = jepa_loss(pred, ema.trace(batch).layers["block2"], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model, ema_momentum(step, total))
        losses.append(float(loss.detach()))
    with torch.no_grad():
        trace = model.eval().trace(batch)
    return losses, pred.detach()[target], trace


def test_a_tiny_jepa_run_learns_and_is_deterministic(toy_batch: JetBatch) -> None:
    batch = dataclasses.replace(
        toy_batch,
        features=toy_batch.features[:24],
        points=toy_batch.points[:24],
        vectors=toy_batch.vectors[:24],
        mask=toy_batch.mask[:24],
        index=toy_batch.index[:24],
    )
    losses, pred, trace = _jepa_run(batch, 40, seed=0)
    assert all(math.isfinite(x) for x in losses)
    assert np.mean(losses[-5:]) < 0.5 * np.mean(losses[:5])
    # a falling loss alone does not show learning: a collapsed JEPA (one vector for everything) has a low loss too.
    # The predictions and the encoder's representations still differ between particles and jets (rank 1: collapsed;
    # this run gives about 6, down from 14 at the start: the dimensional collapse that B.2 asks students to watch).
    assert effective_rank(pred) > 3.0
    assert effective_rank(trace.pooled["block2"]) > 3.0
    assert _jepa_run(batch, 6, seed=0)[0] == losses[:6]  # the same seed: the same run
    assert _jepa_run(batch, 6, seed=1)[0][1:] != losses[1:6]
