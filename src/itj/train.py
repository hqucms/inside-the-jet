"""Training and inference on the CPU: :func:`fit`, :func:`predict`, :class:`History` and the thread and seed helpers.

:func:`fit` is the reference training loop of the course (students write their own first loop in T2.3, then may use
it): AdamW with a cosine schedule, (weighted) cross entropy, the best state by validation loss restored at the end,
an ETA printed after 20 batches, a time limit (``max_minutes``) and early stopping (``patience``). On the CPU a run
is deterministic for a given seed and initial model.

Examples
--------
>>> import numpy as np
>>> from itj.data import JetData, Ragged
>>> from itj.features import JetDataset
>>> from itj.models import build_model
>>> rng = np.random.default_rng(0)
>>> n = 64
>>> counts = np.full(n, 4)
>>> pt = Ragged.from_counts(rng.uniform(10, 100, 4 * n).astype(np.float32), counts)
>>> ang = Ragged.from_counts(rng.normal(0, 0.2, 4 * n).astype(np.float32), counts)
>>> labels = np.repeat(np.eye(3, dtype=bool), [22, 21, 21], axis=0)
>>> data = JetData({"part_px": pt, "part_py": ang, "part_pz": ang, "part_energy": pt.map(lambda p: 1.5 * p),
...     "part_deta": ang, "part_dphi": ang, "jet_pt": rng.uniform(200, 300, n).astype(np.float32),
...     "jet_energy": rng.uniform(300, 400, n).astype(np.float32), "label_QCD": labels[:, 0],
...     "label_Wqq": labels[:, 1], "label_Tbqq": labels[:, 2]})
>>> model = build_model("pfn", phi_sizes=(16, 16), f_sizes=(16,))
>>> history = fit(model, JetDataset(data), epochs=2, batch_size=16, quiet=True)
>>> len(history.train_loss), predict(model, data).shape
(2, (64, 3))
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import time
from collections.abc import Callable, Iterable, Sized
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from torch import nn

from .features import JetBatch, JetDataset, make_batch, make_loader

if TYPE_CHECKING:
    from .data import JetData
    from .models import Tagger

__all__ = ["History", "fit", "predict", "set_seed", "set_threads"]

_log = logging.getLogger("itj")

#: Batches after which :func:`fit` prints its first ETA.
ETA_AFTER = 20


def set_threads(n: int | None = None) -> int:
    """Set the number of torch CPU threads (``torch.set_num_threads``).

    Parameters
    ----------
    n : int, optional
        Number of threads; None: ``min(6, os.cpu_count())`` (the reference machine of the CPU budget has 6 cores).

    Returns
    -------
    int
        The number of threads set.
    """
    if n is None:
        n = min(6, os.cpu_count() or 1)
    if n < 1:
        raise ValueError(f"the number of threads must be >= 1, got {n}")
    torch.set_num_threads(int(n))
    return int(n)


def set_seed(seed: int) -> None:
    """Seed Python's ``random``, NumPy's global generator and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 (the legacy global generator, for code that still uses it)
    torch.manual_seed(seed)


@dataclass
class History:
    """What :func:`fit` measured, one entry per epoch.

    Attributes
    ----------
    train_loss, val_loss : list of float
        Mean loss per jet over the epoch (training: with the weights in training mode; validation: in eval mode,
        NaN without validation data).
    train_acc, val_acc : list of float
        Accuracy (argmax of the logits); NaN without labels or with a custom ``loss``.
    wall_time_s : list of float
        Seconds since the start of :func:`fit` at the end of each epoch (validation included), so
        ``wall_time_s[-1]`` is the total time.
    jets_per_s : list of float
        Training jets per second of the epoch (forward, backward and data handling; validation excluded).
    steps : list of int
        Optimiser steps done since the start of :func:`fit` at the end of each epoch.
    best_epoch : int
        Epoch (0-based) whose state the model holds at the end: lowest validation loss, or the last epoch without
        validation data (-1 before the first epoch).
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    wall_time_s: list[float] = field(default_factory=list)
    jets_per_s: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    best_epoch: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Return the history as a dict of plain lists and ints (JSON-serialisable; NaN stays NaN)."""
        return asdict(self)

    def __len__(self) -> int:
        return len(self.train_loss)


# ----------------------------------------------------------------------------------------------------------------------
# helpers


def _check_compatible(model: nn.Module, dataset: object, what: str) -> None:
    """Raise a ValueError when the model and a JetDataset disagree on the particle or jet features."""
    if not isinstance(dataset, JetDataset):
        return
    m_fs = getattr(model, "feature_set", None)
    d_fs = getattr(dataset, "feature_set", None)
    if m_fs is not None and d_fs is not None and m_fs != d_fs:
        raise ValueError(
            f"the model reads feature_set={m_fs!r}, the {what} dataset has feature_set={d_fs!r}: build it with "
            f"JetDataset(data, {m_fs!r}, ...)"
        )
    m_jf = getattr(model, "jet_features", None)
    d_jf = tuple(getattr(dataset, "jet_feature_names", ()) or ()) or None
    if m_jf is not None and d_jf is not None and tuple(m_jf) != d_jf:
        raise ValueError(
            f"the model reads jet_features={tuple(m_jf)}, the {what} dataset has jet_feature_names={d_jf}: build it "
            "with JetDataset(..., jet_feature_names=model.jet_features)"
        )


def _batches(source: Any, batch_size: int, *, shuffle: bool, seed: int) -> Iterable[JetBatch]:
    """Return a re-iterable source of batches: a loader for a JetDataset, the iterable itself otherwise."""
    if isinstance(source, JetDataset):
        return make_loader(source, batch_size, shuffle=shuffle, seed=seed)
    if not isinstance(source, Sized):
        source = list(source)  # a generator is exhausted after one epoch: keep its batches
    return source


def _n_batches(batches: Iterable[JetBatch]) -> int:
    return len(batches) if isinstance(batches, Sized) else 0


def _cross_entropy(logits: torch.Tensor, batch: JetBatch) -> torch.Tensor:
    if batch.label is None:
        raise ValueError("the batch has no labels: build the dataset with a label_set, or pass loss=...")
    per_jet = nn.functional.cross_entropy(logits, batch.label, reduction="none")
    if batch.weight is None:
        return per_jet.mean()
    w = batch.weight.to(per_jet.dtype)
    return (per_jet * w).sum() / w.sum().clamp(min=torch.finfo(w.dtype).tiny)


def _make_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}; choose from adamw, adam, sgd")


def _lr_factor(step: int, *, total: int, warmup: int, scheduler: str) -> float:
    """Learning-rate factor of optimiser step ``step``: linear warm-up, then cosine to 0 (or constant)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if scheduler == "none" or total <= warmup:
        return 1.0
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _evaluate(
    model: nn.Module, batches: Iterable[JetBatch], loss: Callable[[Any, JetBatch], torch.Tensor] | None, device: str
) -> tuple[float, float]:
    """Return (mean loss per jet, accuracy) of the model in eval mode."""
    model.eval()
    total_loss, n_jets, n_correct, n_labelled = 0.0, 0, 0, 0
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            b = len(batch)
            if loss is None:
                logits = model(batch)
                value = _cross_entropy(logits, batch)
            else:
                value = loss(model, batch)
                logits = model(batch) if batch.label is not None else None
            total_loss += float(value) * b
            n_jets += b
            if logits is not None and batch.label is not None:
                n_correct += int((logits.argmax(dim=1) == batch.label).sum())
                n_labelled += b
    return (total_loss / n_jets if n_jets else math.nan), (n_correct / n_labelled if n_labelled else math.nan)


def _say(quiet: bool, message: str) -> None:
    if quiet:
        _log.debug(message)
    else:
        print(message, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# fit and predict


def fit(
    model: Tagger,
    train: JetDataset | Iterable[JetBatch],
    val: JetDataset | Iterable[JetBatch] | None = None,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw",
    scheduler: Literal["cosine", "none"] = "cosine",
    warmup_steps: int = 0,
    min_steps: int = 0,
    loss: Callable[[Tagger, JetBatch], torch.Tensor] | None = None,
    seed: int = 0,
    device: str = "cpu",
    threads: int | None = None,
    max_minutes: float | None = None,
    patience: int | None = None,
    val_every: int | None = None,
    log_every: int = 50,
    callback: Callable[[int, History], None] | None = None,
    quiet: bool = False,
) -> History:
    """Train a model and return its :class:`History`; the model ends in eval mode with the best state.

    Parameters
    ----------
    model : Tagger
        The model (any ``nn.Module`` taking a :class:`~itj.features.JetBatch` works).
    train : JetDataset or iterable of JetBatch
        Training jets: a ``JetDataset`` is shuffled into batches of ``batch_size`` with a new order every epoch
        (``make_loader(train, batch_size, seed=seed)``); an iterable of batches is used as it is (a generator is
        read into a list first).
    val : JetDataset or iterable of JetBatch, optional
        Validation jets, evaluated after every epoch (or as ``val_every`` says); the state with the lowest validation
        loss is restored at the end. None: no validation, the final state is kept.
    epochs : int
        Number of passes over ``train``.
    batch_size : int
        Jets per batch (``JetDataset`` only).
    lr, weight_decay : float
        Peak learning rate and weight decay.
    optimizer : {"adamw", "adam", "sgd"}
        The optimiser (SGD with momentum 0.9).
    scheduler : {"cosine", "none"}
        Cosine decay of the learning rate to 0 over all steps, or constant; both after ``warmup_steps`` steps of
        linear warm-up.
    warmup_steps : int
        Optimiser steps of linear warm-up.
    min_steps : int
        Plan at least this many optimiser steps: ``epochs`` (and with it the length of the cosine schedule) is
        raised to ``ceil(min_steps * batch_size / n_train)`` (``ceil(min_steps / batches per epoch)`` for an
        iterable) when that is larger, so that a small training set gets enough epochs. ``patience`` and
        ``max_minutes`` can still stop the run earlier. ``patience`` counts validations, by default one per epoch:
        an epoch of 1 000 jets is only 4 steps of 256, so ``patience=5`` alone stops such a set after a few dozen
        steps. Give a small set ``val_every`` as well (``min_steps=2000, val_every=50, patience=5``: stop after 250
        steps without improvement).
    loss : callable, optional
        ``loss(model, batch) -> scalar tensor`` replacing the default cross entropy (weighted by ``batch.weight``
        when the batch has weights: ``sum_i w_i CE_i / sum_i w_i``); the accuracies are then NaN in training.
    seed : int
        Seeds torch, NumPy and ``random`` at the start (dropout, batch order).
    device : str
        Torch device of the training.
    threads : int, optional
        Torch CPU threads during the fit (restored afterwards); None keeps the current setting (see
        :func:`set_threads`).
    max_minutes : float, optional
        Stop after this many minutes of wall time (checked after every batch; the partial epoch is recorded).
    patience : int, optional
        Stop when the validation loss has not improved for this many validations in a row (epochs, unless
        ``val_every`` is given).
    val_every : int, optional
        Validate after the first epoch that ends at least this many optimiser steps after the previous validation,
        instead of after every epoch, and always after the last epoch (also when ``max_minutes`` stops the run). For
        small training sets, whose epochs are a few steps: it makes ``patience`` a number of steps
        (at least ``patience * val_every``) and saves the time of the validations. The epochs without validation have
        NaN ``val_loss`` and ``val_acc`` in the :class:`History`, and the epoch line is printed only at a validation
        (without ``val``, on the same schedule). None: after every epoch.
    log_every : int
        Print the running training loss every ``log_every`` batches (0: never).
    callback : callable, optional
        ``callback(epoch, history)`` after every epoch.
    quiet : bool
        Print nothing (the messages go to the ``itj`` logger at DEBUG level).

    Returns
    -------
    History
        Losses, accuracies, wall time and speed per epoch, and the epoch of the state the model holds.

    Raises
    ------
    ValueError
        If the model and a ``JetDataset`` disagree on ``feature_set`` or ``jet_features`` (checked where neither is
        None), or for invalid arguments.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if val_every is not None and val_every < 1:
        raise ValueError(f"val_every must be >= 1, got {val_every}")
    if patience is not None and patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    if scheduler not in ("cosine", "none"):
        raise ValueError(f"unknown scheduler {scheduler!r}; choose from cosine, none")
    _check_compatible(model, train, "training")
    if val is not None:
        _check_compatible(model, val, "validation")

    old_threads = torch.get_num_threads()
    if threads is not None:
        set_threads(threads)
    try:
        set_seed(seed)
        model.to(device)
        train_batches = _batches(train, batch_size, shuffle=True, seed=seed)
        val_batches = None if val is None else _batches(val, max(batch_size, 1024), shuffle=False, seed=seed)
        per_epoch = _n_batches(train_batches)
        if per_epoch == 0:
            raise ValueError("no training batches")
        if min_steps > 0:
            if isinstance(train, JetDataset):
                needed = math.ceil(min_steps * batch_size / max(1, len(train)))
            else:
                needed = math.ceil(min_steps / per_epoch)
            epochs = max(epochs, needed)
        total_steps = epochs * per_epoch
        opt = _make_optimizer(model, optimizer, lr, weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: _lr_factor(step, total=total_steps, warmup=warmup_steps, scheduler=scheduler)
        )

        history = History()
        best_loss, best_state, since_best = math.inf, None, 0
        start = time.perf_counter()
        deadline = None if max_minutes is None else start + 60.0 * max_minutes
        step = 0
        last_check = 0  # optimiser step of the last validation (or epoch line, without validation data)
        stop = False
        for epoch in range(epochs):
            model.train()
            t0 = time.perf_counter()
            sum_loss, n_jets, n_correct, n_labelled = 0.0, 0, 0, 0
            for i, batch in enumerate(train_batches):
                batch = batch.to(device)
                b = len(batch)
                if b < 2:  # BatchNorm cannot train on one jet
                    continue
                if loss is None:
                    logits = model(batch)
                    value = _cross_entropy(logits, batch)
                else:
                    logits = None
                    value = loss(model, batch)
                opt.zero_grad(set_to_none=True)
                value.backward()
                opt.step()
                sched.step()
                step += 1
                sum_loss += float(value.detach()) * b
                n_jets += b
                if logits is not None and batch.label is not None:
                    n_correct += int((logits.detach().argmax(dim=1) == batch.label).sum())
                    n_labelled += b
                if epoch == 0 and i + 1 == ETA_AFTER:
                    rate = n_jets / max(time.perf_counter() - t0, 1e-9)
                    remaining = (total_steps - step) * (n_jets / (i + 1)) / rate
                    _say(
                        quiet,
                        f"{ETA_AFTER} batches: {rate:,.0f} jets/s, ETA {remaining / 60:.1f} min for {epochs} epochs "
                        "(validation not included)",
                    )
                if log_every and (i + 1) % log_every == 0:
                    _say(quiet, f"epoch {epoch + 1} batch {i + 1}/{per_epoch}: loss {sum_loss / n_jets:.4f}")
                if deadline is not None and time.perf_counter() > deadline:
                    stop = True
                    _say(quiet, f"max_minutes={max_minutes} reached after {step} steps")
                    break
            train_time = time.perf_counter() - t0
            history.train_loss.append(sum_loss / n_jets if n_jets else math.nan)
            history.train_acc.append(n_correct / n_labelled if n_labelled else math.nan)
            history.jets_per_s.append(n_jets / train_time if train_time > 0 else math.nan)
            history.steps.append(step)
            due = val_every is None or step - last_check >= val_every or stop or epoch == epochs - 1
            if due:
                last_check = step
            validate = due and val_batches is not None
            if validate:
                assert val_batches is not None
                v_loss, v_acc = _evaluate(model, val_batches, loss, device)
            else:
                v_loss, v_acc = math.nan, math.nan
            history.val_loss.append(v_loss)
            history.val_acc.append(v_acc)
            history.wall_time_s.append(time.perf_counter() - start)
            if val_batches is None:
                history.best_epoch = epoch
            elif validate:
                if v_loss < best_loss:
                    best_loss, since_best = v_loss, 0
                    history.best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    since_best += 1
            if due:
                _say(
                    quiet,
                    f"epoch {epoch + 1}/{epochs} (step {step}): train loss {history.train_loss[-1]:.4f} acc "
                    f"{history.train_acc[-1]:.4f} | val loss {v_loss:.4f} acc {v_acc:.4f} | "
                    f"{history.jets_per_s[-1]:,.0f} jets/s, {history.wall_time_s[-1]:.1f} s",
                )
            if callback is not None:
                callback(epoch, history)
            if patience is not None and validate and since_best >= patience:
                what = "validations" if val_every is not None else "epochs"
                _say(quiet, f"no improvement of the validation loss for {patience} {what} (step {step}): stopping")
                stop = True
            if stop:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return history
    finally:
        torch.set_num_threads(old_threads)


def predict(
    model: Tagger,
    data: JetData | JetDataset,
    *,
    batch_size: int = 1024,
    device: str = "cpu",
    feature_set: str | None = None,
) -> np.ndarray:
    """Return the softmax probabilities (N, C) of a model on every jet, in the order of the data.

    Parameters
    ----------
    model : Tagger
        The model (evaluated in eval mode; its training mode is restored afterwards).
    data : JetData or JetDataset
        The jets: a ``JetData`` is cut into blocks of ``batch_size`` jets and every block becomes the batch the
        model needs (:func:`~itj.models.make_inputs`, padded to its longest jet); a ``JetDataset`` gives its own
        batches (it must hold the features of the model).
    batch_size : int
        Jets per forward pass (memory grows with it; ParT: B H P^2 floats per block).
    device : str
        Torch device.
    feature_set : str, optional
        Particle features for a model without a ``feature_set`` attribute (a student model); None: the model's.

    Returns
    -------
    numpy.ndarray
        (N, C) float32 probabilities.

    Notes
    -----
    ParticleNet's kNN in feature space can flip between near-equal neighbours when the padded length changes (one
    jet in a few hundred, by up to 1e-3 in the scores): for scores equal to weaver's to float precision, build the
    batches with ``make_inputs(model, data, length=128)`` as weaver does.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    fs = feature_set if feature_set is not None else getattr(model, "feature_set", None)
    jet_features = getattr(model, "jet_features", None)
    if fs is None and jet_features is None and not isinstance(data, JetDataset):
        raise ValueError(
            "the model has neither a feature_set nor jet_features attribute (a model of your own): pass "
            "feature_set=... (for example 'kin'), or predict on a JetDataset"
        )
    was_training = model.training
    model.to(device).eval()
    outputs: list[np.ndarray] = []
    try:
        with torch.no_grad():
            if isinstance(data, JetDataset):
                _check_compatible(model, data, "prediction")
                n = len(data)
                for start in range(0, n, batch_size):
                    batch = data.batch(np.arange(start, min(start + batch_size, n)))
                    outputs.append(torch.softmax(model(batch.to(device)), dim=1).float().cpu().numpy())
            else:
                for chunk in data.chunks(batch_size):
                    batch = make_batch(
                        chunk,
                        fs,
                        label_set=None,
                        jet_feature_names=None if jet_features is None else tuple(jet_features),
                    )
                    outputs.append(torch.softmax(model(batch.to(device)), dim=1).float().cpu().numpy())
    finally:
        model.train(was_training)
    if not outputs:
        n_out = int(getattr(model, "n_classes", 0))
        return np.zeros((0, n_out), dtype=np.float32)
    return np.concatenate(outputs, axis=0)
