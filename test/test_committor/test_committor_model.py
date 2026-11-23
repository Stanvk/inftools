"""Tests for `inft committor_model` and inftools.misc.committor_net."""

import math

import numpy as np
import pytest

from inftools.misc.committor_helper import parse_dropout, parse_int_list

torch = pytest.importorskip("torch")

from inftools.misc.committor_net import (  # noqa: E402
    aimmd_state_dict,
    binomial_loss,
    build_model,
    count_parameters,
    feature_names,
    load_model,
    save_model,
)
from inftools.tistools.committor_model import committor_model  # noqa: E402

ARCH = {
    "cols": list(range(2, 14)),
    "periodic": [8, 9],
    "hidden": [64, 32, 16],
    "activation": "elu",
    "dropout": {},
}


def make(tmp_path, name="model.pt", **kwargs):
    """Run the command with defaults for these tests and load the result."""
    opts = {"cols": "2-13", "periodic": "8,9", "hidden": "64,32,16"}
    opts.update(kwargs)
    path = tmp_path / name
    committor_model(out=str(path), **opts)
    return load_model(path)


def test_parse_int_list():
    assert parse_int_list("2-5") == [2, 3, 4, 5]
    assert parse_int_list("2, 5-7,10") == [2, 5, 6, 7, 10]
    assert parse_int_list("") == []
    assert parse_int_list("64,32,16") == [64, 32, 16]
    with pytest.raises(ValueError):
        parse_int_list("5-3")
    with pytest.raises(ValueError):
        parse_int_list("2-4,3")
    with pytest.raises(ValueError):
        parse_int_list("a")


def test_parse_dropout():
    assert parse_dropout("", 3) == {}
    assert parse_dropout("0:0.1,2:0.2", 3) == {"0": 0.1, "2": 0.2}
    for bad in ["3:0.1", "0:1.0", "0", "0:0.1,0:0.2"]:
        with pytest.raises(ValueError):
            parse_dropout(bad, 3)


def test_create_default_model(tmp_path):
    """12 columns with 2 periodic give 14 features and 3585 parameters."""
    model, ckpt = make(tmp_path)
    assert ckpt["arch"] == ARCH
    assert ckpt["trained"] is False
    assert ckpt["history"] == []
    assert ckpt["meta"]["seed"] == 0
    assert model.transform.n_out == 14
    # (14*64 + 64) + (64*32 + 32) + (32*16 + 16) + (16 + 1)
    assert count_parameters(model) == 3585
    assert feature_names(ckpt["arch"])[-4:] == [
        "sin(c8)",
        "sin(c9)",
        "cos(c8)",
        "cos(c9)",
    ]
    q = model(torch.randn(5, 12))
    assert q.shape == (5, 1)
    # standardization starts as the identity
    assert torch.all(model.transform.mean == 0)
    assert torch.all(model.transform.std == 1)


def test_seed(tmp_path):
    m0, _ = make(tmp_path, "a.pt", seed=0)
    m0b, _ = make(tmp_path, "b.pt", seed=0)
    m1, _ = make(tmp_path, "c.pt", seed=1)
    for (k, v), v1 in zip(m0.state_dict().items(), m1.state_dict().values()):
        assert torch.equal(v, m0b.state_dict()[k])
    w0 = m0.state_dict()["log_predictor.weight"]
    assert not torch.equal(w0, m1.state_dict()["log_predictor.weight"])


def test_features_and_periodicity():
    """Periodic columns become (sin, cos) and q does not see a 2 pi shift."""
    model = build_model(ARCH).eval()
    x = torch.randn(7, 12)
    feats = model.transform.features(x)
    # columns 8 and 9 are at positions 6 and 7 of the 12 inputs
    lin = [i for i in range(12) if i not in (6, 7)]
    assert torch.equal(feats[:, :10], x[:, lin])
    assert torch.allclose(feats[:, 10:12], torch.sin(x[:, [6, 7]]))
    assert torch.allclose(feats[:, 12:14], torch.cos(x[:, [6, 7]]))
    shifted = x.clone()
    shifted[:, 6] += 2 * math.pi
    shifted[:, 7] -= 4 * math.pi
    with torch.no_grad():
        assert torch.allclose(model(x), model(shifted), atol=1e-5)


def test_aimmd_names():
    """Parameter names match aimmd's ModuleStack(FFNet)."""
    model = build_model(ARCH)
    expected = {
        f"module_list.0.hidden_layers.{i}.{p}"
        for i in range(3)
        for p in ("weight", "bias")
    } | {"log_predictor.weight", "log_predictor.bias"}
    assert set(aimmd_state_dict(model)) == expected
    transform = {k for k in model.state_dict() if k.startswith("transform.")}
    assert transform == {
        "transform.lin_idx",
        "transform.per_idx",
        "transform.mean",
        "transform.std",
    }


def test_save_load_keeps_standardization(tmp_path):
    model = build_model(ARCH).eval()
    model.transform.mean.copy_(torch.randn(14))
    model.transform.std.copy_(torch.rand(14) + 0.5)
    save_model(tmp_path / "m.pt", model, ARCH, trained=True)
    loaded, ckpt = load_model(tmp_path / "m.pt")
    assert ckpt["trained"] is True
    x = torch.randn(9, 12)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))


def test_torchscript(tmp_path):
    """The model scripts, and the scripted copy gives the same q."""
    model = build_model(dict(ARCH, dropout={"0": 0.2})).eval()
    scripted = torch.jit.script(model)
    scripted.save(str(tmp_path / "s.pt"))
    reloaded = torch.jit.load(str(tmp_path / "s.pt"))
    x = torch.randn(11, 12)
    with torch.no_grad():
        assert torch.allclose(model(x), reloaded(x))


def test_no_periodic_columns():
    model = build_model(dict(ARCH, periodic=[])).eval()
    assert model.transform.n_out == 12
    assert model(torch.randn(3, 12)).shape == (3, 1)


def test_binomial_loss():
    q = torch.tensor([[-2.0], [0.5], [3.0]])
    shots = torch.tensor([[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    w = torch.tensor([1.0, 1e-3, 10.0])
    expected = sum(
        wi * (na * math.log1p(math.exp(qi)) + nb * math.log1p(math.exp(-qi)))
        for wi, qi, (na, nb) in zip(
            w.tolist(), q[:, 0].tolist(), shots.tolist()
        )
    )
    assert binomial_loss(q, shots, w).item() == pytest.approx(expected)
    # for one shared q the minimum is at q = ln(sum w n_B / sum w n_A)
    q_best = math.log((1e-3 * 1 + 10 * 2) / (1 * 2 + 1e-3 * 1))
    qt = torch.full((3, 1), q_best, requires_grad=True)
    binomial_loss(qt, shots, w).backward()
    assert abs(qt.grad.sum().item()) < 1e-6


def test_invalid_input(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, "a.pt", periodic="15")  # not an input column
    with pytest.raises(ValueError):
        make(tmp_path, "b.pt", cols="0-3", periodic="")  # time column
    with pytest.raises(ValueError):
        make(tmp_path, "c.pt", activation="prelu")
    with pytest.raises(ValueError):
        make(tmp_path, "d.pt", dropout="3:0.1")  # only layers 0-2


def test_no_overwrite(tmp_path):
    make(tmp_path, "m.pt")
    with pytest.raises(ValueError):
        make(tmp_path, "m.pt")
    make(tmp_path, "m.pt", overw=True)


def test_large_inputs_are_finite():
    """softplus keeps the loss finite for large |q|."""
    q = torch.tensor([[-500.0], [500.0]])
    shots = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    loss = binomial_loss(q, shots, torch.ones(2))
    assert np.isfinite(loss.item())
    assert loss.item() == pytest.approx(2 * 500 + 2 * 500)
