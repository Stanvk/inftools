"""Tests for `inft committor_train` and inftools.misc.committor_shots."""

import json
import math

import numpy as np
import pytest

from inftools.misc.committor_shots import (
    calibration_table,
    collect_run,
    read_run_settings,
    split_rows,
)

TOML_AB = """
[simulation]
interfaces = [0.0, 0.2, 0.5, 1.0]
[simulation.tis_set]
n_jumps = 2
"""
TOML_BA = """
[simulation]
interfaces = [-1.0, -0.5, -0.2, 0.0]
[simulation.tis_set]
n_jumps = 2
"""


def jump(ens, cstep, j, sp, bwd, fwd, left, cap, status, fwd_ok=True):
    """A wf_jump record; bwd, fwd are the order parameters of the ends."""
    return {
        "ens": ens, "tag": ["wf_jump", j], "is_extended": False, "jump": j,
        "interfaces": [left, left, cap], "sp_order": list(sp),
        "bwd_end": [bwd, 0.0, 0.0], "fwd_end": [fwd, 0.0, 0.0],
        "bwd_len": 3, "fwd_len": 3, "fwd_ok": fwd_ok, "cstep": cstep,
        "mc_status": status,
    }


def extended(ens, cstep, j, sp, bwd, fwd, status="ACC"):
    """A wf_extended record."""
    return {
        "ens": ens, "tag": ["wf_extended", j], "is_extended": True,
        "jump": j, "interfaces": [0.0, 0.2, 1.0], "sp_order": list(sp),
        "bwd_end": [bwd, 0.0, 0.0], "fwd_end": [fwd, 0.0, 0.0],
        "bwd_len": 0, "fwd_len": 10, "fwd_ok": True, "cstep": cstep,
        "mc_status": status,
    }


def sh(ens, cstep, sp):
    """An sh record."""
    return {
        "ens": ens, "tag": ["sh", -1], "is_extended": False, "jump": -1,
        "interfaces": [0.0, 0.0, 0.2], "sp_order": list(sp),
        "bwd_end": [0.1, 0.0, 0.0], "fwd_end": [0.1, 0.0, 0.0],
        "bwd_len": 2, "fwd_len": 2, "fwd_ok": True, "cstep": cstep,
        "mc_status": "ACC",
    }


def write(path, records):
    with open(path, "w") as wfile:
        for rec in records:
            wfile.write(json.dumps(rec) + "\n")
    return str(path)


def selection_records():
    """One record of every kind for a run with interfaces 0, 0.2, 0.5, 1.

    Halves of a jump in ensemble 002 end at <= 0.2 (exit) or >= 1 (cap);
    ends of an extended path at <= 0 (start state) or >= 1 (far state).
    """
    J = lambda c, j, sp, b, f, st, ok=True: jump(  # noqa: E731
        "002", c, j, sp, b, f, 0.2, 1.0, st, ok
    )
    return [
        # 1: last jump extended to start/far state -> one end in each
        J(1, 0, [0.31, 1, 1], 0.1, 0.1, "ACC"),
        J(1, 1, [0.3, 1, 1], 0.1, 0.15, "ACC"),
        extended("002", 1, 1, [0.3, 1, 1], -0.1, 1.2),
        # 2: both halves of the last jump at the cap
        J(2, 0, [0.4, 2, 2], 0.1, 0.1, "BWI"),
        J(2, 1, [0.4, 2, 2], 1.1, 1.3, "BWI"),
        # 3: one half at the cap, move rejected with BWI
        J(3, 0, [0.5, 3, 3], 0.1, 0.1, "BWI"),
        J(3, 1, [0.5, 3, 3], 0.1, 1.1, "BWI"),
        # 4, 5: extension exceeded the maximum length
        J(4, 1, [0.6, 4, 4], 0.1, 0.1, "BTX"),
        J(5, 1, [0.6, 5, 5], 0.1, 0.1, "FTX"),
        # 6: forward half of the last jump cut off; jump 0 was extended
        J(6, 0, [0.7, 6, 6], 0.1, 0.1, "ACC"),
        extended("002", 6, 0, [0.7, 6, 6], -0.1, -0.2),
        J(6, 1, [0.7, 6, 6], 0.1, 0.5, "ACC", ok=False),
        # 7: extended record with rounded shooting point -> matched
        J(7, 1, [0.35, 7, 7], 0.1, 0.1, "ACC"),
        extended("002", 7, 1, [0.35 + 1e-6, 7, 7], -0.05, -0.2),
        # 8: extended record with a different shooting point -> dropped
        J(8, 1, [0.35, 8, 8], 0.1, 0.1, "ACC"),
        extended("002", 8, 1, [0.45, 8, 8], -0.05, -0.2),
        # 9: accepted, no extended record -> end states unknown
        J(9, 1, [0.35, 9, 9], 0.1, 0.1, "ACC"),
        # sh: [0-] inside / outside the start state, and [0+]
        sh("000", 10, [-0.05, 10, 10]),
        sh("000", 11, [0.05, 11, 11]),
        sh("001", 12, [0.1, 12, 12]),
    ]


def test_selection_ab(tmp_path):
    shots = write(tmp_path / "shots.jsonl", selection_records())
    (tmp_path / "t.toml").write_text(TOML_AB)
    settings = read_run_settings(tmp_path / "t.toml")
    res = collect_run(shots, settings, "AB")
    s = res["summary"]
    assert s["last jumps"] == 9
    assert s["intermediate jumps (dropped)"] == 4
    assert s["  both halves reached the cap"] == 1
    assert s["  extended, end states from wf_extended"] == 2
    assert s["  status BWI, both ends in the far state"] == 1
    assert s["  extension exceeded maximum length (dropped)"] == 2
    assert s["  forward half of the jump cut off (dropped)"] == 1
    assert s["  wf_extended shooting point differs (dropped)"] == 1
    assert s["  end states unknown (dropped)"] == 1
    assert s["  end states unknown, by status"] == {"ACC": 1}
    assert s["  of intermediate jumps (dropped)"] == 1
    assert s["  [0-] shooting points inside the starting state"] == 1
    outside = "  [0-] shooting points outside the starting state (dropped)"
    assert s[outside] == 1
    assert s["  other ensembles (dropped)"] == 1
    # rows: shots 1, 2, 3, 7 and one anchor; counts are [n_A, n_B]
    assert res["shots"].tolist() == [[1, 1], [0, 2], [0, 2], [2, 0], [2, 0]]
    assert res["kind"].tolist() == [0, 0, 0, 0, 1]
    assert res["sp"][:, 0].tolist() == [0.3, 0.4, 0.5, 0.35, -0.05]
    assert res["ens"].tolist() == [2, 2, 2, 2, 0]


def test_selection_ba_swaps_states(tmp_path):
    """In a B->A run the start state is B and the far state is A."""
    shots = write(tmp_path / "shots.jsonl", selection_records())
    (tmp_path / "t.toml").write_text(TOML_AB)
    res = collect_run(shots, read_run_settings(tmp_path / "t.toml"), "BA")
    assert res["shots"].tolist() == [[1, 1], [2, 0], [2, 0], [0, 2], [0, 2]]


def test_no_anchors(tmp_path):
    shots = write(tmp_path / "shots.jsonl", selection_records())
    (tmp_path / "t.toml").write_text(TOML_AB)
    res = collect_run(
        shots, read_run_settings(tmp_path / "t.toml"), "AB", anchors=False
    )
    assert res["kind"].tolist() == [0, 0, 0, 0]
    key = "  [0-] shooting points, anchors switched off (dropped)"
    assert res["summary"][key] == 2


def test_invalid_input(tmp_path):
    (tmp_path / "t.toml").write_text(TOML_AB)
    settings = read_run_settings(tmp_path / "t.toml")
    # cap below the last interface: halves at the cap are not in B
    rec = jump("002", 1, 1, [0.3, 0, 0], 0.1, 0.95, 0.2, 0.9, "ACC")
    bad_cap = write(tmp_path / "a.jsonl", [rec])
    with pytest.raises(ValueError):
        collect_run(bad_cap, settings, "AB")
    with pytest.raises(ValueError):
        collect_run(bad_cap, dict(settings, interface_cap=0.9), "AB")
    old = jump("002", 1, 1, [0.3, 0, 0], 0.1, 0.1, 0.2, 1.0, "ACC")
    del old["cstep"]
    with pytest.raises(ValueError):
        collect_run(write(tmp_path / "b.jsonl", [old]), settings, "AB")
    with pytest.raises(ValueError):
        collect_run(write(tmp_path / "c.jsonl", []), settings, "CD")


def test_split_and_calibration():
    tr, va, te = split_rows(103, [8, 1, 1], seed=3)
    assert len(tr) + len(va) + len(te) == 103
    assert len(set(tr) | set(va) | set(te)) == 103
    p = np.linspace(0.05, 0.95, 40)
    shots = np.stack([2 - 2 * (p > 0.5), 2 * (p > 0.5)], axis=1)
    table = calibration_table(p, shots, n_bins=4)
    assert [row[1] for row in table] == [0.0, 0.0, 1.0, 1.0]
    assert sum(row[3] for row in table) == 80


# end-to-end test with a known committor ---------------------------------

def true_pb(x):
    return 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))


def synthetic_run(path, direction, n, seed):
    """Shots whose halves end in B with probability true_pb(x).

    Column 1 is the run's order parameter (x for A->B, -x for B->A),
    column 2 is x and column 3 is noise.
    """
    rng = np.random.default_rng(seed)
    ab = direction == "AB"
    left, cap = (0.2, 1.0) if ab else (-0.5, 0.0)
    lam0, lamN = (0.0, 1.0) if ab else (-1.0, 0.0)
    records = []
    for c in range(n):
        x = rng.uniform(0, 1)
        sp = [x if ab else -x, x, rng.normal()]
        in_b = rng.uniform(size=2) < true_pb(x)
        # far state of the run: B for A->B, A for B->A
        far = in_b if ab else ~in_b
        if far.all():
            records.append(jump("002", c, 1, sp, cap + 0.1, cap + 0.2,
                                left, cap, "BWI"))
            continue
        records.append(jump("002", c, 1, sp, left - 0.1, left - 0.1,
                            left, cap, "ACC"))
        ends = [lamN + 0.1 if f else lam0 - 0.1 for f in far]
        records.append(extended("002", c, 1, sp, *ends))
    return write(path, records)


@pytest.fixture
def two_runs(tmp_path):
    (tmp_path / "ab.toml").write_text(TOML_AB)
    (tmp_path / "ba.toml").write_text(TOML_BA)
    ab = synthetic_run(tmp_path / "ab.jsonl", "AB", 3000, seed=1)
    ba = synthetic_run(tmp_path / "ba.jsonl", "BA", 3000, seed=2)
    return {
        "shots": [ab, ba],
        "toml": [str(tmp_path / "ab.toml"), str(tmp_path / "ba.toml")],
        "direction": ["AB", "BA"],
    }


def test_build_only(tmp_path, two_runs):
    from inftools.tistools.committor_train import committor_train

    out = tmp_path / "out"
    committor_train(**two_runs, out=str(out), build_only=True)
    data = np.load(out / "dataset.npz")
    assert len(data["sp"]) == 6000
    assert data["run"].tolist() == [0] * 3000 + [1] * 3000
    assert (out / "report.txt").exists()
    assert not (out / "model_trained.pt").exists()


def test_train_recovers_committor(tmp_path, two_runs):
    torch = pytest.importorskip("torch")
    from inftools.misc.committor_net import load_model
    from inftools.tistools.committor_model import committor_model
    from inftools.tistools.committor_train import committor_train

    committor_model(cols="2-3", hidden="16,8",
                    out=str(tmp_path / "m.pt"))
    out = tmp_path / "out"
    committor_train(
        **two_runs, model=str(tmp_path / "m.pt"), out=str(out),
        epochs=300, patience=30, lr=3e-3, batch=512, device="cpu",
    )
    model, ckpt = load_model(out / "model_trained.pt")
    assert ckpt["trained"] is True
    assert ckpt["meta"]["split_sizes"] == [4800, 600, 600]
    x = np.linspace(0.05, 0.95, 91)
    inp = torch.tensor(np.c_[x, np.zeros_like(x)], dtype=torch.float32)
    with torch.no_grad():
        p = torch.sigmoid(model(inp)).numpy().ravel()
    rmse = math.sqrt(np.mean((p - true_pb(x)) ** 2))
    assert rmse < 0.05, rmse
    scripted = torch.jit.load(str(out / "model_scripted.pt"))
    with torch.no_grad():
        assert torch.allclose(scripted(inp), model(inp))
    split = np.load(out / "split.npz")
    allrows = np.concatenate([split[k] for k in split.files])
    assert sorted(allrows.tolist()) == list(range(6000))
    assert "Calibration" in (out / "report.txt").read_text()
    # an existing, non-empty output folder needs -O
    with pytest.raises(ValueError):
        committor_train(**two_runs, model=str(tmp_path / "m.pt"),
                        out=str(out), epochs=1)


def test_column_one_with_both_directions(tmp_path, two_runs):
    pytest.importorskip("torch")
    from inftools.tistools.committor_model import committor_model
    from inftools.tistools.committor_train import committor_train

    committor_model(cols="1-3", out=str(tmp_path / "m1.pt"))
    with pytest.raises(ValueError):
        committor_train(**two_runs, model=str(tmp_path / "m1.pt"),
                        out=str(tmp_path / "o"))
