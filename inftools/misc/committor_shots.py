"""Read infretis shots.jsonl files for `inft committor_train`.

A shot is a shooting point together with the end states of the two
trajectory halves started from it with new Maxwell-Boltzmann velocities.
shots.jsonl is written by infretis when ``tis_set.record_shots`` is set.

Only the last jump of each wire-fencing move is used. For it, the end
states of both halves are known:

* both halves reached the cap (the boundary of the far state) during the
  jump itself;
* or the move was accepted and the matching ``wf_extended`` record gives
  both end states of the extended path;
* or the move was rejected with status BWI, which happens when both ends
  of the extended path lie in the far state.

The last jump is chosen by its index, before the move is run, so keeping
only last jumps does not select on the outcome. Intermediate jumps are
dropped, including the extended ones: an intermediate jump is extended
only when the last jump failed, which depends on the last jump's outcome.
Moves whose extension exceeded the maximum path length (BTX, FTX) have
unknown end states and are dropped; the counts are reported.

Shooting points of [0-] shots (ensemble 000) that lie inside the run's
starting state are added as anchors: their committor is fixed by the
state definition.

The counts in ``shots`` are physical: column 0 counts ends in A, column 1
ends in B. An A->B run starts in A, a B->A run starts in B.
"""

import json

import numpy as np
import tomli

DIRECTIONS = ("AB", "BA")
# largest difference allowed between the shooting-point values stored in a
# wf_jump record and in its wf_extended record (they differ by rounding)
SP_TOL = 1e-4
# relative tolerance when comparing the cap with the last interface
CAP_TOL = 1e-9


def read_run_settings(toml_path) -> dict:
    """State boundaries and wire-fencing settings of one infretis run."""
    with open(toml_path, "rb") as rfile:
        config = tomli.load(rfile)
    intf = config["simulation"]["interfaces"]
    tis_set = config["simulation"].get("tis_set", {})
    cap = tis_set.get("interface_cap")
    return {
        "lambda_0": float(intf[0]),
        "lambda_N": float(intf[-1]),
        "n_jumps": int(tis_set.get("n_jumps", 2)),
        "interface_cap": None if cap is None else float(cap),
    }


def _side(op: float, left: float, right: float) -> str | None:
    """'L' if op <= left, 'R' if op >= right (as in infretis), else None."""
    if op <= left:
        return "L"
    if op >= right:
        return "R"
    return None


def _same_value(a: float, b: float) -> bool:
    return abs(a - b) <= CAP_TOL * max(1.0, abs(a), abs(b))


def collect_run(
    shots_path, settings: dict, direction: str, anchors: bool = True
) -> dict:
    """Collect the usable shots of one run.

    Args:
        shots_path: the run's shots.jsonl.
        settings: output of read_run_settings for the run's toml.
        direction: 'AB' for a run that starts in A, 'BA' for one that
            starts in B.
        anchors: also return the [0-] shooting points inside the starting
            state.

    Returns:
        Dict with ``sp`` (shooting-point order parameters, one row per
        shot), ``shots`` (physical counts [n_A, n_B]), ``kind`` (0 for a
        shot, 1 for an anchor), ``ens`` (ensemble number) and ``summary``
        (counts of used and dropped records, by reason).
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"Direction must be AB or BA, not '{direction}'.")
    lam0, lamN = settings["lambda_0"], settings["lambda_N"]
    last = settings["n_jumps"] - 1
    cap = settings["interface_cap"]
    if cap is not None and not _same_value(cap, lamN):
        raise ValueError(
            f"interface_cap = {cap} differs from the last interface {lamN}:"
            " halves that reach the cap are then not in a state."
        )
    # index of the physical state (0: A, 1: B) of the start and far state
    start_idx, far_idx = (0, 1) if direction == "AB" else (1, 0)

    jumps, extended, sh_records = [], {}, []
    n_other = 0
    with open(shots_path) as rfile:
        for line in rfile:
            if not line.strip():
                continue
            rec = json.loads(line)
            for key in ("cstep", "mc_status"):
                if key not in rec:
                    raise ValueError(
                        f"{shots_path}: records lack '{key}'; this file "
                        "was written by an infretis version without it."
                    )
            tag = rec["tag"][0]
            if tag == "wf_jump":
                jumps.append(rec)
            elif tag == "wf_extended":
                key = (rec["ens"], rec["cstep"], rec["jump"])
                extended.setdefault(key, []).append(rec)
            elif tag == "sh":
                sh_records.append(rec)
            else:
                n_other += 1

    s = {
        "wf_jump records": len(jumps),
        "intermediate jumps (dropped)": 0,
        "last jumps": 0,
        "  both halves reached the cap": 0,
        "  extended, end states from wf_extended": 0,
        "  status BWI, both ends in the far state": 0,
        "  extension exceeded maximum length (dropped)": 0,
        "  forward half of the jump cut off (dropped)": 0,
        "  wf_extended shooting point differs (dropped)": 0,
        "  extended path end outside the states (dropped)": 0,
        "  end states unknown (dropped)": 0,
        "wf_extended records": sum(len(v) for v in extended.values()),
        "  of intermediate jumps (dropped)": 0,
        "  without matching last jump (dropped)": 0,
        "sh records": len(sh_records),
        "  [0-] shooting points inside the starting state": 0,
        "  [0-] shooting points outside the starting state (dropped)": 0,
        "  [0-] shooting points, anchors switched off (dropped)": 0,
        "  other ensembles (dropped)": 0,
        "records of other types (dropped)": n_other,
    }
    unknown_status: dict[str, int] = {}
    used_ext: set[int] = set()
    rows_sp, rows_shots, rows_kind, rows_ens = [], [], [], []

    def add(sp, ends, kind, ens):
        counts = [0, 0]
        for end in ends:
            counts[start_idx if end == "L" else far_idx] += 1
        rows_sp.append(sp)
        rows_shots.append(counts)
        rows_kind.append(kind)
        rows_ens.append(int(ens))

    for rec in jumps:
        if rec["jump"] != last:
            s["intermediate jumps (dropped)"] += 1
            continue
        s["last jumps"] += 1
        left, right = rec["interfaces"][0], rec["interfaces"][2]
        if not _same_value(right, lamN):
            raise ValueError(
                f"{shots_path}: a jump has cap {right}, but the last "
                f"interface is {lamN}; halves that reach the cap are then "
                "not in a state."
            )
        if not rec["fwd_ok"]:
            s["  forward half of the jump cut off (dropped)"] += 1
            continue
        bwd = _side(rec["bwd_end"][0], left, right)
        fwd = _side(rec["fwd_end"][0], left, right)
        if bwd is None or fwd is None:
            raise ValueError(
                f"{shots_path}: a jump half ends between its interfaces "
                f"({rec['bwd_end'][0]}, {rec['fwd_end'][0]}; "
                f"interfaces {left}, {right})."
            )
        if bwd == fwd == "R":
            s["  both halves reached the cap"] += 1
            add(rec["sp_order"], ("R", "R"), 0, rec["ens"])
            continue
        candidates = extended.get((rec["ens"], rec["cstep"], rec["jump"]), [])
        sp = np.asarray(rec["sp_order"])
        match = [
            e
            for e in candidates
            if np.max(np.abs(np.asarray(e["sp_order"]) - sp)) <= SP_TOL
        ]
        if match:
            ext = match[0]
            used_ext.add(id(ext))
            ends = (
                _side(ext["bwd_end"][0], lam0, lamN),
                _side(ext["fwd_end"][0], lam0, lamN),
            )
            if None in ends:
                s["  extended path end outside the states (dropped)"] += 1
                continue
            s["  extended, end states from wf_extended"] += 1
            add(rec["sp_order"], ends, 0, rec["ens"])
        elif candidates:
            s["  wf_extended shooting point differs (dropped)"] += 1
        elif rec["mc_status"] == "BWI":
            s["  status BWI, both ends in the far state"] += 1
            add(rec["sp_order"], ("R", "R"), 0, rec["ens"])
        elif rec["mc_status"] in ("BTX", "FTX"):
            s["  extension exceeded maximum length (dropped)"] += 1
        else:
            s["  end states unknown (dropped)"] += 1
            status = rec["mc_status"]
            unknown_status[status] = unknown_status.get(status, 0) + 1

    for key, recs in extended.items():
        for ext in recs:
            if id(ext) in used_ext:
                continue
            if ext["jump"] != last:
                s["  of intermediate jumps (dropped)"] += 1
            else:
                s["  without matching last jump (dropped)"] += 1

    outside = "  [0-] shooting points outside the starting state (dropped)"
    for rec in sh_records:
        if rec["ens"] != "000":
            s["  other ensembles (dropped)"] += 1
        elif not anchors:
            s["  [0-] shooting points, anchors switched off (dropped)"] += 1
        elif rec["sp_order"][0] > lam0:
            s[outside] += 1
        else:
            s["  [0-] shooting points inside the starting state"] += 1
            add(rec["sp_order"], ("L", "L"), 1, rec["ens"])

    if unknown_status:
        s["  end states unknown, by status"] = unknown_status
    width = len(rows_sp[0]) if rows_sp else 0
    return {
        "sp": np.array(rows_sp, dtype=float).reshape(len(rows_sp), width),
        "shots": np.array(rows_shots, dtype=float).reshape(-1, 2),
        "kind": np.array(rows_kind, dtype=int),
        "ens": np.array(rows_ens, dtype=int),
        "summary": s,
    }


def split_rows(n: int, fractions: list[float], seed: int) -> list:
    """Split row indices 0..n-1 at random into train, validation, test.

    Args:
        n: number of rows.
        fractions: three non-negative numbers, normalized to sum to 1.
        seed: seed of the random permutation.
    """
    f = np.asarray(fractions, dtype=float)
    if f.shape != (3,) or np.any(f < 0) or f.sum() <= 0:
        raise ValueError(f"Invalid split {fractions}.")
    f = f / f.sum()
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(f[0] * n))
    n_val = int(round(f[1] * n))
    return [
        np.sort(perm[:n_train]),
        np.sort(perm[n_train : n_train + n_val]),
        np.sort(perm[n_train + n_val :]),
    ]


def calibration_table(p: np.ndarray, shots: np.ndarray, n_bins: int = 10):
    """Predicted against observed fraction of B ends, in bins of predicted p.

    The rows are sorted by the predicted p_B and cut into n_bins bins with
    equal numbers of rows.

    Returns:
        List of (mean predicted p_B, observed fraction of ends in B,
        standard error of that fraction, number of ends) per bin.
    """
    order = np.argsort(p)
    rows = []
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        ends = shots[idx].sum()
        frac = shots[idx, 1].sum() / ends
        err = np.sqrt(frac * (1 - frac) / ends)
        rows.append((float(p[idx].mean()), float(frac), float(err), ends))
    return rows
