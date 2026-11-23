from typing import Annotated

import typer


def committor_train(
    shots: Annotated[
        list[str],
        typer.Option(
            "-shots",
            help="shots.jsonl of a run. Repeat the option to add runs.",
        ),
    ],
    toml: Annotated[
        list[str],
        typer.Option(
            "-toml",
            help="infretis.toml of each run, in the same order as -shots.",
        ),
    ],
    direction: Annotated[
        list[str],
        typer.Option(
            "-dir",
            help="AB for a run whose paths start in state A, BA for a run "
            "whose paths start in state B; one per -shots.",
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "-model",
            help="Model made by `inft committor_model` (not needed with "
            "-build_only).",
        ),
    ] = "",
    out: Annotated[
        str, typer.Option("-out", help="Output folder")
    ] = "committor_train",
    epochs: Annotated[
        int, typer.Option("-epochs", help="Maximum number of epochs")
    ] = 500,
    patience: Annotated[
        int,
        typer.Option(
            "-patience",
            help="Stop after this many epochs without a lower validation "
            "loss",
        ),
    ] = 30,
    lr: Annotated[
        float, typer.Option("-lr", help="Initial learning rate (AdamW)")
    ] = 1e-3,
    wd: Annotated[
        float, typer.Option("-wd", help="Weight decay (AdamW)")
    ] = 1e-4,
    batch: Annotated[
        int, typer.Option("-batch", help="Shots per mini-batch")
    ] = 4096,
    split: Annotated[
        str,
        typer.Option(
            "-split",
            help="Relative sizes of the training, validation and test sets",
        ),
    ] = "8,1,1",
    seed: Annotated[
        int, typer.Option("-seed", help="Seed for the split and training")
    ] = 0,
    device: Annotated[
        str,
        typer.Option(
            "-device", help="auto (GPU if available), cpu or cuda"
        ),
    ] = "auto",
    no_anchors: Annotated[
        bool,
        typer.Option(
            "-no_anchors",
            help="Do not add [0-] shooting points inside the starting state",
        ),
    ] = False,
    build_only: Annotated[
        bool,
        typer.Option(
            "-build_only",
            help="Only collect the shots, write dataset.npz and the report",
        ),
    ] = False,
    overw: Annotated[
        bool, typer.Option("-O", help="Allow writing into a non-empty -out")
    ] = False,
):
    """Train a committor model on the shots recorded by infretis.

    A shot is a shooting point x with the end states of the two trajectory
    halves started from it with new Maxwell-Boltzmann velocities. The
    model q(x) = ln(p_B / p_A) is fitted by minimizing the binomial
    negative log-likelihood of the end states, with every shot weighted
    equally. This is valid because the rule that selects shooting points
    acts on x only; the end states come from new dynamics.

    Only the last jump of each wire-fencing move is used (see
    inftools/misc/committor_shots.py for which records are used and why);
    the report lists the used and dropped records per run. Shots from any
    number of runs can be combined, e.g. an A->B and a B->A run; no
    relative weights are needed. The model inputs are the order.txt
    columns chosen in `inft committor_model`. Requires PyTorch, except
    with -build_only.
    """
    import csv
    import json
    import math
    import os
    from datetime import datetime

    import numpy as np

    from inftools.misc.committor_shots import (
        DIRECTIONS,
        calibration_table,
        collect_run,
        read_run_settings,
        split_rows,
    )

    if not len(shots) == len(toml) == len(direction):
        raise ValueError(
            "Give one -toml and one -dir for every -shots "
            f"({len(shots)} -shots, {len(toml)} -toml, "
            f"{len(direction)} -dir)."
        )
    dirs = [d.upper() for d in direction]
    for d in dirs:
        if d not in DIRECTIONS:
            raise ValueError(f"-dir must be AB or BA, not '{d}'.")
    if not build_only and not model:
        raise ValueError("-model is required unless -build_only is set.")
    if os.path.isdir(out) and os.listdir(out) and not overw:
        raise ValueError(f"Output folder {out} is not empty! Use -O.")

    if not build_only:
        try:
            import torch
        except ImportError as err:
            raise ImportError(
                "inft committor_train requires PyTorch (pip install torch)."
            ) from err
        from inftools.misc.committor_net import (
            feature_names,
            load_model,
            loss_per_shot,
            predict_q,
            save_model,
            set_standardization,
            train_model,
        )

        net, ckpt = load_model(model)
        arch = ckpt["arch"]
        cols = list(arch["cols"])
        if 1 in cols and len(set(dirs)) > 1:
            raise ValueError(
                "The model uses column 1, the order parameter of each run. "
                "It is defined per run and differs between A->B and B->A "
                "runs (often by its sign). Make a model without column 1."
            )

    os.makedirs(out, exist_ok=True)
    report = []

    # collect the shots of every run
    runs, parts = [], []
    for i, (spath, tpath, d) in enumerate(zip(shots, toml, dirs)):
        settings = read_run_settings(tpath)
        data = collect_run(spath, settings, d, anchors=not no_anchors)
        parts.append(data)
        runs.append(
            {
                "shots": os.path.abspath(spath),
                "toml": os.path.abspath(tpath),
                "direction": d,
                "settings": settings,
                "summary": data["summary"],
            }
        )
        kind0 = data["kind"] == 0
        counts = {
            label: int(np.all(data["shots"][kind0] == pair, axis=1).sum())
            for label, pair in (
                ("A,A", [2, 0]),
                ("A,B", [1, 1]),
                ("B,B", [0, 2]),
            )
        }
        runs[-1]["end_states"] = counts
        report.append(f"Run {i}: {spath} ({d})")
        report.append(
            f"  lambda_0 = {settings['lambda_0']}, lambda_N = "
            f"{settings['lambda_N']}, n_jumps = {settings['n_jumps']}"
        )
        for key, val in data["summary"].items():
            report.append(f"  {key}: {val}")
        report.append(
            f"  used: {int(kind0.sum())} shots, "
            f"{int((~kind0).sum())} anchors"
        )
        report.append(
            "  end states of the used shots (A,A / A,B / B,B): "
            f"{counts['A,A']} / {counts['A,B']} / {counts['B,B']}"
        )
        report.append("")

    widths = {p["sp"].shape[1] for p in parts if len(p["sp"])}
    if not widths:
        raise ValueError("No usable shots in the given files.")
    if len(widths) > 1:
        raise ValueError(
            f"The runs store different numbers of order-parameter values "
            f"({sorted(widths)})."
        )
    sp_all = np.concatenate([p["sp"] for p in parts if len(p["sp"])])
    shots_all = np.concatenate([p["shots"] for p in parts])
    kind = np.concatenate([p["kind"] for p in parts])
    ens = np.concatenate([p["ens"] for p in parts])
    run_idx = np.concatenate(
        [np.full(len(p["kind"]), i, dtype=int) for i, p in enumerate(parts)]
    )
    np.savez(
        os.path.join(out, "dataset.npz"),
        sp=sp_all,
        shots=shots_all,
        kind=kind,
        ens=ens,
        run=run_idx,
        runs=json.dumps(runs),
    )
    report.append(
        f"Total: {int((kind == 0).sum())} shots and "
        f"{int((kind == 1).sum())} anchors from {len(parts)} run(s); "
        f"{sp_all.shape[1]} order-parameter values per shooting point "
        f"(order.txt columns 1-{sp_all.shape[1]})."
    )

    def write_report():
        with open(os.path.join(out, "report.txt"), "w") as wfile:
            wfile.write("\n".join(report) + "\n")

    if build_only:
        write_report()
        print("\n".join(report))
        print(f"Dataset written to {os.path.join(out, 'dataset.npz')}.")
        return

    if max(cols) > sp_all.shape[1]:
        raise ValueError(
            f"The model uses column {max(cols)}, but the shots store "
            f"columns 1-{sp_all.shape[1]}."
        )
    x_all = sp_all[:, np.array(cols) - 1]
    fractions = [float(v) for v in split.split(",")]
    i_tr, i_va, i_te = split_rows(len(x_all), fractions, seed)
    if len(i_tr) == 0 or len(i_va) == 0:
        raise ValueError("The training and validation sets must not be empty.")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    X = torch.tensor(x_all, dtype=torch.float32, device=device)
    S = torch.tensor(shots_all, dtype=torch.float32, device=device)

    def rows(idx):
        """Torch index tensor for numpy row indices."""
        return torch.as_tensor(idx, dtype=torch.long, device=device)

    t_tr, t_va = rows(i_tr), rows(i_va)
    net.to(device)
    if ckpt["trained"]:
        report.append(
            "The model was trained before: its standardization is kept and "
            "training continues from its parameters."
        )
    else:
        set_standardization(net, X[t_tr])

    report.append("")
    report.append(
        f"Inputs: order.txt columns {cols}, features "
        f"{', '.join(feature_names(arch))}"
    )
    report.append(
        f"Split (rows): train {len(i_tr)}, validation {len(i_va)}, "
        f"test {len(i_te)}; seed {seed}; device {device}"
    )
    print("\n".join(report))
    n_printed = len(report)

    history, best_epoch = train_model(
        net,
        (X[t_tr], S[t_tr]),
        (X[t_va], S[t_va]),
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        weight_decay=wd,
        patience=patience,
        seed=seed,
    )

    # evaluation
    def constant_loss(idx, f):
        """Loss per end of predicting p_B = f everywhere."""
        f = min(max(f, 1e-12), 1 - 1e-12)
        n_a, n_b = shots_all[idx, 0].sum(), shots_all[idx, 1].sum()
        return -(n_b * math.log(f) + n_a * math.log(1 - f)) / (n_a + n_b)

    f_train = float(shots_all[i_tr, 1].sum() / shots_all[i_tr].sum())
    losses = {}
    report.append("")
    report.append(f"Best epoch {best_epoch} of {len(history)}.")
    report.append(
        "Loss per shot end (ln 2 = 0.6931 for p_B = 1/2 everywhere; "
        f"'constant' predicts the training fraction of B ends, "
        f"{f_train:.4f}, everywhere):"
    )
    for name, idx in (("train", i_tr), ("validation", i_va), ("test", i_te)):
        if len(idx) == 0:
            continue
        losses[name] = loss_per_shot(net, X[rows(idx)], S[rows(idx)])
        report.append(
            f"  {name:10s} model {losses[name]:.5f}   constant "
            f"{constant_loss(idx, f_train):.5f}"
        )
    if len(i_te):
        for r in range(len(parts)):
            idx = i_te[run_idx[i_te] == r]
            if len(idx):
                loss_r = loss_per_shot(net, X[rows(idx)], S[rows(idx)])
                report.append(
                    f"  test, run {r}: {loss_r:.5f} ({len(idx)} rows)"
                )
        idx = i_te[kind[i_te] == 0]
        p = torch.sigmoid(predict_q(net, X[rows(idx)])).cpu().numpy()
        report.append("")
        report.append(
            "Calibration on the test shots (bins of equal size in predicted "
            "p_B):"
        )
        report.append(
            "  mean predicted p_B   fraction of ends in B   std. error   ends"
        )
        for pred, frac, err, n_ends in calibration_table(
            p, shots_all[idx], n_bins=10
        ):
            report.append(
                f"  {pred:18.4g}   {frac:21.4g}   {err:10.2g}   {n_ends:5.0f}"
            )

    # outputs
    net.cpu().eval()
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": "inft committor_train",
        "model_in": os.path.abspath(model),
        "runs": runs,
        "split_sizes": [len(i_tr), len(i_va), len(i_te)],
        "seed": seed,
        "losses": losses,
        "best_epoch": best_epoch,
        "settings": {
            "epochs": epochs,
            "patience": patience,
            "lr": lr,
            "weight_decay": wd,
            "batch": batch,
            "anchors": not no_anchors,
        },
        "torch_version": str(torch.__version__),
    }
    prev_history = list(ckpt["history"]) if ckpt["trained"] else []
    save_model(
        os.path.join(out, "model_trained.pt"),
        net,
        arch,
        trained=True,
        history=prev_history + history,
        meta=meta,
    )
    torch.jit.script(net).save(os.path.join(out, "model_scripted.pt"))
    with open(os.path.join(out, "history.csv"), "w", newline="") as wfile:
        writer = csv.DictWriter(
            wfile, fieldnames=["epoch", "train_loss", "val_loss", "lr"]
        )
        writer.writeheader()
        writer.writerows(history)
    np.savez(
        os.path.join(out, "split.npz"), train=i_tr, validation=i_va, test=i_te
    )
    report.append("")
    report.append(
        "Written: model_trained.pt, model_scripted.pt, history.csv, "
        "dataset.npz, split.npz, report.txt"
    )
    write_report()
    print("\n".join(report[n_printed:]))
