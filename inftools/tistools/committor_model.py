from typing import Annotated

import typer


def committor_model(
    cols: Annotated[
        str,
        typer.Option(
            "-cols",
            help="order.txt columns used as inputs, e.g. '2-13' or "
            "'2,3,5-7'. Column 0 is the time step and cannot be used.",
        ),
    ],
    out: Annotated[
        str, typer.Option("-out", help="Output model file (.pt)")
    ] = "committor_model.pt",
    periodic: Annotated[
        str,
        typer.Option(
            "-periodic",
            help="Input columns holding angles in radians that wrap "
            "around, e.g. '8,9'. Each is replaced by its sine and cosine.",
        ),
    ] = "",
    hidden: Annotated[
        str,
        typer.Option(
            "-hidden", help="Units per hidden layer, e.g. '64,32,16'"
        ),
    ] = "64,32,16",
    activation: Annotated[
        str,
        typer.Option(
            "-activation",
            help="Activation after each hidden layer: elu, relu, "
            "leakyrelu, tanh, silu, gelu or softplus",
        ),
    ] = "elu",
    dropout: Annotated[
        str,
        typer.Option(
            "-dropout",
            help="Dropout after hidden layers as layer:p pairs, e.g. "
            "'0:0.1,1:0.1'. Layers count from 0. Default: none.",
        ),
    ] = "",
    seed: Annotated[
        int, typer.Option("-seed", help="Seed for the initial weights")
    ] = 0,
    overw: Annotated[
        bool, typer.Option("-O", help="Force overwriting of files")
    ] = False,
):
    """Create an untrained committor model for `inft committor_train`.

    The model predicts q = ln(p_B / p_A) from order.txt columns. It is a
    feed-forward network with the layout and parameter names of aimmd's
    ModuleStack and FFNet, written in plain PyTorch so that it runs without
    openpathsampling. Periodic columns are replaced by their sine and
    cosine, and all inputs are standardized; `inft committor_train` sets
    the standardization from the reweighted training data.

    Requires PyTorch.
    """
    import os
    from datetime import datetime

    from inftools.misc.committor_helper import parse_dropout, parse_int_list

    try:
        import torch
    except ImportError as err:
        raise ImportError(
            "inft committor_model requires PyTorch (pip install torch)."
        ) from err
    from inftools.misc.committor_net import (
        build_model,
        count_parameters,
        feature_names,
        save_model,
    )

    if not overw and os.path.exists(out):
        raise ValueError(f"Out file {out} exists! Use -O to overwrite.")

    n_hidden = parse_int_list(hidden)
    arch = {
        "cols": parse_int_list(cols),
        "periodic": parse_int_list(periodic),
        "hidden": n_hidden,
        "activation": activation.lower(),
        "dropout": parse_dropout(dropout, len(n_hidden)),
    }

    torch.manual_seed(seed)
    model = build_model(arch)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        # str(): TorchVersion is not allowed by torch.load(weights_only=True)
        "torch_version": str(torch.__version__),
    }
    save_model(out, model, arch, meta=meta)

    names = feature_names(arch)
    units = " -> ".join(str(n) for n in [len(names)] + n_hidden + [1])
    drop = (
        ", ".join(f"{k}:{v}" for k, v in arch["dropout"].items()) or "none"
    )
    print(f"Committor model saved to {out}.")
    print(f"  inputs:     order.txt columns {arch['cols']}")
    if arch["periodic"]:
        print(f"  periodic:   {arch['periodic']} (as sine and cosine)")
    print(f"  features:   {', '.join(names)}")
    print(f"  network:    {units}, activation {arch['activation']}")
    print(f"  dropout:    {drop}")
    print(f"  parameters: {count_parameters(model)} (seed {seed})")
    print(
        "The input standardization is set by `inft committor_train`; "
        "the model is untrained."
    )
