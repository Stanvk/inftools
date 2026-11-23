"""Helpers for the committor commands that do not need PyTorch."""


def parse_int_list(text: str) -> list[int]:
    """Parse integers and ranges such as '2-5,8,10-11' into a list.

    Ranges include both ends. Order is kept and duplicates are rejected.

    Args:
        text: comma-separated integers and ``start-stop`` ranges.

    Returns:
        The integers in the given order.
    """
    out: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part[1:]:
            # split at the range dash, not at a leading minus sign
            i = part.index("-", 1)
            start, stop = int(part[:i]), int(part[i + 1 :])
            if stop < start:
                raise ValueError(f"Range '{part}' runs backwards.")
            out += list(range(start, stop + 1))
        else:
            out.append(int(part))
    if len(set(out)) != len(out):
        raise ValueError(f"'{text}' contains duplicate values.")
    return out


def parse_dropout(text: str, n_layers: int) -> dict[str, float]:
    """Parse dropout settings such as '0:0.1,2:0.2'.

    Args:
        text: comma-separated ``layer:probability`` pairs. Layers count
            from 0; dropout is applied after that hidden layer.
        n_layers: number of hidden layers, to check the layer indices.

    Returns:
        Dict in aimmd's format, mapping the layer index (as a string) to
        the dropout probability.
    """
    out: dict[str, float] = {}
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Dropout '{part}' is not of the form layer:p.")
        layer, prob = part.split(":")
        idx, p = int(layer), float(prob)
        if not 0 <= idx < n_layers:
            raise ValueError(
                f"Dropout layer {idx} does not exist; there are {n_layers} "
                f"hidden layers (0 to {n_layers - 1})."
            )
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability {p} is not in [0, 1).")
        if str(idx) in out:
            raise ValueError(f"Dropout for layer {idx} is given twice.")
        out[str(idx)] = p
    return out
