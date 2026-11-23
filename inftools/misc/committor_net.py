"""Committor network in plain PyTorch, compatible with aimmd.

The network has the layout of aimmd's ``ModuleStack`` holding one
``FFNet``: hidden linear layers, each followed by an activation and
optional dropout, then a last linear layer ``log_predictor`` that returns
q = ln(p_B / p_A). The parameter names match aimmd's, so trained weights
can be loaded into aimmd (see :func:`aimmd_state_dict`). This is an
independent implementation; aimmd itself is not imported, because it
requires openpathsampling.

In front of the network sits :class:`InputTransform`. It replaces angles
that wrap around by their sine and cosine and standardizes the features
with a stored mean and standard deviation. A saved model therefore takes
raw order.txt columns as input.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "inft_committor_v1"

# Only activations without parameters, so that the parameter names (and
# hence the state dict) are the same as aimmd's.
ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "softplus": nn.Softplus,
}


class InputTransform(nn.Module):
    """Map raw descriptor columns to standardized network inputs.

    Non-periodic columns are passed on unchanged. Each periodic column (an
    angle in radians) is replaced by its sine and cosine. The features are
    ordered as: non-periodic columns, then the sines, then the cosines,
    each in input order. They are then standardized with the buffers
    ``mean`` and ``std``, which are 0 and 1 until training sets them.
    """

    def __init__(self, n_cols: int, periodic_idx: list[int]):
        super().__init__()
        per = sorted(periodic_idx)
        lin = [i for i in range(n_cols) if i not in per]
        self.n_cols = n_cols
        self.n_out = len(lin) + 2 * len(per)
        self.register_buffer("lin_idx", torch.tensor(lin, dtype=torch.long))
        self.register_buffer("per_idx", torch.tensor(per, dtype=torch.long))
        self.register_buffer("mean", torch.zeros(self.n_out))
        self.register_buffer("std", torch.ones(self.n_out))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the features before standardization."""
        per = x[:, self.per_idx]
        return torch.cat(
            [x[:, self.lin_idx], torch.sin(per), torch.cos(per)], dim=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.features(x) - self.mean) / self.std


class FFNet(nn.Module):
    """Hidden layers with activation and dropout, named as in aimmd."""

    def __init__(
        self,
        n_in: int,
        n_hidden: list[int],
        activation: str = "elu",
        dropout: dict[str, float] | None = None,
    ):
        super().__init__()
        dropout = dropout or {}
        n_units = [n_in] + list(n_hidden)
        self.n_out = n_units[-1]
        self.activations = nn.ModuleList(
            [ACTIVATIONS[activation]() for _ in n_hidden]
        )
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(n_units[i], n_units[i + 1])
                for i in range(len(n_hidden))
            ]
        )
        self.dropout_layers = nn.ModuleList(
            [
                nn.Dropout(dropout[str(i)])
                if str(i) in dropout
                else nn.Identity()
                for i in range(len(n_hidden))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lay, act, drop in zip(
            self.hidden_layers, self.activations, self.dropout_layers
        ):
            x = drop(act(lay(x)))
        return x


class CommittorNet(nn.Module):
    """Input transform, hidden layers and the log-predictor layer."""

    def __init__(
        self,
        n_cols: int,
        periodic_idx: list[int],
        n_hidden: list[int],
        activation: str = "elu",
        dropout: dict[str, float] | None = None,
    ):
        super().__init__()
        self.transform = InputTransform(n_cols, periodic_idx)
        ffnet = FFNet(self.transform.n_out, n_hidden, activation, dropout)
        self.module_list = nn.ModuleList([ffnet])
        self.log_predictor = nn.Linear(ffnet.n_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return q = ln(p_B / p_A), shape (n, 1), for raw columns x."""
        x = self.transform(x)
        for module in self.module_list:
            x = module(x)
        return self.log_predictor(x)


def check_arch(arch: dict) -> None:
    """Raise a ValueError if the architecture dict is not valid."""
    cols, periodic = list(arch["cols"]), list(arch["periodic"])
    if not cols:
        raise ValueError("No input columns given.")
    if min(cols) < 1:
        raise ValueError(
            "Column 0 of order.txt is the time step and cannot be an input."
        )
    if len(set(cols)) != len(cols):
        raise ValueError(f"Input columns {cols} contain duplicates.")
    missing = [c for c in periodic if c not in cols]
    if missing:
        raise ValueError(f"Periodic columns {missing} are not input columns.")
    if not arch["hidden"] or min(arch["hidden"]) < 1:
        raise ValueError(f"Invalid hidden layers {arch['hidden']}.")
    if arch["activation"] not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{arch['activation']}'; choose from "
            f"{', '.join(ACTIVATIONS)}."
        )
    for key, p in arch["dropout"].items():
        if not 0 <= int(key) < len(arch["hidden"]) or not 0.0 <= p < 1.0:
            raise ValueError(f"Invalid dropout {key}:{p}.")


def build_model(arch: dict) -> CommittorNet:
    """Create an untrained network from an architecture dict.

    Args:
        arch: dict with ``cols`` (order.txt columns used as inputs),
            ``periodic`` (the subset holding angles that wrap around),
            ``hidden`` (units per hidden layer), ``activation`` (a key of
            ACTIVATIONS) and ``dropout`` (aimmd format, e.g. {"0": 0.1}).
    """
    check_arch(arch)
    cols = list(arch["cols"])
    periodic_idx = [cols.index(c) for c in arch["periodic"]]
    return CommittorNet(
        len(cols),
        periodic_idx,
        list(arch["hidden"]),
        arch["activation"],
        dict(arch["dropout"]),
    )


def feature_names(arch: dict) -> list[str]:
    """Names of the network inputs, in the order InputTransform makes them."""
    cols, per = list(arch["cols"]), set(arch["periodic"])
    return (
        [f"c{c}" for c in cols if c not in per]
        + [f"sin(c{c})" for c in cols if c in per]
        + [f"cos(c{c})" for c in cols if c in per]
    )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binomial_loss(
    q: torch.Tensor, shots: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Weighted negative log-likelihood of the shot results.

    sum_i w_i [n_A,i ln(1 + e^q_i) + n_B,i ln(1 + e^-q_i)], with
    q = ln(p_B / p_A) and shots[:, 0] = n_A, shots[:, 1] = n_B. This is
    aimmd's binomial loss, written with softplus, which stays finite for
    any finite q (log(1 + e^q) overflows in float32 for q > 88). The sum
    is taken in float64 so that the summed value keeps 15 significant
    digits; the gradient of each term does not pass through this sum. The
    counts may be fractional. The loss is not normalized.
    """
    q = q.reshape(-1)
    terms = shots[:, 0] * F.softplus(q) + shots[:, 1] * F.softplus(-q)
    return (weights.double() * terms.double()).sum()


def save_model(
    path,
    model: CommittorNet,
    arch: dict,
    trained: bool = False,
    history: list | None = None,
    meta: dict | None = None,
) -> None:
    """Save the model as a dict; it loads with weights_only=True."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "format": FORMAT,
            "arch": arch,
            "state_dict": state,
            "trained": trained,
            "history": history or [],
            "meta": meta or {},
        },
        path,
    )


def load_model(path, device: str = "cpu") -> tuple[CommittorNet, dict]:
    """Load a model saved by :func:`save_model`.

    Returns:
        The network (in eval mode, on ``device``) and the saved dict.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} committor model.")
    model = build_model(ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def set_standardization(model: CommittorNet, x: torch.Tensor) -> None:
    """Set the input standardization to the mean and std of x's features.

    Every feature (after the sine/cosine step) gets mean 0 and standard
    deviation 1 over the rows of x. A feature with standard deviation
    below 1e-12 is only shifted, not scaled.
    """
    with torch.no_grad():
        feats = model.transform.features(x)
        mean = feats.mean(dim=0)
        std = feats.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-12, torch.ones_like(std), std)
        model.transform.mean.copy_(mean)
        model.transform.std.copy_(std)


def predict_q(model: CommittorNet, x: torch.Tensor, batch: int = 65536):
    """q for every row of x, in eval mode and without gradients."""
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [model(x[i : i + batch]) for i in range(0, len(x), batch)]
        ).reshape(-1)


def loss_per_shot(
    model: CommittorNet, x: torch.Tensor, shots: torch.Tensor
) -> float:
    """Binomial loss divided by the number of shot ends, without gradients.

    ln 2 = 0.693 corresponds to predicting p_B = 1/2 for every end.
    """
    q = predict_q(model, x)
    total = binomial_loss(q, shots, torch.ones_like(q))
    return float(total / shots.sum().double())


def train_model(
    model: CommittorNet,
    train: tuple,
    val: tuple,
    epochs: int = 500,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    plateau_patience: int = 10,
    plateau_factor: float = 0.5,
    min_lr: float = 1e-6,
    min_delta: float = 1e-5,
    seed: int = 0,
    log=print,
) -> tuple[list[dict], int]:
    """Train on shot data with all weights equal to 1.

    Each optimizer step minimizes the binomial loss of a random mini-batch
    divided by its number of shot ends. After every epoch the loss per
    shot end on the training and validation rows is recorded. The
    learning rate is multiplied by plateau_factor when the validation
    loss has not decreased by min_delta for plateau_patience epochs, and
    training stops when it has not done so for patience epochs. The state
    with the lowest validation loss is restored at the end. A non-finite
    batch loss restores that state and halves the learning rate.

    Args:
        model: the network; its parameters and buffers must be on the
            same device as the data.
        train, val: tuples (x, shots) of tensors, x with the raw input
            columns and shots with the counts [n_A, n_B].

    Returns:
        The history (one dict per epoch) and the best epoch.
    """
    x_tr, s_tr = train
    x_va, s_va = val
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=plateau_factor,
        patience=plateau_patience,
        threshold=min_delta,
        threshold_mode="abs",
        min_lr=min_lr,
    )
    ones = torch.ones(len(x_tr), device=x_tr.device)
    best_loss, best_epoch = loss_per_shot(model, x_va, s_va), 0
    best_state = copy.deepcopy(model.state_dict())
    history, n_bad, n_restarts = [], 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr), generator=gen).to(x_tr.device)
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            q = model(x_tr[idx])
            loss = binomial_loss(q, s_tr[idx], ones[idx])
            loss = loss / s_tr[idx].sum().double()
            if not torch.isfinite(loss):
                n_restarts += 1
                if n_restarts > 5:
                    raise RuntimeError("Loss not finite after 5 restarts.")
                model.load_state_dict(best_state)
                for group in opt.param_groups:
                    group["lr"] *= 0.5
                log(f"Non-finite loss in epoch {epoch}: restored the best "
                    "state and halved the learning rate.")
                break
            loss.backward()
            opt.step()
        tr = loss_per_shot(model, x_tr, s_tr)
        va = loss_per_shot(model, x_va, s_va)
        sched.step(va)
        lr_now = opt.param_groups[0]["lr"]
        history.append(
            {"epoch": epoch, "train_loss": tr, "val_loss": va, "lr": lr_now}
        )
        if va < best_loss - min_delta:
            best_loss, best_epoch, n_bad = va, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            n_bad += 1
        if epoch == 1 or epoch % 10 == 0 or n_bad >= patience:
            log(f"epoch {epoch:5d}  train {tr:.5f}  val {va:.5f}  "
                f"lr {lr_now:.2e}  best epoch {best_epoch}")
        if n_bad >= patience:
            log(f"Stopped: no improvement for {patience} epochs.")
            break
    model.load_state_dict(best_state)
    model.eval()
    return history, best_epoch


def aimmd_state_dict(model: CommittorNet) -> dict:
    """State dict without the input transform, with aimmd's key names.

    It loads into ``aimmd.pytorch.networks.ModuleStack(n_out=1,
    modules=[FFNet(n_in, n_hidden, activation, dropout)])``. That network
    expects the transformed (standardized) inputs, i.e. the output of
    ``model.transform``.
    """
    return {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith("transform.")
    }
