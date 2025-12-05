from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


def fit_scale_rl(
    samples: Iterable[Tuple[float, float]],
    max_iters: int = 1500,
    lr: float = 0.05,
) -> dict | None:
    """
    Fit the ScaleRL sigmoid curve:
        R(C) = R0 + (A - R0) / (1 + (C_mid / C) ** B)

    Args:
        samples: iterable of (compute, reward) pairs.
        max_iters: optimization steps.
        lr: optimizer learning rate.

    Returns:
        dict with fitted parameters {R0, A, C_mid, B, loss} or None if insufficient data.
    """

    data = [(float(c), float(r)) for c, r in samples if c is not None and c > 0]
    if len(data) < 4:
        return None

    compute = torch.tensor([c for c, _ in data], dtype=torch.float64)
    reward = torch.tensor([r for _, r in data], dtype=torch.float64)

    if torch.isnan(compute).any() or torch.isnan(reward).any():
        return None

    min_r = float(torch.min(reward).item())
    max_r = float(torch.max(reward).item())
    if not torch.isfinite(torch.tensor(min_r)) or not torch.isfinite(torch.tensor(max_r)):
        return None

    # Initial guesses
    r0_init = min_r
    delta_init = max(max_r - min_r, 1e-3)
    c_mid_init = float(torch.median(compute).item())
    if c_mid_init <= 0:
        c_mid_init = float(torch.mean(compute).item())
    c_mid_init = max(c_mid_init, 1.0)
    b_init = 2.0

    r0 = torch.nn.Parameter(torch.tensor(r0_init, dtype=torch.float64))
    delta_param = torch.nn.Parameter(torch.tensor(delta_init, dtype=torch.float64))
    cmid_param = torch.nn.Parameter(torch.tensor(c_mid_init, dtype=torch.float64))
    b_param = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float64))

    optimizer = torch.optim.Adam([r0, delta_param, cmid_param, b_param], lr=lr)

    for _ in range(max_iters):
        optimizer.zero_grad()

        delta = F.softplus(delta_param) + 1e-6
        c_mid = F.softplus(cmid_param) + 1e-6
        b = F.softplus(b_param) + 1e-3

        pred = r0 + delta / (1.0 + (c_mid / compute) ** b)
        loss = torch.mean((pred - reward) ** 2)

        if torch.isnan(loss):
            return None

        loss.backward()
        optimizer.step()

    delta = F.softplus(delta_param).detach().item() + 1e-6
    c_mid = F.softplus(cmid_param).detach().item() + 1e-6
    b = F.softplus(b_param).detach().item() + 1e-3
    r0_val = r0.detach().item()
    a_val = r0_val + delta

    final_pred = r0_val + delta / (1.0 + (c_mid / compute) ** b)
    final_loss = torch.mean((final_pred - reward) ** 2).item()

    return {
        "R0": float(r0_val),
        "A": float(a_val),
        "C_mid": float(c_mid),
        "B": float(b),
        "loss": float(final_loss),
        "data_points": len(data),
        "compute_min": float(torch.min(compute).item()),
        "compute_max": float(torch.max(compute).item()),
        "reward_min": float(min_r),
        "reward_max": float(max_r),
    }


