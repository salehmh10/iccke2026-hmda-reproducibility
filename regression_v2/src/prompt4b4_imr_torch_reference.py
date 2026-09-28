"""Tiny autograd reference matching the authors' IMr tutorial equation."""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from torch.distributions import MultivariateNormal as MVN


def main() -> int:
    y_true = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    y_pred = np.array([0.2, 0.8, 2.5], dtype=np.float64)
    means = np.array([0.0, 2.0], dtype=np.float64)
    weights = np.array([0.7, 0.3], dtype=np.float64)
    variances = np.array([1.0, 2.0], dtype=np.float64)
    target = torch.tensor(y_true).reshape(-1, 1)
    pred = torch.tensor(y_pred).reshape(-1, 1).requires_grad_()
    mu = torch.tensor(means).reshape(-1, 1)
    var = torch.tensor(variances).reshape(-1, 1, 1)
    pi = torch.tensor(weights)
    identity = torch.eye(1, dtype=torch.float64)
    noise_var = 1.0
    mse_term = -MVN(target, noise_var * identity).log_prob(pred)
    balancing = MVN(mu, var + noise_var * identity).log_prob(pred.unsqueeze(1)) + pi.log()
    balancing = torch.logsumexp(balancing, dim=1)
    loss = ((mse_term + balancing) * (2.0 * noise_var)).sum()
    loss.backward()
    payload = {"gradient": pred.grad.detach().numpy().reshape(-1).tolist(), "hessian": [1.0] * len(y_true)}
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
