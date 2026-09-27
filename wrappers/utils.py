import torch
import torch.nn.functional as F
import ot
import numpy as np
import math
import warnings
from scipy.optimize import linear_sum_assignment

@torch.no_grad()
def select_low_loss_indices(ema_model, batch, z0, percentile, model='meanflow'):
    x, cond = batch

    b = x.size(0)
    t_temp = torch.linspace(0,1,2, device=x.device).unsqueeze(1)  # [2, 1]
    t_temp = t_temp.expand(-1, b)

    t_temp_exp = t_temp.view(2, b, *[1] * (z0.ndim - 1))  # [2, B, 1, 1, 1]
    zt = (1 - t_temp_exp) * z0.unsqueeze(0) + t_temp_exp * x.unsqueeze(0)  # [2, B, C, H, W]

    # Ground-truth vector field.
    ut_gt = (x - z0).unsqueeze(0).expand_as(zt)  # [2, B, C, H, W]

    # Model prediction.
    zt_flat = zt.flatten(0, 1)        # [2*B, C, H, W]
    t_flat = t_temp.flatten() # [2*B, ]
    cond_exp = cond.unsqueeze(0).expand(2, -1).reshape(-1)
    if model == 'meanflow':
        h = torch.ones_like(t_flat).to(x.device)*int(0)
    else:
        h = torch.ones_like(t_flat).to(x.device)*int(math.log2(128))

    ut_pred = ema_model(zt_flat, t_flat, h, cond_exp).view_as(ut_gt)  # [2, B, C, H, W]

    # Compute loss.
    loss = F.mse_loss(ut_pred, ut_gt, reduction='none').mean(dim=(0, 2, 3, 4))

    k = int(len(loss) * percentile)

    # Operate directly on tensors without converting to NumPy.
    _, indices_low = torch.topk(loss, k, largest=False)

    return indices_low

@torch.no_grad()
def _rank01(v):
    """Scores -> ranks in [0, 1] (used to average scores with different scales)."""
    r = torch.empty_like(v)
    r[torch.argsort(v)] = torch.arange(len(v), device=v.device, dtype=v.dtype)
    return r / max(len(v) - 1, 1)


@torch.no_grad()
def meanflow_pair_scores(ema_model, batch, z0, score='h0'):
    """
    Per-pair MAC score for MeanFlow (lower = better aligned). shape [B].
    Convention of this repo: z_t = (1-t) z0 + t x, t=0 noise, t=1 data,
    model(z, t, h, cond) with h = r - t.

      h0  : original MAC. instantaneous velocity (h=0) at both endpoints
            0.5 * (||u(z0,0,0) - (x-z0)||^2 + ||u(x,1,0) - (x-z0)||^2)
      h1  : one-step (average velocity over the whole interval, h=1)
            ||u(z0,0,1) - (x-z0)||^2  == ||(z0 + u(z0,0,1)) - x||^2
            i.e. distance between the model's 1-NFE sample from z0 and x.
      mix : rank average of h0 and h1
    """
    x, cond = batch
    b = x.size(0)

    s0 = s1 = None
    if score in ('h0', 'mix'):
        # identical to select_low_loss_indices(model='meanflow')
        t_temp = torch.linspace(0, 1, 2, device=x.device).unsqueeze(1).expand(-1, b)
        t_exp = t_temp.reshape(2, b, *[1] * (z0.ndim - 1))
        zt = (1 - t_exp) * z0.unsqueeze(0) + t_exp * x.unsqueeze(0)
        ut_gt = (x - z0).unsqueeze(0).expand_as(zt)
        t_flat = t_temp.flatten()
        h = torch.zeros_like(t_flat)
        cond_exp = cond.unsqueeze(0).expand(2, -1).reshape(-1)
        pred = ema_model(zt.flatten(0, 1), t_flat, h, cond_exp).view_as(ut_gt)
        s0 = F.mse_loss(pred, ut_gt, reduction='none').mean(dim=(0, 2, 3, 4))
    if score in ('h1', 'mix'):
        t0 = torch.zeros(b, device=x.device, dtype=x.dtype)
        h1 = torch.ones(b, device=x.device, dtype=x.dtype)
        u = ema_model(z0, t0, h1, cond)
        s1 = F.mse_loss(u, x - z0, reduction='none').mean(dim=(1, 2, 3))

    if score == 'h0':
        return s0
    if score == 'h1':
        return s1
    if score == 'mix':
        return 0.5 * (_rank01(s0) + _rank01(s1))
    raise ValueError(f"Unknown mac_score: {score}")


@torch.no_grad()
def select_indices_meanflow(ema_model, batch, z0, percentile, score='h0',
                            selection='model', generator=None):
    """MAC selection for MeanFlow. selection='random' is the control
    (same number of pairs, same weight, chosen at random)."""
    b = z0.size(0)
    k = int(b * percentile)
    if selection == 'random':
        perm = torch.randperm(b, generator=generator, device='cpu').to(z0.device)
        return perm[:k]
    if selection != 'model':
        raise ValueError(f"Unknown mac_selection: {selection}")
    s = meanflow_pair_scores(ema_model, batch, z0, score=score)
    _, idx = torch.topk(s, k, largest=False)
    return idx


@torch.no_grad()
def get_ot_pair(ema_model, batch, z0, global_step, model='meanflow',
                ot_method='sinkhorn', reg=None, reg_warmup_steps=20000,
                sinkhorn_sample=False):
    # Compute model-aligned OT coupling between noise z0 and data x1.
    # ot_method: 'exact' or 'sinkhorn'
    # reg: Sinkhorn regularization (None = warmup schedule)
    # sinkhorn_sample: if True, sample pairs from soft plan; if False, extract hard assignment
    images, labels = batch
    x_0 = z0
    x_1 = images
    B = images.shape[0]

    if model == 'meanflow':
        h = torch.ones_like(labels).to(images.device) * int(0)
    else:
        h = torch.ones_like(labels).to(images.device) * int(math.log2(128))

    # 1) Predict velocity at t=0 and t=1
    t0 = torch.zeros(B, dtype=torch.float32, device=images.device)
    v0_pred = ema_model(x_0, t0, h, labels)
    v0_flat = v0_pred.view(B, -1)

    t1 = torch.ones(B, dtype=torch.float32, device=images.device)
    v1_pred = ema_model(x_1, t1, h, labels)
    v1_flat = v1_pred.view(B, -1)

    # 2) Model-aligned cost matrix
    x0_flat = x_0.view(B, -1)
    x1_flat = x_1.view(B, -1)
    target_diff = x1_flat.unsqueeze(0) - x0_flat.unsqueeze(1)  # [B, B, D]

    pred0 = v0_flat.unsqueeze(1).expand(-1, B, -1)
    mse0 = F.mse_loss(pred0, target_diff, reduction='none').mean(dim=2)

    pred1 = v1_flat.unsqueeze(1).expand(-1, B, -1)
    mse1 = F.mse_loss(pred1, target_diff, reduction='none').mean(dim=2)

    M = 0.5 * (mse0 + mse1)  # [B, B]

    # Normalize cost matrix
    eps = 1e-12
    M = M / M.max()
    mean = M.mean()
    std = M.std().clamp_min(eps)
    M = (M - mean) / std
    M_np = M.cpu().numpy()

    a = np.ones((B,)) / B
    b = np.ones((B,)) / B

    # 3) Solve OT
    if ot_method == 'exact':
        # Exact OT (Earth Mover's Distance) via linear programming
        gamma = ot.emd(a, b, M_np)
    elif ot_method == 'sinkhorn':
        # Sinkhorn with optional reg warmup
        if reg is None:
            reg_val = max(0.2, 2.0 - global_step * (2.0 - 0.2) / reg_warmup_steps)
        else:
            reg_val = reg
        gamma = ot.sinkhorn(a, b, M_np, reg=reg_val)
    else:
        raise ValueError(f"Unknown ot_method: {ot_method}. Use 'exact' or 'sinkhorn'.")

    # 4) Extract assignment from transport plan
    if not np.all(np.isfinite(gamma)):
        warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
        gamma = np.ones_like(gamma) / gamma.size

    if ot_method == 'sinkhorn' and sinkhorn_sample:
        # Sinkhorn soft plan: sample pairs according to gamma
        p = gamma.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            gamma.shape[0] * gamma.shape[1], p=p, size=B, replace=True
        )
        row_ind, col_ind = np.divmod(choices, gamma.shape[1])
    else:
        # Hard assignment (exact EMD or sinkhorn with hard matching)
        row_ind, col_ind = linear_sum_assignment(-gamma)

    x0 = x_0[row_ind]
    x1 = images[col_ind]
    y = labels[col_ind]

    return x0, x1, y
