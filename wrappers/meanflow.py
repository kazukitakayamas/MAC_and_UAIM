import copy
import torch
import torch.nn.functional as F
from .utils import select_low_loss_indices, get_ot_pair


class MACWrapper:
    # ============================================================
    # [ORIGINAL]
    # def __init__(self, model, vae=None, add_weight=0.1, ln=True, model_type='full'):
    #
    # [KIM CHANGE 1]
    # Kim progressive L_u weighting用に以下を追加:
    #   kim_mode   : 'none' / 'progressive'
    #   kim_k      : s = 1 - (i / T)^k の k
    #   kim_lambda : beta(Δt, s) の lambda
    # ============================================================
    def __init__(
        self,
        model,
        vae=None,
        add_weight=0.1,
        ln=True,
        model_type='full',
        kim_mode='none',
        kim_k=1.0,
        kim_lambda=None,
        norm_p=0.75
    ):
        self.model = model
        self.vae = vae
        self.ema_model = copy.deepcopy(model).eval()
        self.ln = ln
        self.BOOTSTRAP_EVERY = 8
        self.DENOISE_TIMESTEPS = 128
        self.CLASS_DROPOUT_PROB = 0.1
        self.NUM_CLASSES = 10
        self.decay = 0.999
        self.add_weight = add_weight
        self.time_mu = 0.4
        self.time_sigma = 1.0
        self.ratio_r_not_equal_t = 0.25

        self.norm_p = float(norm_p)
        self.norm_eps = 1e-3
        self.model_type = model_type

        # ========================================================
        # [KIM CHANGE 2] Kim progressive weightingの設定を保持
        # ========================================================
        self.kim_mode = kim_mode
        self.kim_k = float(kim_k)
        self.kim_lambda = kim_lambda

        if self.kim_mode not in ('none', 'progressive'):
            raise ValueError(
                f"Unknown kim_mode: {self.kim_mode}. "
                "Choose from {'none', 'progressive'}."
            )

        if self.kim_mode == 'progressive':
            if self.kim_k <= 0:
                raise ValueError('kim_k must be > 0 for progressive weighting.')

            # kim_lambdaを指定しない場合は、現在のMAC MeanFlowの
            # t/r samplerに合わせて
            #   lambda = 1 / E[1 - Δt]
            # をMonte Carlo推定する。
            #
            # 重要:
            # ここでの期待値は r != t の L_u サンプルのgap分布に対して取る。
            # r=tを75%混ぜた「後」の分布ではない。
            if self.kim_lambda is None:
                self.kim_lambda = self._estimate_kim_lambda()
            else:
                self.kim_lambda = float(self.kim_lambda)

            if self.kim_lambda <= 0:
                raise ValueError('kim_lambda must be > 0 for progressive weighting.')

            print(
                f"[Kim] progressive L_u weighting enabled: "
                f"k={self.kim_k:.4f}, lambda={self.kim_lambda:.6f}"
            )

    @torch.no_grad()
    def update_ema(self):
        for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def logit_normal_timestep_sample(
        self,
        P_mean: float,
        P_std: float,
        num_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        rnd_normal = torch.randn((num_samples,), device=device)
        time = torch.sigmoid(rnd_normal * P_std + P_mean)
        time = torch.clip(time, min=0.0, max=1.0)
        return time

    def sample_time_steps(self, time_sampler, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler."""
        # Step1: Sample two time points
        if time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
            # Step2: sort the two sampled times
            sorted_samples, _ = torch.sort(time_samples, dim=1)
            r, t = sorted_samples[:, 0], sorted_samples[:, 1]

        elif time_sampler == "logit_normal":
            # [ORIGINAL MAC behavior]
            # まず +mu と -mu のLogit-Normalから1点ずつサンプルする。
            time_a = self.logit_normal_timestep_sample(
                self.time_mu,
                self.time_sigma,
                batch_size,
                device=device,
            )
            time_b = self.logit_normal_timestep_sample(
                -self.time_mu,
                self.time_sigma,
                batch_size,
                device=device,
            )

            # このwrapperでは、modelへの入力は
            #   t = smaller/current time
            #   r = larger/next time
            # として使っているため、t <= r にする。
            sorted_samples, _ = torch.sort(
                torch.stack([time_a, time_b], dim=1),
                dim=1,
            )
            t, r = sorted_samples[:, 0], sorted_samples[:, 1]

        else:
            raise ValueError(f"Unknown time sampler: {time_sampler}")

        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal

        # このwrapperでは t <= r なので、equal sampleでは t=r とする。
        t = torch.where(equal_mask, r, t)

        return r, t

    # ============================================================
    # [KIM CHANGE 3]
    # 現在のMAC MeanFlow time samplerに対して
    #   lambda = 1 / E[1 - Δt]
    # を推定する。
    #
    # Progressive weightingはL_u (r != t)だけに適用するため、
    # r=tを混ぜる前のgap分布に対して推定する。
    # ============================================================
    def _estimate_kim_lambda(self, n_samples=200000, seed=12345):
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)

        time_a = torch.randn(n_samples, generator=g)
        time_b = torch.randn(n_samples, generator=g)

        time_a = torch.sigmoid(time_a * self.time_sigma + self.time_mu)
        time_b = torch.sigmoid(time_b * self.time_sigma - self.time_mu)

        # 現在のwrapperでは t <= r なので、Kim論文の temporal gap に
        # 対応する量は Δt = r - t = |time_a - time_b|。
        delta_t = torch.abs(time_a - time_b)

        expectation = (1.0 - delta_t).mean()

        if expectation <= 0:
            raise RuntimeError(
                'Failed to estimate kim_lambda because E[1 - delta_t] <= 0.'
            )

        kim_lambda = 1.0 / expectation
        return float(kim_lambda)

    # ============================================================
    # [KIM CHANGE 4]
    # Kim et al. の progressive L_u weighting:
    #
    #   s = 1 - (i / T)^k
    #   beta(Δt, s) = 1 - s + lambda * s * (1 - Δt)
    #
    # r=t のサンプルは instantaneous velocity loss L_v なので
    # beta=1のままにし、r!=t の L_u にだけbetaを掛ける。
    # ============================================================
    def _get_kim_progressive_weights(
        self,
        r,
        t,
        global_step,
        total_steps,
    ):
        if self.kim_mode != 'progressive':
            return torch.ones_like(t)

        if global_step is None or total_steps is None:
            raise ValueError(
                'global_step and total_steps are required when '
                "kim_mode='progressive'."
            )

        total_steps = max(int(total_steps), 1)

        # Lightningのglobal_stepは通常 0, ..., T-1。
        # 論文の i/T に合わせ、[0, 1]にclipする。
        progress = min(max(float(global_step) / float(total_steps), 0.0), 1.0)
        s = 1.0 - progress ** self.kim_k

        # このMAC wrapperでは t <= r のため temporal gap は r-t。
        delta_t = (r - t).clamp(min=0.0, max=1.0)

        # まず全サンプルを1にする。これによりL_vは変更しない。
        beta = torch.ones_like(delta_t)

        # r=tはsample_time_steps()で厳密に代入されるが、
        # 数値的安全性のため小さいepsilonで判定する。
        u_mask = delta_t > 1e-8

        beta_u = (
            1.0
            - s
            + self.kim_lambda * s * (1.0 - delta_t)
        )

        beta = torch.where(u_mask, beta_u, beta)
        return beta

    # ============================================================
    # [ORIGINAL]
    # def get_loss(self, images, z0, labels, indices=None):
    #
    # [KIM CHANGE 5]
    # progressive schedule計算のため global_step / total_steps を追加。
    # ============================================================
    def get_loss(
        self,
        images,
        z0,
        labels,
        indices=None,
        global_step=None,
        total_steps=None,
    ):
        self.ema_model.eval()

        if self.add_weight == 0:
            weights = torch.ones(images.shape[0], device=images.device)
        else:
            if indices is not None:
                weights = torch.ones(images.shape[0], device=images.device)
                weights[indices] = 1 + self.add_weight

        device = images.device
        current_batch_size = images.shape[0]
        r, t = self.sample_time_steps(
            "logit_normal",
            current_batch_size,
            device,
        )
        t_full = t.view(-1, 1, 1, 1)
        r_full = r.view(-1, 1, 1, 1)

        # get dx at timestep t
        x_t = (1 - t_full) * z0 + t_full * images

        ut_gt = images - z0

        labels_dropout = torch.bernoulli(
            torch.full(labels.shape, self.CLASS_DROPOUT_PROB)
        ).to(images.device)
        labels_dropped = torch.where(
            labels_dropout.bool(),
            self.NUM_CLASSES,
            labels,
        )

        def u_func(z, t_in, r_in):
            h = r_in - t_in
            return self.model(z, t_in, h, labels_dropped)

        dtdt = torch.ones_like(t)
        drdt = torch.zeros_like(r)

        with torch.amp.autocast("cuda", enabled=False):
            u_pred, dudt = torch.func.jvp(
                u_func,
                (x_t, t, r),
                (ut_gt, dtdt, drdt),
            )
            u_tgt = (ut_gt + (r_full - t_full) * dudt).detach()

            loss = (u_pred - u_tgt) ** 2
            loss = loss.sum(dim=(1, 2, 3))  # squared L2 loss, per sample

            # ----------------------------------------------------
            # [ORIGINAL MeanFlow/MAC] adaptive weighting
            # ----------------------------------------------------
            adp_wt = (loss.detach() + self.norm_eps) ** self.norm_p
            loss = loss / adp_wt

            # ====================================================
            # [KIM CHANGE 6] progressive L_u weighting
            #
            # 重要な順序:
            # 1. MeanFlow adaptive weighting
            # 2. Kim progressive L_u weighting
            # 3. MAC selected-coupling weighting
            #
            # Kim betaはL_u(r!=t)だけに適用し、L_v(r=t)には掛けない。
            # ====================================================
            if self.kim_mode == 'progressive':
                kim_weights = self._get_kim_progressive_weights(
                    r=r,
                    t=t,
                    global_step=global_step,
                    total_steps=total_steps,
                )
                loss = loss * kim_weights

            # ----------------------------------------------------
            # [ORIGINAL MAC] selected low-loss couplings are upweighted
            # ----------------------------------------------------
            if indices is not None:
                loss = loss * weights

            loss = loss.mean()  # mean over batch dimension

        return loss

    # ============================================================
    # [ORIGINAL]
    # def forward(self, x, c, percentile, global_step):
    #
    # [KIM CHANGE 7]
    # progressive schedule用に total_steps=None を追加。
    # noneモードなら従来通り4引数でも動く。
    # ============================================================
    def forward(
        self,
        x,
        c,
        percentile,
        global_step,
        total_steps=None,
    ):
        z0 = torch.randn_like(x)
        batch = (x, c)

        if self.model_type == 'select':
            if self.add_weight == 0:
                loss = self.get_loss(
                    x,
                    z0,
                    c,
                    global_step=global_step,
                    total_steps=total_steps,
                )
            else:
                indices_low = select_low_loss_indices(
                    self.ema_model,
                    batch,
                    z0,
                    percentile,
                    model='meanflow',
                )
                loss = self.get_loss(
                    x,
                    z0,
                    c,
                    indices=indices_low,
                    global_step=global_step,
                    total_steps=total_steps,
                )

        elif self.model_type == 'full':
            z0, x, c = get_ot_pair(
                self.ema_model,
                batch,
                z0,
                global_step,
                model='meanflow',
            )
            loss = self.get_loss(
                x,
                z0,
                c,
                global_step=global_step,
                total_steps=total_steps,
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return loss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=16, cfg=2.0):
        """
        MeanFlow forward sampling: t=0 -> t=1.
        This wrapper uses t=current time, r=next time, so h=r-t>0.
        Each step updates z_next = z + (r-t) * u(z, t, h, cond).
        """
        device = z.device
        dtype = z.dtype
        b = z.size(0)

        if self.vae is not None:
            self.vae.to(device)

        # Keep stepwise frames for visualization.
        images = [
            z
            if self.vae is None
            else self.vae.decode(z / self.vae.config.scaling_factor)[0]
        ]

        # Time grid including both endpoints.
        time_steps = torch.linspace(
            0.0,
            1.0,
            sample_steps + 1,
            device=device,
            dtype=dtype,
        )

        for i in range(sample_steps):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]
            dt = t_next - t_cur

            # Build batch-shaped t/r/h tensors.
            r = torch.full(
                (b,),
                t_next.item(),
                device=device,
                dtype=dtype,
            )
            t = torch.full(
                (b,),
                t_cur.item(),
                device=device,
                dtype=dtype,
            )
            h = r - t

            # Compute conditional/unconditional vector fields.
            if null_cond is not None:
                vc = self.model(z, t, h, cond)
                vu = self.model(z, t, h, null_cond)
                v = vu + cfg * (vc - vu)
            else:
                v = self.model(z, t, h, cond)

            # MeanFlow forward step: z_next = z + dt * v.
            z = z + dt * v

        # Append the final visualization frame.
        if self.vae is not None:
            decoded = self.vae.decode(
                z / self.vae.config.scaling_factor
            )[0]
        else:
            decoded = z
        images.append(decoded)

        return images
