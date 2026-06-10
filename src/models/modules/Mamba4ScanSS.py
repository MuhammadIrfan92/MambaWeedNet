import torch
import torch.nn as nn
from functools import partial
from typing import Callable
from timm.models.layers import DropPath
import math
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class SS2D_dirSel(nn.Module):
    """
    Single-direction SS2D with selectable scan direction.

    Supported directions (strings):
      (1) "hw_forward"   : HW forward (original feature map)
      (2) "wh_forward"   : WH forward (transpose H<->W before flatten)
      (3) "hw_reverse"   : HW reversed (flip the HW-flattened sequence)
      (4) "wh_reverse"   : WH reversed (transpose then flip the sequence)

    Key guarantee:
      - We scan only ONE sequence (no K=4 duplicates).
      - Regardless of scan direction, the returned y is aligned back to HW order
        so downstream code remains unchanged.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # pre-processing
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # single-direction projections (no K=4 stacks)
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.dt_proj = self.dt_init(
            self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs
        )

        # single copy of SSM parameters (no copies=4)
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def _make_sequence(self, x: torch.Tensor, direction: str) -> tuple[torch.Tensor, dict]:
        """
        x: (B, D, H, W)
        Returns:
          xs: (B, D, L)   sequence to scan
          meta: info needed to re-align output back to HW order
        """
        B, D, H, W = x.shape
        L = H * W
        meta = {"direction": direction, "H": H, "W": W}

        if direction == "hw_forward":
            xs = x.contiguous().view(B, D, L)
        elif direction == "wh_forward":
            # transpose to (B, D, W, H), then flatten in WH order
            xt = x.transpose(2, 3).contiguous()
            xs = xt.view(B, D, L)
        elif direction == "hw_reverse":
            xs = x.contiguous().view(B, D, L)
            xs = torch.flip(xs, dims=[-1])
        elif direction == "wh_reverse":
            xt = x.transpose(2, 3).contiguous()
            xs = xt.view(B, D, L)
            xs = torch.flip(xs, dims=[-1])
        else:
            raise ValueError(
                f"Unknown direction='{direction}'. Choose from: "
                f"hw_forward, wh_forward, hw_reverse, wh_reverse"
            )
        return xs, meta

    def _align_to_hw(self, y_seq: torch.Tensor, meta: dict) -> torch.Tensor:
        """
        y_seq: (B, D, L) in the scan order used.
        Returns:
          y_hw: (B, D, L) aligned to HW-forward order.
        """
        direction = meta["direction"]
        H, W = meta["H"], meta["W"]
        B, D, L = y_seq.shape
        assert L == H * W

        if direction == "hw_forward":
            return y_seq

        if direction == "hw_reverse":
            # undo reversal
            return torch.flip(y_seq, dims=[-1])

        if direction == "wh_forward":
            # y is in WH order => reshape as (B, D, W, H), transpose back to (B, D, H, W), then flatten HW
            y_wh = y_seq.view(B, D, W, H)
            y_hw = y_wh.transpose(2, 3).contiguous().view(B, D, L)
            return y_hw

        if direction == "wh_reverse":
            # undo reversal first, then undo WH transpose
            y_unrev = torch.flip(y_seq, dims=[-1])
            y_wh = y_unrev.view(B, D, W, H)
            y_hw = y_wh.transpose(2, 3).contiguous().view(B, D, L)
            return y_hw

        return y_seq

    def forward_core(self, x: torch.Tensor, direction: str = "hw_forward") -> torch.Tensor:
        """
        x: (B, d_inner, H, W)
        direction: one of the four options
        Returns:
          y: (B, d_inner, L) aligned to HW-forward order
        """
        B, D, H, W = x.shape
        L = H * W
        # 1) build ONE sequence according to requested direction (no duplicates)
        xs, meta = self._make_sequence(x, direction=direction)  # (B, D, L)
        # 2) input-conditioned parameters via learned projections
        x_dbl = self.x_proj(xs.transpose(1, 2)).transpose(1, 2).contiguous()  # (B, dt_rank+2*d_state, L)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = self.dt_proj(dts.transpose(1, 2)).transpose(1, 2).contiguous()
        # 3) selective scan (single stream)
        xs_f = xs.float()                      # (B, D, L)
        dts_f = dts.float()                    # (B, D, L)
        Bs_f = Bs.float().unsqueeze(1)         # (B, 1, d_state, L)  (K=1)
        Cs_f = Cs.float().unsqueeze(1)         # (B, 1, d_state, L)

        Ds = self.Ds.float().view(-1)          # (D,)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (D, d_state)
        dt_bias = self.dt_proj.bias.float().view(-1)                 # (D,)

        y_seq = self.selective_scan(
            xs_f,
            dts_f,
            As,
            Bs_f,
            Cs_f,
            Ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        )  # (B, D, L)

        # 4) re-align output to HW-forward order so the rest of the pipeline is stable
        y_hw = self._align_to_hw(y_seq, meta)
        assert y_hw.shape == (B, D, L)
        return y_hw

    def forward(self, x: torch.Tensor, direction: str = "hw_forward", **kwargs):
        """
        x: (B, H, W, C=d_model)
        direction: selects scan order (hw_forward / wh_forward / hw_reverse / wh_reverse)
        """
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (B, d_inner, H, W)
        y = self.forward_core(x, direction=direction)  # (B, d_inner, L) aligned to HW
        y = y.transpose(1, 2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock_seldir(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        direction: str = "hw_forward",
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.direction = direction
        self.self_attention = SS2D_dirSel(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        layer1 = self.ln_1(input)
        ssd_layer = self.self_attention(layer1, direction=self.direction)
        x = input + self.drop_path(ssd_layer)
        return x
    

Mamba4Scan = VSSBlock_seldir