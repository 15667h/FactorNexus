"""
model_core/strategy_factory/models/ssm_model.py — M4 模型池：SSM（S4）对照

状态空间模型（State Space Model）对照——Mamba 家族的基座（Mamba = S4 +
选择机制 + 硬件感知扫描）。本实现为纯 PyTorch 对角化 S4（零外部依赖，
不要求 mamba-ssm 编译环境），作为「序列结构编码器」对照：

  每样本特征向量 x ∈ R^F → 视为长度为 F 的序列 → S4 层迭代编码 → 末态
  实部 → 线性读出 → 预测。

与 MLP（非序列前馈）形成对照：验证「显式序列状态建模」相对普通前馈
在因子数据上是否带来增益（业界 Mamba 论文的典型对照设计）。

若环境安装了 mamba_ssm，make_mamba_regressor() 会尝试包装真 Mamba
（时序面板场景，需按 (股票, 时间) 组织样本——当前 walk_forward 行式
接口下默认回退 S4 并提示）。

用法：
    from model_core.strategy_factory.models.ssm_model import make_s4_regressor
    factory = lambda: make_s4_regressor(d_state=64, epochs=20)
"""
from __future__ import annotations

import time

import numpy as np

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    _HAS_TORCH = False


def _default_device() -> str:
    if _HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _S4Cell(nn.Module):
    """对角化 S4 层：h_t = A⊙h_{t-1} + B·x_t，y_t = Re(C⊙h_t + D·x_t)。"""

    def __init__(self, d_in: int, d_state: int = 64, seed: int = 42) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # A 对角：实部为负保证稳定（-0.5），虚部随机（频率）
        theta = torch.randn(d_state, generator=g) * 0.5
        self.A_real = nn.Parameter(torch.full((d_state,), -0.5))
        self.A_imag = nn.Parameter(theta)
        # B/C/D 输入投影
        self.B = nn.Parameter(torch.randn(d_state, d_in,
                                          generator=g) / (d_in ** 0.5))
        self.C = nn.Parameter(torch.randn(d_state, generator=g)
                              / (d_state ** 0.5))
        self.D = nn.Parameter(torch.zeros(1))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: [B, F, d_in]（序列=特征维）
        Bsz, F, _ = x.shape
        A = torch.complex(self.A_real, self.A_imag)          # [d_state]
        B = torch.complex(self.B, torch.zeros_like(self.B))  # [d_state, d_in]
        C = torch.complex(self.C, torch.zeros_like(self.C))  # [d_state]
        h = torch.zeros(Bsz, A.shape[0], dtype=torch.complex64,
                        device=x.device)
        ys = []
        for t in range(F):
            h = A * h + torch.einsum("sd,bd->bs", B, x[:, t])
            # D·x 保持 [B,1] 以广播到 [B, d_state]（每状态一个输出通道）
            y = (C * h).real + self.D * x[:, t, :1]
            ys.append(y)
        return torch.stack(ys, dim=1)                        # [B, F, d_state]


class S4Regressor:
    """S4 编码器 + 线性头回归器（sklearn 风格，walk_forward 兼容）。"""

    def __init__(self, d_state: int = 64, epochs: int = 20,
                 batch_size: int = 8192, lr: float = 1e-3,
                 patience: int = 3, seed: int = 42, device: str = "auto",
                 verbose: int = 0) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("S4 需要 PyTorch：pip install torch")
        self.d_state = int(d_state)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = lr
        self.patience = int(patience)
        self.seed = int(seed)
        self.device = _default_device() if device == "auto" else device
        self.verbose = verbose
        self._net = None
        self._scaler = None
        self._n_features = 0

    def _build_net(self) -> "nn.Module":
        return nn.Sequential(
            _S4Cell(self._n_features, self.d_state, seed=self.seed),
            nn.Linear(self.d_state, 1),
        )

    def fit(self, X, y) -> "S4Regressor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2 or len(y) != X.shape[0]:
            raise ValueError(f"X 形状 {X.shape} 与 y 长度 {len(y)} 不匹配")
        self._n_features = X.shape[1]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        Xc = np.nan_to_num(X, nan=0.0)
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler().fit(Xc)
        Xs = self._scaler.transform(Xc)

        n_val = max(int(len(Xs) * 0.1), 64)
        Xtr, Xva = Xs[:-n_val], Xs[-n_val:]
        ytr, yva = y[:-n_val], y[-n_val:]

        torch.manual_seed(self.seed)
        net = self._build_net().to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_f = nn.MSELoss()

        Xt = torch.from_numpy(Xtr).unsqueeze(1).to(self.device)  # [n, 1, F]
        Xt = Xt.transpose(1, 2)                                  # [n, F, 1]
        yt = torch.from_numpy(ytr).unsqueeze(1).to(self.device)
        Xv = torch.from_numpy(Xva).unsqueeze(1).to(self.device)
        Xv = Xv.transpose(1, 2)
        yv = torch.from_numpy(yva).unsqueeze(1).to(self.device)
        n = len(Xt)
        n_batches = max(1, int(np.ceil(n / self.batch_size)))

        best_loss = float("inf")
        best_state = None
        wait = 0
        t0 = time.time()
        for epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(n, device=self.device)
            for b in range(n_batches):
                idx = perm[b * self.batch_size:(b + 1) * self.batch_size]
                opt.zero_grad()
                out = net(Xt[idx])[:, -1]                        # 末态读出
                loss = loss_f(out, yt[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                va_loss = float(loss_f(net(Xv)[:, -1], yv).item())
            if self.verbose:
                print(f"      [S4] epoch {epoch + 1}/{self.epochs} "
                      f"val_mse={va_loss:.6f} ({time.time() - t0:.1f}s)")
            if va_loss < best_loss - 1e-7:
                best_loss = va_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = net
        return self

    def predict(self, X) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("先 fit 再 predict")
        X = np.asarray(X, dtype=np.float32)
        Xc = np.nan_to_num(X, nan=0.0)
        Xs = self._scaler.transform(Xc)
        self._net.eval()
        with torch.no_grad():
            xt = torch.from_numpy(Xs).unsqueeze(1).transpose(1, 2).to(
                self.device)
            out = self._net(xt)[:, -1]
        return out.squeeze(1).detach().cpu().numpy().astype(np.float64)

    def score(self, X, y) -> float:
        p = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def make_s4_regressor(d_state: int = 64, epochs: int = 20,
                      batch_size: int = 8192, lr: float = 1e-3,
                      patience: int = 3, seed: int = 42,
                      device: str = "auto", verbose: int = 0) -> S4Regressor:
    """构造 S4 对照回归器（M4；walk_forward model_factory 直接调用）。"""
    return S4Regressor(d_state=d_state, epochs=epochs,
                       batch_size=batch_size, lr=lr, patience=patience,
                       seed=seed, device=device, verbose=verbose)


def make_mamba_regressor(epochs: int = 20, device: str = "auto",
                         verbose: int = 0):
    """真 Mamba 包装（需 pip install mamba-ssm，CUDA 编译环境）。

    当前 walk_forward 行式样本接口无法提供 (股票, 时间) 序列轴，
    因此优先回退 S4 对照实现；面板时序场景可自行扩展。
    """
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "mamba_ssm 未安装（需要 CUDA 编译环境）。已回退方案："
            "make_s4_regressor() 纯 PyTorch S4 对照。") from None
    # mamba_ssm 的 MixerModel 面向语言建模（B,L,D）；因子横截面行式样本
    # 需先构造 (样本, 序列, 特征) 张量，接口与 sklearn 不兼容——见 S4Regressor。
    raise NotImplementedError(
        "真 Mamba 需面板时序数据组织（见 docstring）；本环境使用 S4 对照。")
