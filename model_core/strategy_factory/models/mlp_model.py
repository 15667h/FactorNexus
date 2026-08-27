"""
model_core/strategy_factory/models/mlp_model.py — M4 模型池：MLP 回归器

非线性对照模型（树模型之外的第二种结构）：3 层全连接 MLP，PyTorch 训练，
sklearn 风格接口（fit(X, y) → predict(X)），供 walk_forward 直接调用。

设计要点（机构级）：
  1. 时间顺序验证集：最后 10% 时间样本做早停验证（不打乱行序采样——
     金融时序样本行序=时间，随机 split 会把未来信息漏进验证集）
  2. NaN 特征 0 填充（GBDT 原生支持缺失；NN 需要显式填充。
     填充后再 StandardScaler，0 即均值水平）
  3. 早停 patience + 最优权重回滚
  4. device 自动（CUDA 可用则 GPU）

用法：
    from model_core.strategy_factory.models.mlp_model import make_mlp_regressor
    factory = lambda: make_mlp_regressor(hidden=(256, 128), epochs=20)
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


class MLPRegressor:
    """PyTorch MLP 回归器（sklearn 风格）。"""

    def __init__(self, hidden: tuple = (256, 128), dropout: float = 0.3,
                 epochs: int = 20, batch_size: int = 8192, lr: float = 1e-3,
                 patience: int = 3, seed: int = 42, device: str = "auto",
                 verbose: int = 0) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("MLP 需要 PyTorch：pip install torch")
        self.hidden = tuple(int(h) for h in hidden)
        self.dropout = dropout
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

    # ── 内部模型 ──────────────────────────────────────────────────────

    def _build_net(self) -> "nn.Module":
        dims = [self._n_features] + list(self.hidden) + [1]
        layers: list = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(dims[-2], 1))
        return nn.Sequential(*layers)

    # ── sklearn 接口 ──────────────────────────────────────────────────

    def fit(self, X, y) -> "MLPRegressor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2 or len(y) != X.shape[0]:
            raise ValueError(f"X 形状 {X.shape} 与 y 长度 {len(y)} 不匹配")
        self._n_features = X.shape[1]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # NaN → 0（NN 不支持缺失；0 经标准化后 = 均值水平）
        Xc = np.nan_to_num(X, nan=0.0)
        # 标准化（时间顺序：scaler 只用训练段统计量）
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler().fit(Xc)
        Xs = self._scaler.transform(Xc)

        # 验证集 = 最后 10% 时间样本（行序 = 时间序，防未来泄漏）
        n_val = max(int(len(Xs) * 0.1), 64)
        Xtr, Xva = Xs[:-n_val], Xs[-n_val:]
        ytr, yva = y[:-n_val], y[-n_val:]

        torch.manual_seed(self.seed)
        net = self._build_net().to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_f = nn.MSELoss()

        Xt = torch.from_numpy(Xtr).to(self.device)
        yt = torch.from_numpy(ytr).unsqueeze(1).to(self.device)
        Xv = torch.from_numpy(Xva).to(self.device)
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
                out = net(Xt[idx])
                loss = loss_f(out, yt[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                va_loss = float(loss_f(net(Xv), yv).item())
            if self.verbose:
                print(f"      [MLP] epoch {epoch + 1}/{self.epochs} "
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
            out = self._net(torch.from_numpy(Xs).to(self.device))
        return out.squeeze(1).detach().cpu().numpy().astype(np.float64)

    def score(self, X, y) -> float:
        """R²（sklearn 兼容，便于 cross_val 等工具）。"""
        p = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def make_mlp_regressor(hidden: tuple = (256, 128), dropout: float = 0.3,
                       epochs: int = 20, batch_size: int = 8192,
                       lr: float = 1e-3, patience: int = 3, seed: int = 42,
                       device: str = "auto", verbose: int = 0) -> MLPRegressor:
    """构造 MLP 回归器（M4；walk_forward model_factory 直接调用）。"""
    return MLPRegressor(hidden=hidden, dropout=dropout, epochs=epochs,
                        batch_size=batch_size, lr=lr, patience=patience,
                        seed=seed, device=device, verbose=verbose)
