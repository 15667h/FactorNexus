import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig
from .vocab import FORMULA_VOCAB


class NewtonSchulzLowRankDecay:
    """
    Low-Rank Decay (LoRD) using Newton-Schulz iteration.
    
    A more efficient regularization method that targets low-rank structure
    in attention and key parameters. Uses Newton-Schulz iteration to compute
    the minimum singular vectors without explicit SVD.
    
    Args:
        named_parameters: Model's named parameters
        decay_rate: Strength of low-rank decay
        num_iterations: Number of Newton-Schulz iterations (default: 5)
        target_keywords: If specified, only decay parameters matching these keywords
    """
    def __init__(self, named_parameters, decay_rate=1e-3, num_iterations=5, target_keywords=None):
        self.decay_rate = decay_rate
        self.num_iterations = num_iterations
        self.target_keywords = target_keywords or ["qk_norm", "attention"]
        self.params_to_decay = []
        
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            self.params_to_decay.append((name, param))
    
    @torch.no_grad()
    def step(self):
        """Apply Newton-Schulz low-rank decay to attention parameters."""
        for name, W in self.params_to_decay:
            orig_dtype = W.dtype
            X = W.float()
            r, c = X.shape
            
            # Transpose if needed for efficiency
            transposed = False
            if r > c:
                X = X.T
                transposed = True
            
            # Normalize by spectral norm
            norm = X.norm() + 1e-8
            X = X / norm
            
            # Initialize Y for Newton-Schulz iteration
            Y = X
            I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
            
            # Newton-Schulz iteration: Y_{k+1} = 0.5 * Y_k * (3*I - Y_k^T * Y_k)
            # This converges to the orthogonal matrix with same singular vectors
            for _ in range(self.num_iterations):
                A = Y.T @ Y
                Y = 0.5 * Y @ (3.0 * I - A)
            
            if transposed:
                Y = Y.T
            
            # Apply low-rank decay
            W.sub_(self.decay_rate * Y.to(orig_dtype))


class StableRankMonitor:
    """Monitor the effective rank (stable rank) of model parameters."""
    def __init__(self, model, target_keywords=None):
        self.model = model
        self.target_keywords = target_keywords or ["q_proj", "k_proj", "attention"]
        self.history = []
    
    @torch.no_grad()
    def compute(self):
        """Compute average stable rank of target parameters."""
        ranks = []
        for name, param in self.model.named_parameters():
            if param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            
            W = param.detach().float()
            S = torch.linalg.svdvals(W)
            # Stable Rank = ||W||_F^2 / ||W||_2^2
            stable_rank = (S.norm() ** 2) / (S[0] ** 2 + 1e-9)
            ranks.append(stable_rank.item())
        
        avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
        self.history.append(avg_rank)
        return avg_rank


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QKNorm(nn.Module):
    """Query-Key Normalization for Attention.

    【P2-2 死代码警告】：本类定义保留向后兼容，但 LoopedTransformerLayer
    已不再实例化它（forward 中未调用）。nn.MultiheadAttention 不直接支持
    QK-Norm 钩子，若未来需要接入需自定义 attention 实现。
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model) * (d_model ** -0.5))

    def forward(self, q, k):
        # Normalize Q and K independently
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return q_norm * self.scale, k_norm * self.scale


class SwiGLU(nn.Module):
    """Swish GLU activation function"""
    def __init__(self, d_in, d_ff):
        super().__init__()
        self.w = nn.Linear(d_in, d_ff * 2)
        self.fc = nn.Linear(d_ff, d_in)

    def forward(self, x):
        x_glu = self.w(x)
        x, gate = x_glu.chunk(2, dim=-1)
        x = x * F.silu(gate)  # Swish activation
        return self.fc(x)


class MTPHead(nn.Module):
    """Multi-Task Pooling Head.

    【P2-3 简化说明】：原设计含 3 个 task head + router，但 engine.py 未使用
    task_probs（无辅助 loss 约束 router），导致 multi-task 退化为单 head 加权
    平均。此处简化为单 head（nn.Linear）以减少参数浪费，forward 仍返回
    (logits, None) 保持调用方签名兼容。
    """
    def __init__(self, d_model, vocab_size, num_tasks=3):
        super().__init__()
        # 简化为单 head：删除 task_heads/task_weights/task_router
        self.head = nn.Linear(d_model, vocab_size)
        # 保留 num_tasks 属性供旧代码引用
        self.num_tasks = num_tasks

    def forward(self, x):
        logits = self.head(x)
        # 返回 (logits, task_probs=None) 保持调用方签名兼容
        return logits, None


class LoopedTransformerLayer(nn.Module):
    """Looped Transformer Layer - recurrent processing within a layer"""
    def __init__(self, d_model, nhead, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.num_loops = num_loops
        self.d_model = d_model
        self.nhead = nhead

        # 【P2-2 修复】：移除未使用的 qk_norm 实例（原代码 forward 中从未调用）
        # 若未来需要 QK-Norm，可在 forward 中接入 QKNorm 类
        # self.qk_norm = QKNorm(d_model // nhead)

        # Standard attention components
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # SwiGLU FFN instead of standard FFN
        self.ffn = SwiGLU(d_model, dim_feedforward)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, is_causal=False):
        # Looped processing - recurrent refinement
        for _ in range(self.num_loops):
            # Self-attention with residual
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask, is_causal=is_causal)
            x = x + self.dropout(attn_out)

            # FFN with residual
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            x = x + self.dropout(ffn_out)

        return x


class LoopedTransformer(nn.Module):
    """Looped Transformer Encoder with multiple loop iterations"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LoopedTransformerLayer(d_model, nhead, dim_feedforward, num_loops, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None, is_causal=False):
        for layer in self.layers:
            x = layer(x, mask=mask, is_causal=is_causal)
        return x


class AlphaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # d_model 64→96：vocab 剪枝后为 ~94，96 维 embedding 足以支撑，
        # 兼顾容量与 CPU 自回归采样的速度（128 维每步耗时过高）。
        self.d_model = 96
        self.features_list = list(FORMULA_VOCAB.feature_names)
        self.ops_list = list(FORMULA_VOCAB.operator_names)
        
        self.vocab = list(FORMULA_VOCAB.token_names)
        self.vocab_size = FORMULA_VOCAB.size
        
        # Embedding
        # pos_emb 用固定上限 20，与 MAX_FORMULA_LEN 解耦：
        # - 阶段A (len=8) 和阶段B (len=14) 都能用同一模型权重
        # - 测试无需随配置变更而调整
        # - 手工评估或 14-token 公式都在范围内
        _POS_EMB_MAX = 20
        self._max_seq = _POS_EMB_MAX
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, _POS_EMB_MAX, self.d_model))
        
        # Enhanced Transformer with Looped Transformer
        # num_layers 2→3、dim_feedforward 128→192：配合 d_model=96 适度扩容。
        # 4 层 looped(×3 loops) 在 CPU 自回归采样下每步耗时过高，取 3 层平衡。
        # nhead=4 → head_dim=24。
        self.blocks = LoopedTransformer(
            d_model=self.d_model,
            nhead=4,
            num_layers=3,
            dim_feedforward=192,
            num_loops=3,
            dropout=0.1
        )
        
        # RMSNorm instead of LayerNorm
        self.ln_f = RMSNorm(self.d_model)

        # MTPHead for multi-task output（已简化为单 head）
        self.mtp_head = MTPHead(self.d_model, self.vocab_size, num_tasks=3)
        # 【P2-1 修复】：移除未使用的 head_critic（engine.py 三处调用均丢弃 value）
        # 若未来实现 actor-critic，可重新添加
        # self.head_critic = nn.Linear(self.d_model, 1)

    def forward(self, idx):
        # idx: [Batch, SeqLen]
        B, T = idx.size()
        if T > self._max_seq:
            raise ValueError(
                f"Input sequence length {T} exceeds max_seq {self._max_seq}. "
                f"Increase ModelConfig.MAX_FORMULA_LEN."
            )
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]

        # Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)

        # Process through looped transformer
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)

        last_emb = x[:, -1, :]

        # MTPHead 简化为单 head，返回 (logits, None)
        logits, task_probs = self.mtp_head(last_emb)
        # 【P2-1 修复】：value 已废弃，返回 None 保持 3 元组签名兼容调用方
        value = None

        return logits, value, task_probs
