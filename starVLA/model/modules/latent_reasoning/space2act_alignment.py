"""训练期将少量 Space2Act reasoning token 扩展到教师 token 空间。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLatentAligner(nn.Module):
    """用 cross-attention 将 K 个类型化 VLM token 映射为 N 个教师 token。"""

    def __init__(
        self,
        input_dim: int,
        target_tokens: int,
        target_dim: int,
        alignment_dim: int = 1024,
        num_heads: int = 8,
        query_chunk_size: int = 128,
        mse_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if alignment_dim % num_heads:
            raise ValueError("alignment_dim must be divisible by num_heads")
        self.target_tokens = int(target_tokens)
        self.target_dim = int(target_dim)
        self.query_chunk_size = int(query_chunk_size)
        self.mse_weight = float(mse_weight)

        # compact hidden 先投影到共享 alignment space，作为 cross-attention 的 K/V。
        self.memory_projection = nn.Linear(input_dim, alignment_dim)
        self.memory_norm = nn.LayerNorm(alignment_dim)
        # target_queries 的数量与教师 token 数完全一致。query 之间不做 self-attention，
        # 每个位置只从紧凑 reasoning token 中读取信息，最后投影到教师特征维度。
        self.target_queries = nn.Parameter(torch.empty(self.target_tokens, alignment_dim))
        self.cross_attention = nn.MultiheadAttention(
            alignment_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.query_norm = nn.LayerNorm(alignment_dim)
        self.ffn = nn.Sequential(
            nn.Linear(alignment_dim, alignment_dim * 4),
            nn.GELU(),
            nn.Linear(alignment_dim * 4, alignment_dim),
        )
        self.ffn_norm = nn.LayerNorm(alignment_dim)
        self.output_projection = nn.Linear(alignment_dim, self.target_dim)
        nn.init.normal_(self.target_queries, std=0.02)

    def _predict_chunk(self, memory: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """只展开一段 target query，返回 [B, end-start, target_dim]。"""
        queries = self.target_queries[start:end].unsqueeze(0).expand(memory.shape[0], -1, -1)
        attended, _ = self.cross_attention(queries, memory, memory, need_weights=False)
        hidden = self.query_norm(queries + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        return self.output_projection(hidden)

    def compute_loss(
        self, compact_hidden: torch.Tensor, teachers: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """返回传统的等权 batch mean，对旧配置保持兼容。"""
        return self.compute_per_example_loss(compact_hidden, teachers).mean()

    def compute_per_example_loss(
        self, compact_hidden: torch.Tensor, teachers: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """返回每个样本的 cosine + MSE alignment loss，shape 为 [B]。"""
        if compact_hidden.ndim != 3:
            raise ValueError(f"compact_hidden must be [B,K,H], got {tuple(compact_hidden.shape)}")
        if len(teachers) != compact_hidden.shape[0]:
            raise ValueError("Teacher batch size does not match compact_hidden")
        expected = (self.target_tokens, self.target_dim)
        for teacher in teachers:
            if tuple(teacher.shape) != expected:
                raise ValueError(f"Expected teacher shape {expected}, got {tuple(teacher.shape)}")

        memory = self.memory_norm(self.memory_projection(compact_hidden))
        batch_size = compact_hidden.shape[0]
        cosine_sum = compact_hidden.new_zeros((batch_size,), dtype=torch.float32)
        mse_sum = compact_hidden.new_zeros((batch_size,), dtype=torch.float32)
        # teacher 通常是 CPU 上的大 tensor。按 query_chunk_size 预测和传输，避免一次
        # 将完整 [B,N,D_teacher] 目标及其计算图放入显存；分块不改变最终 loss 定义。
        for start in range(0, self.target_tokens, self.query_chunk_size):
            end = min(start + self.query_chunk_size, self.target_tokens)
            prediction = self._predict_chunk(memory, start, end).float()
            teacher_chunk = torch.stack(
                [teacher[start:end] for teacher in teachers], dim=0
            ).to(device=prediction.device, dtype=torch.float32, non_blocking=True)
            # 在教师维度上做无参数 LayerNorm，降低不同教师特征尺度的影响；cosine
            # 约束方向，MSE 作为较小权重的逐维补充。
            prediction = F.layer_norm(prediction, (self.target_dim,))
            teacher_chunk = F.layer_norm(teacher_chunk, (self.target_dim,))
            cosine_sum = cosine_sum + (1.0 - F.cosine_similarity(prediction, teacher_chunk, dim=-1)).sum(
                dim=1
            )
            mse_sum = mse_sum + (prediction - teacher_chunk).square().sum(dim=(1, 2))

        # 显式按 batch/token（以及 MSE 的 feature dim）归一化，使 loss 尺度不随
        # 教师 token 数或维度变化，便于 S/F/P 通道使用统一 loss_weight。
        cosine_loss = cosine_sum / self.target_tokens
        mse_loss = mse_sum / (self.target_tokens * self.target_dim)
        return cosine_loss + self.mse_weight * mse_loss
