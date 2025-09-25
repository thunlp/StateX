from typing import Tuple
import math

from torch import nn, Tensor
import torch
import torch.nn.functional as F
from einops import rearrange, einsum
from .utils import RMSNorm
from .rope import apply_rotary_emb


class KVCache(nn.Module):
    def __init__(
        self,
        batch_size: int,
        seq_length: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.register_buffer(
            "cache_k", torch.zeros(cache_shape, dtype=dtype, device=device)
        )
        self.register_buffer(
            "cache_v", torch.zeros(cache_shape, dtype=dtype, device=device)
        )

    def update(self, start_pos: int, xk: Tensor, xv: Tensor) -> Tuple[Tensor, Tensor]:
        seqlen = xk.size(1)
        self.cache_k[:, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, start_pos : start_pos + seqlen] = xv
        xk = self.cache_k[:, : start_pos + seqlen]
        xv = self.cache_v[:, : start_pos + seqlen]
        return xk, xv


class CausalSelfAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_head: int,
        n_kv_head: int,
        use_q_norm: bool = False,
        use_k_norm: bool = False,
        tie_kv: bool = False,
        max_len: int = 4096,
        head_mixing: int = 1,
        device: str = "cuda",
        k_to_v: bool = False,
    ):
        super().__init__()

        self.device = device
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.use_q_norm = use_q_norm
        self.use_k_norm = use_k_norm
        self.tie_kv = tie_kv
        self.head_mixing = head_mixing
        self.k_to_v = k_to_v

        self.group_size = n_head // n_kv_head

        self.w_q = nn.Linear(d_model, n_head * dim_k, bias=False)
        if self.tie_kv:
            assert dim_k == dim_v, "tie_kv requires dim_k == dim_v"
            self.w_kv = nn.Linear(d_model, n_kv_head * dim_k, bias=False)
        else:
            self.w_v = nn.Linear(d_model, n_kv_head * dim_v, bias=False)
            if self.k_to_v:
                self.w_k = nn.Linear(dim_v, dim_k, bias=False)
            else:
                self.w_k = nn.Linear(d_model, n_kv_head * dim_k, bias=False)

        self.w_o = nn.Linear(n_head * dim_v, d_model, bias=False)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.use_flash_attn = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if not self.use_flash_attn:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len),
            )

        if self.use_q_norm:
            self.q_norm = RMSNorm(dim_k)

        if self.use_k_norm and not self.tie_kv:
            self.k_norm = RMSNorm(dim_k)

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor | None = None,  # Not used currently.
        return_kvs: bool = False,
        kv_cache: None | Tensor | tuple[Tensor, Tensor] | KVCache = None,
        pos_embs: None | Tensor = None,
    ) -> dict[str, Tensor | None | tuple[Tensor, Tensor]]:
        """
        Args:
            x: (B, T, D)
            pos_embs: (T, dim_k/2, 2, 2), often called freqs_cis.

        ---
        B: batch size
        T: sequence length
        D: model dimension
        """
        B, T, D = x.shape
        q = self.w_q(x)  # (B, T, H * dim_k)
        q = rearrange(q, "b t (h dk) -> b t h dk", h=self.n_head)

        if self.use_q_norm:
            q = self.q_norm(q)

        if self.tie_kv:
            # (B, T, H * dim_k)
            kv = self.w_kv(x)
            kv = rearrange(kv, "b t (h dk) -> b t h dk", h=self.n_kv_head)

            if self.use_k_norm:
                kv = self.k_norm(kv)
        else:
            # (B, T, H * dim_v)
            v = self.w_v(x)  # (B, T, H * dim_v)
            v = rearrange(v, "b t (h dv) -> b t h dv", h=self.n_kv_head)

            if self.k_to_v:
                k = self.w_k(v)
            else:
                k = self.w_k(x)  # (B, T, H * dim_k)
                k = rearrange(k, "b t (h dk) -> b t h dk", h=self.n_kv_head)

            if self.use_k_norm:
                k = self.k_norm(k)

        # Add RoPE
        if pos_embs is not None:
            q, k = apply_rotary_emb(
                xq=q,
                xk=k,
                seq_dim=1,
                freqs_cis=pos_embs[0:T],
            )

        # Concatenate with KV cache
        if kv_cache is not None:
            # Concatenate current KVs and the KV cache
            if self.tie_kv:
                assert isinstance(kv_cache, Tensor)
                kv = torch.cat((kv_cache, kv), dim=1)  # (B, T, H, dim_kv)
            else:
                assert isinstance(kv_cache, tuple)
                # kv_cache is a tuple of (k, v)
                k_cache, v_cache = kv_cache  # (B, T_ctx, H, dim_k)
                k = torch.cat((k_cache, k), dim=1)  # (B, T_ctx + T, H, dim_k)
                v = torch.cat((v_cache, v), dim=1)  # (B, T_ctx + T, H, dim_v)

        # Apply GQA and/or head mixing.
        if self.head_mixing > 1:
            if self.group_size > 1:
                """
                 Q: 0123456701234567
                KV: 0000111122223333
                """
                # TODO: implement head mixing
                raise NotImplementedError("Head mixing with GQA is not implemented")
            else:
                """
                 Q: 000111222
                KV: 012012012
                """
                q = q.repeat(
                    (1, 1, self.head_mix_size, 1)
                )  # (B, T, H * mix_size, dim_k)
                k = k.repeat_interleave(
                    self.head_mix_size, dim=2
                )  # (B, T, H * mix_size, dim_k)
                v = v.repeat_interleave(
                    self.head_mix_size, dim=2
                )  # (B, T, H * mix_size, dim_v)
        elif self.group_size > 1:
            # GQA: duplicate the keys and values for each group
            if self.tie_kv:
                kv = kv.repeat_interleave(
                    self.group_size, dim=2
                )  # (B, T, H * G, dim_kv)
            else:
                k = k.repeat_interleave(self.group_size, dim=2)  # (B, T, H * G, dim_k)
                v = v.repeat_interleave(self.group_size, dim=2)  # (B, T, H * G, dim_v)

        # Return KV cache if needed
        if return_kvs:
            if self.tie_kv:
                kvs = kv
            else:
                kvs = (k, v)
        else:
            kvs = None

        # Causal self-attention
        # (B, H, T, dim_v) x (B, H, dim_v, T)
        # -> (B, H, T, T)
        if self.use_flash_attn:
            # NOTE: When dim_k != dim_v, we cannot use flash_attn
            # efficient attention using Flash Attention CUDA kernels
            if self.tie_kv:
                q, kv = map(lambda e: e.transpose(1, 2), (q, kv))  # (B, H, T, dim_kv)
                output = F.scaled_dot_product_attention(q, kv, kv, is_causal=True)
            else:
                q, k, v = map(lambda e: e.transpose(1, 2), (q, k, v))  # (B, H, T, dim_k)
                output = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # (B, H, T, dim_v)
        else:
            # manual implementation of attention
            # note: my implementation, not sure if it's correct
            att: Tensor = einsum(q, k, "b h t dk, b h s dk -> b h t s") / math.sqrt(
                self.dim_k
            )  # (B, H, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            # (B, H, T, T) x (B, H, T, DV) -> (B, H, T, DV)
            output: Tensor = att @ v

        output = rearrange(output, "b h t dv -> b t (h dv)").contiguous()
        output = self.w_o(output)  # (B, T, D)
        return {
            "output": output,
            "kvs": kvs,
        }
