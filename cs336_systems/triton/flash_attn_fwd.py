"""
1.3.2-complete, single-head implementation.

What you get:
- Triton kernel that:
  * Loads one Q tile [Q_TILE, D] once
  * Loops K/V tiles [K_TILE, D] exactly ONCE (no S^2 allocation)
  * Applies online softmax per row (m, l) and accumulates O
  * Optional causal masking
  * Writes O and L (where L = m + log(l)) for backward/recompute
- PyTorch autograd.Function wrapper that calls the kernel in forward.
- Small reference check vs. PyTorch attention for sanity.
"""

import math
import argparse
import torch
import triton
import triton.language as tl




@triton.jit
def flash_fwd_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    # Strides (in elements)
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    # Problem sizes (runtime)
    N_QUERIES, N_KEYS,
    scale,                                      # 1 / sqrt(D)
    # Meta-parameters (compile-time)
    D: tl.constexpr,                            # head dimension
    Q_TILE_SIZE: tl.constexpr,                  # tile size for queries (M dimension)
    K_TILE_SIZE: tl.constexpr,                  # tile size for keys/values (N dimension)
    is_causal: tl.constexpr,
):
    """
    One Triton program handles:
      - one query tile (Q_TILE_SIZE rows)
      - one batch index b
    and streams over all K/V tiles with a single loop.
    """

    # ---- Program IDs ----
    qtile = tl.program_id(0)  # which query tile
    b     = tl.program_id(1)  # which batch (you can fold heads into B upstream)

    # ---- Block pointers for Q, O, L (nice 2D views with boundary checks) ----
    Q_block = tl.make_block_ptr(
        base=Q_ptr + b * stride_qb,
        shape=(N_QUERIES, D),                   # full [S, D]
        strides=(stride_qq, stride_qd),         # how to move in Q
        offsets=(qtile * Q_TILE_SIZE, 0),       # this tile's top-left
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),                           # D is fastest
    )
    O_block = tl.make_block_ptr(                # Output Blocks
        base=O_ptr + b * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(qtile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block = tl.make_block_ptr(                # LogSumExp Blocks
        base=L_ptr + b * stride_lb,
        shape=(N_QUERIES,),                  # [S]
        strides=(stride_lq,),
        offsets=(qtile * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # ---- On-chip state (fp32 compute path) ----
    Q_tile = tl.load(Q_block, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # [bq, D]
    O_acc  = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)                                   # [bq, D]
    m      = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)                       # [bq]
    l      = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)                                     # [bq]

    # Absolute query indices for causal masking
    if is_causal:
        q_abs = (qtile * Q_TILE_SIZE) + tl.arange(0, Q_TILE_SIZE)  # [bq]

    # ---- Single loop over K/V tiles ----
    k0 = 0
    while k0 < N_KEYS: # loop over key tiles
        # Block pointers for current K/V tiles
        K_block = tl.make_block_ptr(
            base=K_ptr + b * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(k0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block = tl.make_block_ptr(
            base=V_ptr + b * stride_vb,
            shape=(N_KEYS, D),  # teaching simplification: dv == D
            strides=(stride_vk, stride_vd),
            offsets=(k0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        # Load K and V tiles
        K_tile = tl.load(K_block, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # [bk, D]
        V_tile = tl.load(V_block, boundary_check=(0, 1), padding_option="zero")                  # [bk, D]

        # Scores S_ij = Q_tile @ K_tile^T * scale  → [bq, bk], fp32 accumulate
        S_ij = tl.dot(Q_tile, tl.trans(K_tile), out_dtype=tl.float32) * scale

        # tail mask (keep valid keys; set OOB to big negative)
        k_abs = k0 + tl.arange(0, K_TILE_SIZE)     # [bk]
        k_mask = k_abs < N_KEYS                    # [bk]
        S_ij = tl.where(k_mask[None, :], S_ij, -1e6)

        # causal (only if enabled)
        if is_causal:
            mask_tri = k_abs[None, :] > q_abs[:, None]   # [bq, bk]
            S_ij = tl.where(mask_tri, -1e6, S_ij)

        # Online softmax update (rowwise over bk):
        # m_new = max(m, rowmax(S_ij));  l = exp(m - m_new)*l + sum(exp(S_ij - m_new));
        # O_acc = exp(m - m_new)*O_acc + (exp(S_ij - m_new) @ V_tile)
        tmax  = tl.max(S_ij, axis=1)                       # [bq]
        m_new = tl.maximum(m, tmax)                        # [bq]
        P_tld = tl.exp(S_ij - m_new[:, None])              # [bq, bk]
        alpha = tl.exp(m - m_new)                          # [bq]

        # Accumulate O in fp32; allow V_tile’s dtype as input, but keep out_dtype fp32
        O_acc = alpha[:, None] * O_acc + tl.dot(P_tld, V_tile.to(tl.float32), out_dtype=tl.float32)
        l     = alpha * l + tl.sum(P_tld, axis=1)
        m     = m_new

        k0 += K_TILE_SIZE

    # Finalize: O = O_acc / l, and L = m + log(l) (for backward/recompute)
    O_out = O_acc / l[:, None]
    tl.store(O_block, O_out.to(O_block.type.element_ty), boundary_check=(0, 1))

    L_row = m + tl.log(l + 1e-20)
    tl.store(L_block, L_row, boundary_check=(0,))



# ================================
class FlashAttention2TritonForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False,
                q_tile: int = 128, k_tile: int = 128):
        """
        Inputs:
          Q, K, V: [B, S, D], CUDA tensors (single head; for multi-head, fold H into B or add another pid).
          is_causal: whether to apply causal masking.
          q_tile, k_tile: tile sizes (try 128/128 to start).
        Returns:
          O: [B, S, D]
        Side-effect:
          Saves (Q, K, V, O, L) into ctx for potential backward/recompute (L is log-sum-exp per row).
        """
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        B, S, D = Q.shape
        device = Q.device
        scale  = 1.0 / math.sqrt(D)

        O = torch.empty_like(Q)
        # L per row (float32) — required for exact recomputation-based backward
        L = torch.empty(B, S, device=device, dtype=torch.float32)

        grid = (triton.cdiv(S, q_tile), B)

        flash_fwd_kernel[grid](
            # Pointers
            Q, K, V, O, L,
            # Strides (elements)
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            # Sizes + scale
            S, S, scale,
            # Meta
            D=D, Q_TILE_SIZE=q_tile, K_TILE_SIZE=k_tile, is_causal=is_causal,
        )

        # Store what you’ll need for backward/recompute (not implemented here)
        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, O, L)
        return O


def flash_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                  is_causal: bool = False, q_tile: int = 128, k_tile: int = 128) -> torch.Tensor:
    """Convenience wrapper."""
    return FlashAttention2TritonForward.apply(Q, K, V, is_causal, q_tile, k_tile)


# ================================
# Small correctness checks & (optional) bench
# ================================
def check_small(causal=False, dtype=torch.bfloat16):
    torch.manual_seed(0)
    B, S, D = 1, 128, 64
    Q = torch.randn(B, S, D, device="cuda", dtype=dtype)
    K = torch.randn(B, S, D, device="cuda", dtype=dtype)
    V = torch.randn(B, S, D, device="cuda", dtype=dtype)

    O = flash_forward(Q, K, V, is_causal=causal, q_tile=128, k_tile=128)

    # PyTorch reference: do softmax in fp32 to match our numerics
    scale = 1.0 / math.sqrt(D)
    S_full = torch.einsum("bsd,btd->bst", Q.float(), K.float()) * scale
    if causal:
        idx = torch.arange(S, device=Q.device)
        mask = idx[None, :] <= idx[:, None]  # allow only keys <= query
        S_full = torch.where(mask, S_full, torch.full_like(S_full, -1e6))
    P = torch.softmax(S_full, dim=-1)
    O_ref = torch.einsum("bst,btd->bsd", P, V.float()).to(dtype)

    atol, rtol = (1e-3, 1e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)
    torch.testing.assert_close(O, O_ref, atol=atol, rtol=rtol)
    print(f"[OK] causal={causal}  dtype={dtype}  B={B} S={S} D={D}")


def bench_small(dtype=torch.bfloat16):
    import time
    torch.manual_seed(0)
    B, S, D = 2, 512, 128
    Q = torch.randn(B, S, D, device="cuda", dtype=dtype)
    K = torch.randn(B, S, D, device="cuda", dtype=dtype)
    V = torch.randn(B, S, D, device="cuda", dtype=dtype)

    # warmup
    for _ in range(10):
        _ = flash_forward(Q, K, V, is_causal=False)
    torch.cuda.synchronize()

    iters = 50
    t0 = time.time()
    for _ in range(iters):
        _ = flash_forward(Q, K, V, is_causal=False)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Triton forward: {(t1 - t0)*1000/iters:.2f} ms/iter")

    # reference (SxS materialization) — only for rough comparison
    scale = 1.0 / math.sqrt(D)
    t2 = time.time()
    for _ in range(iters):
        S_full = torch.einsum("bsd,btd->bst", Q.float(), K.float()) * scale
        P = torch.softmax(S_full, dim=-1)
        _ = torch.einsum("bst,btd->bsd", P, V.float())
    torch.cuda.synchronize()
    t3 = time.time()
    print(f"PyTorch ref   : {(t3 - t2)*1000/iters:.2f} ms/iter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="run a small benchmark")
    args = parser.parse_args()

    # Correctness: first non-causal, then causal (small shapes)
    check_small(causal=False, dtype=torch.bfloat16)
    check_small(causal=True,  dtype=torch.bfloat16)

    if args.bench:
        bench_small(dtype=torch.bfloat16)
