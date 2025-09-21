import triton, triton.language as tl
import torch
from torch.testing import assert_close



@triton.jit
def row_online_softmax_kernel(
    X_ptr, Y_ptr,
    M, N,                          # runtime sizes
    stride_xm, stride_xn,          # X strides (elements)
    stride_ym, stride_yn,          # Y strides (elements)
    BLOCK_N: tl.constexpr,         # tile width along columns
):
    row_id = tl.program_id(0)
    # safety if you ever overlaunch grid
    if row_id >= M:
        return

    # --------------------------------------------
    # Pass 1: compute row-wise max and sum(exp(x - max))
    # Using an online update so we can stream tiles.
    # --------------------------------------------
    m = -float("inf")              # running row max (scalar)
    l = 0.0                        # running row sum of exp(x - m) (scalar)

    col = 0
    while col < N:
        n = col + tl.arange(0, BLOCK_N)
        mask = n < N

        # load tile, upcast to fp32 for numerics
        x = tl.load(X_ptr + row_id * stride_xm + n * stride_xn,
                    mask=mask, other=-float("inf")).to(tl.float32)

        # tile max, then online update of global max / sum
        tmax = tl.max(x, axis=0)
        m_new = tl.maximum(m, tmax)                 # scalar
        # online softmax update for the partition function
        l = tl.exp(m - m_new) * l + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

        col += BLOCK_N

    # --------------------------------------------
    # Pass 2: write normalized outputs
    # y = exp(x - m) / l, tile by tile
    # --------------------------------------------
    inv_l = 1.0 / l
    col = 0
    while col < N:
        n = col + tl.arange(0, BLOCK_N)
        mask = n < N

        # reload tile (cheap & simple; avoids temp buffers)
        x_raw = tl.load(X_ptr + row_id * stride_xm + n * stride_xn,
                        mask=mask, other=-float("inf"))
        x = x_raw.to(tl.float32)
        y = tl.exp(x - m) * inv_l

        # store in the same dtype as the input/output tensor
        tl.store(Y_ptr + row_id * stride_ym + n * stride_yn,
                 y.to(x_raw.dtype), mask=mask)

        col += BLOCK_N




def row_softmax(x):
    y = torch.empty_like(x)
    M, N = x.shape
    sxm, sxn = x.stride()
    sym, syn = y.stride()
    BLOCK_N = 1024
    grid = (M,)
    row_online_softmax_kernel[grid](
        x, y, M, N, sxm, sxn, sym, syn,
        BLOCK_N=BLOCK_N,
    )
    return y

torch.manual_seed(0)
for dtype in (torch.float32, torch.bfloat16):
    for M, N in [(1,5),(3,17),(7,127),(13,1024),(4,4097)]:
        x = torch.randn(M,N, device="cuda", dtype=dtype)
        ref = torch.softmax(x.to(torch.float32), dim=-1).to(dtype)
        out = row_softmax(x)
        atol, rtol = (1e-6,1e-5) if dtype==torch.float32 else (1e-2,1e-2)
        assert_close(out, ref, atol=atol, rtol=rtol)
print("row-softmax OK ✅")