import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from torch import Tensor
from einops import einsum
import math
from jaxtyping import Float, Bool, Int
from timeit import default_timer as timer
from contextlib import nullcontext

def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.masked_fill(attention_scores, ~mask, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")



if __name__ == "__main__":
    torch.manual_seed(0)
    
    parser = argparse.ArgumentParser(description="Benchmark scaled dot product attention")
    parser.add_argument("-b", type=int, default=8, help="Batch size")
    parser.add_argument("-s", type=int, default=256, help="Sequence length")
    parser.add_argument("-d", type=int, default=768, help="attn Dimension")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--steps", type=int, default=100, help="Number of benchmark steps")
    parser.add_argument("--forward-only", action="store_true", help="Only do forward pass")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use, e.g., 'cuda' or 'cpu'")
    parser.add_argument("--dtype", type=str, choices=["float16","bfloat16","float32"], default="bfloat16", help="Data type to use")
    args = parser.parse_args()
    
    Q = torch.randn((args.b, args.s, args.d), device=args.device, dtype=getattr(torch, args.dtype))
    K = torch.randn((args.b, args.s, args.d), device=args.device, dtype=getattr(torch, args.dtype))
    V = torch.randn((args.b, args.s, args.d), device=args.device, dtype=getattr(torch, args.dtype))
    if not args.forward_only:
        Q.requires_grad = K.requires_grad = V.requires_grad = True
    else:
        Q.requires_grad = K.requires_grad = V.requires_grad = False
    
    mask = None
    
    if args.dtype == "float16":
        autocast_ctx = torch.amp.autocast('cuda',dtype=torch.float16)
    elif args.dtype == "bfloat16":
        autocast_ctx = torch.amp.autocast('cuda',dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext
        
    device = torch.device(args.device)
        
        
    # if backward, Measure how much memory is in use before the backward pass starts, and time 100 backward passes.
    # torch.cuda.synchronize() after each fwd/bwd to make sure we are measuring the right thing.
    
    # Warmup
    step = 0
    sum_fwd_time = 0.0
    sum_bwd_time = 0.0
    sum_mem_before_bwd = 0.0
    sum_peak_total_iter = 0.0
    sum_peak_bwd = 0.0
    
    try:
        for _ in range(args.warmup):
            with autocast_ctx:
                out = scaled_dot_product_attention(Q, K, V, mask)
            if not args.forward_only:
                out.sum().backward()
                Q.grad = K.grad = V.grad = None
        torch.cuda.reset_peak_memory_stats(device)
        
        for step in range(args.steps):
            torch.cuda.reset_peak_memory_stats(device)  # reset at iter start (optional)
            # ---- forward ----
            torch.cuda.synchronize()
            t0 = timer()
            with autocast_ctx:
                out = scaled_dot_product_attention(Q, K, V, mask)
            torch.cuda.synchronize()
            t1 = timer()
            mem_before_bwd = torch.cuda.memory_allocated(device)   # REQUIRED metric
            peak_total_iter = torch.cuda.max_memory_allocated(device)  # forward peak this iter

            # ---- backward ----
            torch.cuda.reset_peak_memory_stats(device)  # measure backward peak separately (optional)
            t2 = timer()
            out.sum().backward()
            Q.grad = K.grad = V.grad = None
            torch.cuda.synchronize()
            t3 = timer()
            peak_bwd = torch.cuda.max_memory_allocated(device)

            # print(f"[step {step+1}] fwd={ (t1-t0)*1e3:.2f} ms  bwd={ (t3-t2)*1e3:.2f} ms")
            # print(f"    mem_before_bwd={mem_before_bwd/2**20:.2f} MB (REQUIRED)")
            # print(f"    peak_forward_this_iter={peak_total_iter/2**20:.2f} MB (optional)")
            # print(f"    peak_backward_only={peak_bwd/2**20:.2f} MB (optional)")
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"OOM at step {step+1}, batch size {args.b}, sequence length {args.s}, dimension {args.d}, dtype {args.dtype}")
            torch.cuda.empty_cache()
        else:
            raise e

            
    if step == args.steps - 1:
        print(f"Success: batch size {args.b}, sequence length {args.s}, dimension {args.d}, dtype {args.dtype}")
        print(f"    fwd={ (t1-t0)*1e3:.2f} ms  bwd={ (t3-t2)*1e3:.2f} ms")
        if not args.forward_only:
            print(f"    mem_before_bwd={mem_before_bwd/2**20:.2f} MB (REQUIRED)")
            print(f"    peak_forward_this_iter={peak_total_iter/2**20:.2f} MB (optional)")
            print(f"    peak_backward_only={peak_bwd/2**20:.2f} MB (optional)")