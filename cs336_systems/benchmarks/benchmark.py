from cs336_basics.model import BasicsTransformerLM
import torch
import argparse
import json
import csv
import pandas as pd
from timeit import default_timer as timer
import contextlib
import torch.nn.functional as F
torch.manual_seed(0)

MODEL_FACTORY = {
  "small":  (768, 3072, 12, 12),
  "medium": (1024, 4096, 24, 16),
  "large":  (1280, 5120, 36, 20),
  "xl":     (1600, 6400, 48, 25),
  "2.7B":   (2560, 10240, 32, 32),
}


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM")
    parser.add_argument("--size", type=str, choices=["small","medium","large","xl","2.7B"], default="small", help="Model size, use it to construct BasicsTransformerLM")
    parser.add_argument("--ctx_len", type=int, default=1024, help="Context length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--steps", type=int, default=10, help="Number of benchmark steps")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile to compile the model")
    parser.add_argument("--do-forward-only", action="store_true", help="Only do forward pass")
    parser.add_argument("--outfile", type=str, default=None, help="Output json/csv to save the benchmark results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use, e.g., 'cuda' or 'cpu'")
    parser.add_argument("--dtype", type=str, choices=["float16","bfloat16","float32"], default="bfloat16", help="Data type to use")
    return parser.parse_args()

def build_model(size, vocab_size, ctx_len, device="cuda", compile=False):
    d_model, d_ff, n_layers, n_heads = MODEL_FACTORY[size]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=n_layers,
        num_heads=n_heads,
        context_length=ctx_len
    )
    
    model = model.to(device=device)
    if compile:
        model = torch.compile(model)
    
    return model

def rnd_generate_batch(batch_size, ctx_len, vocab_size, device="cuda"):
    ids = torch.randint(0, vocab_size, (batch_size, ctx_len), device=device, dtype=torch.long)
    x_in = ids[:, :-1]
    y_out = ids[:, 1:]
    return ids, x_in, y_out

def do_forward_step(model, x_in, autocast_ctx):
    with autocast_ctx():
        logits = model(x_in) # (B, L, V)
    # torch.cuda.synchronize()
    
def do_train_step(model, x_in, y_out, optimizer, autocast_ctx):
    optimizer.zero_grad()
    with autocast_ctx():
        logits = model(x_in) # (B, L, V)
        B, L, V = logits.shape
        loss = F.cross_entropy(logits.view(B*L, V), y_out.view(B*L))
    
    loss.backward()
    optimizer.step()
    # torch.cuda.synchronize()
    
    
def main():
    args = get_args()
    use_amp = args.dtype in ("bfloat16", "float16")
    amp_type = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = contextlib.nullcontext if not use_amp else torch.amp.autocast(device_type=args.device, dtype=amp_type)
    
    # Build model
    model = build_model(args.size, args.vocab_size, args.ctx_len, device=args.device, compile=args.compile)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas =(0.9, 0.99), weight_decay=0.0)
    
    # data
    ids, x_in, y_out = rnd_generate_batch(args.batch_size, args.ctx_len, args.vocab_size, device=args.device)
    
    # warmup
    for _ in range(args.warmup):
        if args.do_forward_only:
            do_forward_step(model, x_in, autocast_ctx)
        else:
            do_train_step(model, x_in, y_out, optimizer, autocast_ctx)
    
    t = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        start = timer()
        if args.do_forward_only:
            do_forward_step(model, x_in, autocast_ctx)
        else:
            do_train_step(model, x_in, y_out, optimizer, autocast_ctx)
        torch.cuda.synchronize()
        end = timer()
        t.append(end - start)
        
    mean_ms = 1000 * sum(t) / len(t)
    std_ms = 1000 * (sum((x - mean_ms/1000)**2 for x in t) / len(t))**0.5
    
    row = dict(
        size=args.size, ctx_len=args.ctx_len, batch_size=args.batch_size,
        dtype=args.dtype, forward_only=bool(args.forward_only), compile=bool(args.compile),
        warmup=args.warmup, steps=args.steps, mean_ms=round(mean_ms,2), std_ms=round(std_ms,2)
    )

    if args.outfile:
        import pathlib, csv
        p = pathlib.Path(args.outfile)
        header_needed = (not p.exists())
        with p.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if header_needed: w.writeheader()
            w.writerow(row)
    else:
        print(json.dumps(row, ensure_ascii=False))
        
        
if __name__ == "__main__":
    main()