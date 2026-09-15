from __future__ import annotations

import argparse
import statistics
import time

import torch

from litedsps.model import LiteDSPSConfig, LiteDSPSMamba


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--visual-dim", type=int, default=136)
    p.add_argument("--audio-dim", type=int, default=25)
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 4096, 10000])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("profiling script expects CUDA")
    device = torch.device("cuda")
    model = LiteDSPSMamba(LiteDSPSConfig(visual_dim=args.visual_dim, audio_dim=args.audio_dim)).to(device).eval()
    for length in args.lengths:
        v = torch.randn(1, length, args.visual_dim, device=device)
        a = torch.randn(1, length, args.audio_dim, device=device)
        lens = torch.tensor([length], device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model(visual=v, visual_lengths=lens, audio=a, audio_lengths=lens)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(visual=v, visual_lengths=lens, audio=a, audio_lengths=lens)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        print({"length": length, "latency_ms": statistics.mean(times), "peak_mb": torch.cuda.max_memory_allocated() / 2**20})


if __name__ == "__main__":
    main()
