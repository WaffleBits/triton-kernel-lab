"""Hero chart of Triton vs torch.compile vs eager p50 latency from a committed artifact."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
COLORS = {"triton": "#58a6ff", "torch_compile": "#f778ba", "pytorch_eager": "#8b949e"}
LABELS = {"triton": "Triton", "torch_compile": "torch.compile", "pytorch_eager": "PyTorch eager"}

# Representative FP16 shapes shown in the hero chart.
SELECT = [("rmsnorm", [512, 4096]), ("rmsnorm", [2048, 4096]),
          ("swiglu", [512, 4096]), ("swiglu", [2048, 4096])]
DTYPE = "float16"


def main(artifact_path: str, out_path: str) -> None:
    data = json.loads(Path(artifact_path).read_text())
    gpu = data["system"]["gpu"]
    by_key = {(c["kernel"], tuple(c["shape"]), c["dtype"]): c for c in data["cases"]}

    groups = []
    for kernel, shape in SELECT:
        case = by_key[(kernel, tuple(shape), DTYPE)]
        sp = case["torch_compile"]["p50_ms"] / case["triton"]["p50_ms"]
        label = f"{kernel.upper()}\n{shape[0]} x {shape[1]}\n{sp:.2f}x vs torch.compile"
        groups.append((label, case))

    backends = ["triton", "torch_compile", "pytorch_eager"]
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    n = len(groups)
    width = 0.26
    xs = list(range(n))
    for i, b in enumerate(backends):
        offs = (i - 1) * width
        vals = [g[1][b]["p50_ms"] for g in groups]
        bars = ax.bar([x + offs for x in xs], vals, width=width,
                      color=COLORS[b], label=LABELS[b])
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", color=FG, fontsize=8)

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], color=FG, fontsize=9.5)
    ax.set_ylabel("p50 latency (ms, log scale, lower is better)", color=FG, fontsize=11)
    ax.set_title(f"Triton kernel p50 latency on {gpu}\nFP16, cache-cold, 500 timed samples per case",
                 color=FG, fontsize=13, pad=12)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, framealpha=1.0, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "artifacts/rtx-5070-ti-rmsnorm-swiglu.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "assets/rmsnorm-swiglu-p50.png"
    main(src, dst)
