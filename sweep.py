"""Run the clip grid: every (clip_low, clip_high) cell x seeds, in parallel.

Each run is its own subprocess with torch pinned to 1 thread, so N parallel
runs actually use N cores instead of fighting over all of them. Finished
runs (checkpoint.pt exists) are skipped, so the sweep is resumable — kill it
and rerun the same command any time.

The default grid is the experiment:
    clip_low  in {0.05, 0.1, 0.2, 0.3}
    clip_high in {0.1, 0.2, 0.3, 0.5}
with ent_coef = 0 everywhere. (0.2, 0.2) is the PPO baseline cell.

Example:
    uv run python sweep.py --env minatar/breakout --seeds 1 2 3 4 5 \
        --total-steps 3000000 --jobs 6 --out runs/breakout-grid
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_CLIP_LOWS = [0.05, 0.1, 0.2, 0.3]
DEFAULT_CLIP_HIGHS = [0.1, 0.2, 0.3, 0.5]


def run_one(job: tuple[list[str], Path]) -> int:
    cmd, out_dir = job
    if (out_dir / "checkpoint.pt").exists():
        print(f"skip (done): {out_dir}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (out_dir / "stdout.log").open("w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    status = "ok" if code == 0 else f"FAILED({code})"
    print(f"{status}: {out_dir}")
    return code


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True)
    p.add_argument("--clip-lows", type=float, nargs="+", default=DEFAULT_CLIP_LOWS)
    p.add_argument("--clip-highs", type=float, nargs="+", default=DEFAULT_CLIP_HIGHS)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--ent-coef", type=float, default=0.0)
    # MinAtar experiment regime (picked by probing, see GUIDE.md): the default
    # epochs=4 / lr=2.5e-4 leaves the clip engaged on <0.3% of samples, which
    # would make clip asymmetry inert. epochs=8 / lr=1e-3 clips ~2% per side
    # and learns faster.
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    jobs = []
    for cl, ch, seed in itertools.product(args.clip_lows, args.clip_highs, args.seeds):
        out_dir = Path(args.out) / f"cl{cl}_ch{ch}" / f"seed{seed}"
        cmd = [
            sys.executable,
            "train.py",
            "--env",
            args.env,
            "--clip-low",
            str(cl),
            "--clip-high",
            str(ch),
            "--ent-coef",
            str(args.ent_coef),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--total-steps",
            str(args.total_steps),
            "--seed",
            str(seed),
            "--torch-threads",
            "1",
            "--out",
            str(out_dir),
        ]
        jobs.append((cmd, out_dir))

    print(f"{len(jobs)} runs, {args.jobs} parallel")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(run_one, jobs))
    failed = sum(1 for c in codes if c != 0)
    print(f"done: {len(jobs) - failed} ok, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
