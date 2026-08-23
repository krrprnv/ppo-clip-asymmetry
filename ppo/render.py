"""Render two trained agents side by side as a GIF, with each policy's live
action distribution and entropy under its board.

The point of the figure: entropy differences stop being a number and become
visible behavior — a min-entropy policy's bars collapse to a single action,
a max-entropy policy's stay spread. Actions are SAMPLED from each policy
(this is the behavior policy, the same one every entropy figure measures).

Defaults reproduce the README GIF from the full-grid checkpoints:

    uv run python -m ppo.render

If ffmpeg is on PATH the GIF is palette-optimized (~2-3x smaller); otherwise
the raw PIL GIF is kept.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.distributions import Categorical

from ppo.core import make_agent
from ppo.envs import MinAtarGym

CELL, PAD_CELL = 36, 3
BOARD = 10 * CELL
PAGE, CARD, CARD_EDGE = (14, 14, 16), (23, 23, 26), (42, 42, 48)
INK, INK_DIM = (232, 232, 226), (150, 150, 145)
BLUE, ORANGE, AQUA, YELLOW = (
    (42, 120, 214),
    (235, 104, 52),
    (27, 175, 122),
    (237, 161, 0),
)
BOARD_BG = (18, 18, 21)
ACTION_NAMES = ["n", "l", "u", "r", "d", "f"]


def _font(size: int, mono: bool = False):
    candidates = (
        ["/System/Library/Fonts/Menlo.ttc"]
        if mono
        else ["/System/Library/Fonts/Helvetica.ttc"]
    ) + ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _blend(c, bg, a):
    return tuple(int(a * x + (1 - a) * y) for x, y in zip(c, bg))


def _load_agent(run_dir: Path):
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = json.loads((run_dir / "config.json").read_text())
    game = cfg["env_id"].removeprefix("minatar/")
    env = MinAtarGym(game)
    agent = make_agent(env.obs_shape, env.n_actions)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent, env, cfg


def _simulate(run_dir: Path, env_seed: int, steps: int):
    agent, env, cfg = _load_agent(Path(run_dir))
    obs = env.reset(seed=env_seed)
    seq, score = [], 0.0
    for _ in range(steps):
        with torch.no_grad():
            logits, _ = agent.logits_and_value(torch.as_tensor(obs)[None])
            dist = Categorical(logits=logits)
            probs = dist.probs[0].numpy().copy()
            entropy = float(dist.entropy()[0])
            action = int(dist.sample())
        seq.append((obs.copy(), probs, entropy, score))
        obs, r, term, trunc = env.step(action)
        score += r
        if term or trunc:
            obs, score = env.reset(), 0.0
    subtitle = f"clip_low {cfg['clip_low']:.2f} · clip_high {cfg['clip_high']:.2f}"
    return seq, subtitle


def _draw_board(draw, x0, y0, obs):
    draw.rounded_rectangle(
        [x0 - 6, y0 - 6, x0 + BOARD + 6, y0 + BOARD + 6], radius=10, fill=BOARD_BG
    )
    layers = [(2, _blend(YELLOW, BOARD_BG, 0.30)), (3, AQUA), (0, BLUE), (1, ORANGE)]
    for c, color in layers:
        if c >= obs.shape[0]:
            continue
        for r, cc in zip(*np.nonzero(obs[c] > 0.5)):
            px, py = x0 + cc * CELL, y0 + r * CELL
            draw.rounded_rectangle(
                [
                    px + PAD_CELL,
                    py + PAD_CELL,
                    px + CELL - PAD_CELL,
                    py + CELL - PAD_CELL,
                ],
                radius=7,
                fill=color,
            )


def _draw_bars(draw, x0, y0, probs, entropy, fonts):
    bar_w, gap, h_max = 40, 10, 64
    baseline = y0 + h_max
    for i, p in enumerate(probs):
        bx = x0 + i * (bar_w + gap)
        h = max(3, int(p * h_max))
        draw.rounded_rectangle(
            [bx, baseline - h, bx + bar_w, baseline],
            radius=4,
            fill=_blend(BLUE, CARD, 0.35 + 0.65 * min(1.0, p * 2)),
        )
        draw.text(
            (bx + bar_w / 2, baseline + 8),
            ACTION_NAMES[i],
            font=fonts["s"],
            fill=INK_DIM,
            anchor="ma",
        )
    end_x = x0 + len(probs) * bar_w + (len(probs) - 1) * gap
    draw.line([x0 - 2, baseline + 1, end_x + 2, baseline + 1], fill=CARD_EDGE, width=1)
    draw.text((end_x + 26, baseline - 40), "H", font=fonts["m"], fill=INK_DIM)
    draw.text(
        (end_x + 26, baseline - 20), f"{entropy:.2f}", font=fonts["mono"], fill=INK
    )


def _draw_panel(draw, x0, y0, title, subtitle, state, fonts):
    obs, probs, entropy, score = state
    w, h = BOARD + 32, 52 + BOARD + 12 + 100 + 10
    draw.rounded_rectangle(
        [x0, y0, x0 + w, y0 + h], radius=14, fill=CARD, outline=CARD_EDGE, width=1
    )
    draw.text((x0 + 18, y0 + 12), title, font=fonts["l"], fill=INK)
    draw.text(
        (x0 + w - 18, y0 + 16),
        f"score {score:.0f}",
        font=fonts["mono"],
        fill=INK_DIM,
        anchor="ra",
    )
    draw.text((x0 + 18, y0 + 36), subtitle, font=fonts["s"], fill=INK_DIM)
    _draw_board(draw, x0 + 16, y0 + 58, obs)
    _draw_bars(draw, x0 + 22, y0 + 58 + BOARD + 22, probs, entropy, fonts)


def _optimize(path: Path, fps: int) -> None:
    if not shutil.which("ffmpeg"):
        return
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    filters = (
        f"fps={fps},split[s0][s1];[s0]palettegen=max_colors=64[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vf", filters, "-loop", "0", str(tmp_path)],
        capture_output=True,
    )
    if result.returncode == 0 and tmp_path.stat().st_size < path.stat().st_size:
        tmp_path.replace(path)
    else:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", default="runs/breakout-grid/cl0.3_ch0.1/seed1")
    p.add_argument("--right", default="runs/breakout-grid/cl0.05_ch0.5/seed1")
    p.add_argument("--left-title", default="min-entropy policy")
    p.add_argument("--right-title", default="max-entropy policy")
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--out", default="figures/entropy_comparison.gif")
    args = p.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    fonts = {
        "l": _font(21),
        "m": _font(15),
        "s": _font(13),
        "mono": _font(15, mono=True),
    }
    sides = [
        (args.left_title, *_simulate(args.left, args.env_seed, args.steps)),
        (args.right_title, *_simulate(args.right, args.env_seed, args.steps)),
    ]

    panel_w, panel_h = BOARD + 32, 52 + BOARD + 12 + 100 + 10
    margin, gap = 18, 22
    cw, ch = 2 * panel_w + gap + 2 * margin, panel_h + 2 * margin + 26
    frames = []
    for t in range(args.steps):
        canvas = Image.new("RGB", (cw, ch), PAGE)
        draw = ImageDraw.Draw(canvas)
        for i, (title, seq, subtitle) in enumerate(sides):
            _draw_panel(
                draw,
                margin + i * (panel_w + gap),
                margin,
                title,
                subtitle,
                seq[t],
                fonts,
            )
        draw.text(
            (cw // 2, ch - 20),
            "same PPO, same seed, ent_coef = 0 — only the clip bounds differ",
            font=fonts["s"],
            fill=INK_DIM,
            anchor="ma",
        )
        if args.scale != 1.0:
            canvas = canvas.resize(
                (int(cw * args.scale), int(ch * args.scale)), Image.LANCZOS
            )
        frames.append(canvas)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    _optimize(out, args.fps)
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
