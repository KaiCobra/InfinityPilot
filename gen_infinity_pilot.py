#!/usr/bin/env python3
"""
Batch-generation helper for InfinityPilot.

This script reuses the runtime defined in tools/run_infinity_pilot.py to
generate one or more images in a single process.  It keeps the model in
memory while iterating over prompts and seeds, making it convenient for
interactive experimentation or quick sweeps.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import torch

# Ensure repository root is on sys.path before importing helper utilities.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from tools.run_infinity_pilot import (  # noqa: E402
    _as_device,
    _build_control_tokens,
    _control_paths_from_args,
    _parse_list_arg,
    add_arguments,
    build_infinity_pilot,
    gen_one_img_pilot,
    load_tokenizer,
    load_visual_tokenizer,
)


def _read_prompt_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines()]
    return [line for line in lines if line]


def _resolve_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompt_file:
        prompts.extend(_read_prompt_file(args.prompt_file))
    if not prompts:
        raise ValueError("No prompts provided. Use --prompt and/or --prompt_file.")
    return prompts


def _resolve_seeds(args: argparse.Namespace) -> Sequence[int]:
    if args.seed_list:
        seeds = [int(x) for x in args.seed_list.split(",") if x.strip()]
        if not seeds:
            raise ValueError("Parsed an empty --seed_list.")
        if any(seed < 0 for seed in seeds):
            raise ValueError("Seeds must be non-negative integers.")
        return tuple(seeds)
    if args.shots_per_prompt <= 1:
        if args.seed < 0:
            raise ValueError("Seed must be non-negative.")
        return (args.seed,)
    if args.seed > 0:
        return tuple(args.seed + i for i in range(args.shots_per_prompt))
    rng = np.random.default_rng()
    return tuple(int(rng.integers(0, 2**31 - 1)) for _ in range(args.shots_per_prompt))


def _format_filename(pattern: str, prompt_idx: int, shot_idx: int, seed: int, global_idx: int) -> str:
    try:
        return pattern.format(
            prompt_idx=prompt_idx,
            shot_idx=shot_idx,
            seed=seed,
            idx=global_idx,
        )
    except KeyError as exc:
        raise ValueError(
            f"filename pattern missing placeholder: {exc}. "
            "Available placeholders: prompt_idx, shot_idx, seed, idx."
        ) from exc


def _choose_template(h_div_w: float) -> float:
    idx = int(np.argmin(np.abs(h_div_w_templates - h_div_w)))
    return float(h_div_w_templates[idx])


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate multiple images with InfinityPilot in a single run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_arguments(parser)
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="",
        help="Path to a text file with one prompt per line.",
    )
    parser.add_argument(
        "--shots_per_prompt",
        type=int,
        default=1,
        help="Number of seeds to sample per prompt when --seed_list is not provided.",
    )
    parser.add_argument(
        "--seed_list",
        type=str,
        default="",
        help="Comma-separated list of seeds (overrides --seed and --shots_per_prompt when set).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_pilot",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--filename_pattern",
        type=str,
        default="prompt{prompt_idx:02d}_shot{shot_idx:02d}_seed{seed}.jpg",
        help="Python format string for output filenames. Available placeholders: prompt_idx, shot_idx, seed, idx.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation when the target output file already exists.",
    )
    return parser


def _prepare_scale_schedule(args: argparse.Namespace) -> Iterable[tuple[int, int, int]]:
    template = _choose_template(args.h_div_w_template)
    scales = dynamic_resolution_h_w[template][args.pn]["scales"]
    return [(1, h, w) for (_, h, w) in scales]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prompts = _resolve_prompts(args)
    seeds = _resolve_seeds(args)
    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    device = _as_device()
    torch.manual_seed(int(seeds[0]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seeds[0]))

    cfg_values = _parse_list_arg(args.cfg)
    tau_values = _parse_list_arg(args.tau)

    _log("Loading text tokenizer and encoder")
    tokenizer, text_encoder = load_tokenizer(args.text_encoder_ckpt, device)

    _log("Loading visual tokenizer (VAE)")
    vae = load_visual_tokenizer(args, device)

    _log("Building InfinityPilot")
    infinity_pilot = build_infinity_pilot(args, vae, device)

    scale_schedule = list(_prepare_scale_schedule(args))
    control_tokens = _build_control_tokens(
        _control_paths_from_args(args),
        vae,
        scale_schedule,
        device,
        bool(args.apply_spatial_patchify),
    )

    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    total = 0
    for prompt_idx, prompt in enumerate(prompts):
        for shot_idx, seed in enumerate(seeds):
            filename = _format_filename(args.filename_pattern, prompt_idx, shot_idx, seed, total)
            save_path = output_dir / filename
            if args.skip_existing and save_path.exists():
                _log(f"Skipping existing file: {save_path}")
                total += 1
                continue

            _log(f"Generating prompt #{prompt_idx} (shot {shot_idx}, seed {seed}): {prompt[:80]}...")
            start = time.time()

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            image = gen_one_img_pilot(
                infinity_pilot,
                vae,
                tokenizer,
                text_encoder,
                prompt,
                scale_schedule=scale_schedule,
                cfg_list=cfg_values,
                tau_list=tau_values,
                device=device,
                negative_prompt=args.negative_prompt,
                g_seed=seed,
                cfg_sc=args.cfg_sc,
                cfg_exp_k=args.cfg_exp_k,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                top_k=args.top_k,
                top_p=args.top_p,
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=bool(args.enable_positive_prompt),
                control_tokens=control_tokens,
                amp_dtype=amp_dtype,
            )

            cv2.imwrite(str(save_path), image.cpu().numpy())

            elapsed = time.time() - start
            _log(f"Saved {save_path} (elapsed {elapsed:.2f}s)")
            total += 1

    _log(f"Completed {total} generations")


if __name__ == "__main__":
    main()


"""
python gen_infinity_pilot.py --pn 1M --model_path <path> --text_encoder_ckpt <path> --vae_path <path> --prompt "a city street at dusk"
"""