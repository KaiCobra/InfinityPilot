import torch

from .common import (
    build_infinity_pilot,
    bsq_encode_tokens,
    build_text_condition,
    create_default_args,
    format_stats,
    generate_random_images,
    get_scale_schedule,
    prepare_conditioning,
    run_car_var_pipeline,
)


def main() -> None:
    args = create_default_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this analysis.")
    device = torch.device(args.device)

    model, vae, _ = build_infinity_pilot(args, device)
    schedule = get_scale_schedule(args)

    batch_size = 2
    base_images = generate_random_images(schedule, batch_size, device, seed=29)
    print("===== Synthetic inputs (Infinity only) =====")
    print(format_stats("var_images", base_images))

    tokens_concat, input_tokens, _ = bsq_encode_tokens(vae, base_images, schedule, args, device)
    print("\n===== BSQ latent tokens =====")
    print(format_stats("tokens_concat", tokens_concat))
    for si, tokens in enumerate(input_tokens):
        if si == 0 or tokens is None:
            continue
        print(format_stats(f"stage{si}_tokens", tokens))

    kv_tuple = build_text_condition(batch_size, args.Ct5, args.tlen, device, seed=31)
    conditioning = prepare_conditioning(model, kv_tuple, tokens_concat, schedule, need_drop=False)

    stats_infinity = run_car_var_pipeline(
        model=model,
        conditioning=conditioning,
        x_tokens_wo_prefix=tokens_concat,
        scale_schedule=schedule,
        scenario_name="infinity_only",
        control_mode=None,
    )

    print("\n===== infinity_only statistics =====")
    for key in sorted(stats_infinity):
        stats = stats_infinity[key]
        print(
            f"{key:<40} "
            f"mean={stats['mean']:+.6f}  "
            f"std={stats['std']:+.6f}  "
            f"min={stats['min']:+.6f}  "
            f"max={stats['max']:+.6f}"
        )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
