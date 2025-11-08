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
    base_images = generate_random_images(schedule, batch_size, device, seed=5)
    control_images = generate_random_images(schedule, batch_size, device, seed=11)

    print("===== Synthetic inputs =====")
    print(format_stats("var_images", base_images))
    print(format_stats("control_images", control_images))

    tokens_concat, input_tokens, quant_tokens = bsq_encode_tokens(vae, base_images, schedule, args, device)
    ctrl_tokens_concat, ctrl_input_tokens, ctrl_quant_tokens = bsq_encode_tokens(vae, control_images, schedule, args, device)

    print("\n===== BSQ latent tokens (shared VAE approach) =====")
    print(format_stats("tokens_concat", tokens_concat))
    print(format_stats("control_tokens_concat", ctrl_tokens_concat))
    for si, (var_tok, ctrl_tok) in enumerate(zip(input_tokens, ctrl_input_tokens)):
        if si == 0 or var_tok is None or ctrl_tok is None:
            continue
        print(format_stats(f"stage{si}_var_tokens", var_tok))
        print(format_stats(f"stage{si}_control_tokens", ctrl_tok))

    kv_tuple = build_text_condition(batch_size, args.Ct5, args.tlen, device, seed=19)
    conditioning = prepare_conditioning(model, kv_tuple, tokens_concat, schedule, need_drop=False)

    stats_shared = run_car_var_pipeline(
        model=model,
        conditioning=conditioning,
        x_tokens_wo_prefix=tokens_concat,
        scale_schedule=schedule,
        scenario_name="shared_vqvae_control",
        control_mode="shared_vae",
        control_input_tokens=ctrl_input_tokens,
        control_quant_tokens=ctrl_quant_tokens,
    )

    print("\n===== shared_vqvae_control statistics =====")
    for key in sorted(stats_shared):
        stats = stats_shared[key]
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
