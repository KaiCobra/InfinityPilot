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
    base_images = generate_random_images(schedule, batch_size, device, seed=13)
    control_images = generate_random_images(schedule, batch_size, device, seed=21)

    print("===== Synthetic inputs =====")
    print(format_stats("var_images", base_images))
    print(format_stats("control_images", control_images))

    tokens_concat, input_tokens, quant_tokens = bsq_encode_tokens(vae, base_images, schedule, args, device)
    ctrl_tokens_concat, ctrl_input_tokens, ctrl_quant_tokens = bsq_encode_tokens(vae, control_images, schedule, args, device)

    print("\n===== BSQ latent tokens (VAR path) =====")
    print(format_stats("tokens_concat", tokens_concat))
    for si, tokens in enumerate(input_tokens):
        if si == 0 or tokens is None:
            continue
        print(format_stats(f"stage{si}_input_tokens", tokens))
    print(format_stats("control_tokens_concat", ctrl_tokens_concat))

    kv_tuple = build_text_condition(batch_size, args.Ct5, args.tlen, device, seed=37)
    conditioning = prepare_conditioning(model, kv_tuple, tokens_concat, schedule, need_drop=False)

    control_tensors = model.prepare_control_for_scales(control_images, schedule)
    stats_conv = run_car_var_pipeline(
        model=model,
        conditioning=conditioning,
        x_tokens_wo_prefix=tokens_concat,
        scale_schedule=schedule,
        scenario_name="current_conv_control",
        control_mode="conv",
        control_tensors=control_tensors,
    )

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

    print("\n===== current_conv_control statistics =====")
    for key in sorted(stats_conv):
        stats = stats_conv[key]
        print(
            f"{key:<40} "
            f"mean={stats['mean']:+.6f}  "
            f"std={stats['std']:+.6f}  "
            f"min={stats['min']:+.6f}  "
            f"max={stats['max']:+.6f}"
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
