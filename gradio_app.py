import os
import argparse
import torch
import cv2
import numpy as np
import gradio as gr
import random
from tools.run_infinity import *
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.models.infinity import dynamic_resolution_h_w, h_div_w_templates

# Set device
torch.cuda.set_device(0)

# Default paths
DEFAULT_MODEL_PATH = './weights/mm_2b.pth'
DEFAULT_VAE_PATH = './weights/infinity_vae_d32_reg.pth'
DEFAULT_TEXT_ENCODER_CKPT = "./weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001"
DEFAULT_MASK_PATH = 'walmart.jpg'

# Default strength values
DEFAULT_STRENGTH = [0.35, 0.94, 0.97, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Initialize global variables
vae = None
infinity = None
text_tokenizer = None
text_encoder = None


def initialize_models(model_path, vae_path, text_encoder_ckpt):
    """Initialize all necessary models"""
    global vae, infinity, text_tokenizer, text_encoder
    
    # Create args namespace
    args = argparse.Namespace(
        pn='1M',
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_2b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='./cache',
        checkpoint_type='torch',
        enable_model_cache=True,
        seed=0,
        bf16=1,
        save_file='output.jpg',
        noise_apply_layers=1,
        noise_apply_requant=1,
        noise_apply_strength=0.01,
    )
    
    # Load models
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    
    return "Models loaded successfully!"


def generate_image(
    prompt,
    mask_path,
    strength_values,
    cfg_scale=4.0,
    tau=0.3,
    seed=None,
    progress=gr.Progress()
):
    """Generate image based on the given parameters"""
    global vae, infinity, text_tokenizer, text_encoder
    
    if seed is None or seed < 0:
        seed = random.randint(0, 10000)
    
    # Parse strength values
    try:
        strength = [float(x) for x in strength_values.split(',')]
        if len(strength) != 12:
            return None, "Error: Please provide exactly 12 strength values separated by commas"
    except ValueError:
        return None, "Error: Invalid strength values. Please provide 12 numbers separated by commas"
    
    # Set up scale schedule based on aspect ratio (1:1 by default)
    h_div_w = 1.0  # Default aspect ratio (height/width = 1.0 for square)
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template]['1M']['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    
    # Set up mask and strength
    # Create args object with all required attributes for BitwiseSelfCorrection
    class Args:
        def __init__(self):
            self.noise_apply_layers = 1
            self.noise_apply_requant = 1
            self.noise_apply_strength = 0.01
            self.apply_spatial_patchify = 0
            self.debug_bsc = False
    
    bitwise_self_correction = BitwiseSelfCorrection(vae, Args())
    infinity.setup_mask_processor(vae, scale_schedule, bitwise_self_correction)
    infinity.set_mask(
        mask_path=mask_path,
        scale_idx=list(range(12)),
        alpha=0.3,
        strength=strength
    )
    
    # Generate image
    try:
        # Create a list of cfg_scale and tau values for each scale
        cfg_list = [float(cfg_scale)] * len(scale_schedule)
        tau_list = [float(tau)] * len(scale_schedule)
        
        generated_image = gen_one_img(
            infinity_test=infinity,
            vae=vae,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            prompt=prompt,
            g_seed=seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=cfg_list,
            tau_list=tau_list,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[0],  # Ensure this is a list
            vae_type=32,
            sampling_per_bits=1,
            enable_positive_prompt=0,
        )
        
        # Save the generated image
        output_path = 'generated_output.jpg'
        cv2.imwrite(output_path, generated_image.cpu().numpy())
        return output_path, f"Image generated successfully with seed: {seed}"
    
    except Exception as e:
        return None, f"Error generating image: {str(e)}"


def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="Scene Text Generation") as demo:
        gr.Markdown("# Scene Text Generation with Infinity")
        gr.Markdown("Generate images with custom text using the Infinity model")
        
        # Default prompts for each image
        walmart_prompt = '''Render the text "Target" in the following image:
This image shows the exterior of a large retail store on a sunny day. The building has a modern, clean architectural design with neutral tones of white and gray. A prominent feature is the bright blue rectangular sign centered above the entrance, which displays the name "Target" in bold, white letters. Next to the text is a simple, iconic yellow spark-like symbol that accompanies the branding.
Below the sign is a covered entranceway supported by steel rods and beams, providing shade and protection to customers entering or exiting the store. There are several sets of glass sliding doors, and shopping carts can be seen inside and just outside the entrance. A row of bollards stands in front of the store, ensuring pedestrian safety.
To the left of the entrance, a few people are walking or pushing shopping carts, and the pavement includes clearly marked pedestrian pathways with blue lines. Trees with green leaves frame the top of the image, suggesting it's a pleasant, mild-weather day. In the background, additional buildings and greenery are visible, giving the impression that the store is located in a well-developed suburban or urban area.'''
        
        wall_street_prompt = '''Render the text "Wall Street" in the following image:
The image captures a close-up view of a street sign at the iconic intersection of Wall Street and Broad Street in New York City. The "WALL ST" sign is prominently displayed in the center of the image, mounted on a metal pole along with another sign below it labeled "BROAD ST". The Wall Street sign includes the range “11–21 →” and features a small grayscale image of George Washington's statue standing in front of the Federal Hall, symbolizing the financial significance of the area.
A yellow streetlight, mounted above the signs, casts a warm glow in the overcast lighting, giving the scene a moody urban feel. In the background, towering skyscrapers with a mix of classical and modern architecture dominate the view, emphasizing the dense and historic character of the Financial District in Manhattan. The image has a gritty, realistic texture, possibly taken during a light drizzle, as suggested by the subtle wet sheen on the pole and light specks in the air.
'''
        
        # Create the prompt textbox first
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=5,
                    placeholder="Describe the scene and text you want to generate...",
                    value=walmart_prompt
                )
        
        # Display both images side by side with radio buttons for selection
        with gr.Row():
            with gr.Column():
                selected_image = gr.Radio(
                    ["Walmart Storefront", "Wall Street"],
                    label="Select Default Scene",
                    value="Walmart Storefront"
                )
                
                # Display the selected image
                image_display = gr.Gallery(
                    label="Selected Scene",
                    columns=1,
                    height=400,  # 設置固定高度為 400 像素
                    object_fit="contain"  # 保持圖片比例
                )
                
                # Function to update the displayed image and prompt
                def update_display(choice):
                    if choice == "Walmart Storefront":
                        return ["walmart.jpg"], walmart_prompt
                    else:
                        return ["wall_street.jpg"], wall_street_prompt
                
                # Connect radio button to update function
                selected_image.change(
                    fn=update_display,
                    inputs=selected_image,
                    outputs=[image_display, prompt]
                )
        
        with gr.Row():
            with gr.Column():
                # Other inputs will go here
                pass
                
                mask_upload = gr.File(
                    label="Upload Mask Image",
                    type="filepath",
                    file_count="single",
                    file_types=["image"],
                    value=DEFAULT_MASK_PATH
                )
                
                strength_values = gr.Textbox(
                    label="Strength Values (comma-separated, 12 values)",
                    value=", ".join(map(str, DEFAULT_STRENGTH)),
                    placeholder="0.35, 0.94, 0.97, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0"
                )
                
                with gr.Row():
                    cfg_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=10.0,
                        value=4.0,
                        step=0.1
                    )
                    
                    tau = gr.Slider(
                        label="Tau",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.3,
                        step=0.01
                    )
                    
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )
                
                generate_btn = gr.Button("Generate Image", variant="primary")
                
            with gr.Column():
                # Outputs
                output_image = gr.Image(label="Generated Image")
                output_text = gr.Textbox(label="Status", interactive=False)
        
        # Event handlers
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                mask_upload,
                strength_values,
                cfg_scale,
                tau,
                seed
            ],
            outputs=[output_image, output_text]
        )
    
    return demo


if __name__ == "__main__":
    # Initialize models
    print("Initializing models...")
    status = initialize_models(
        DEFAULT_MODEL_PATH,
        DEFAULT_VAE_PATH,
        DEFAULT_TEXT_ENCODER_CKPT
    )
    print(status)
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", share=True)
