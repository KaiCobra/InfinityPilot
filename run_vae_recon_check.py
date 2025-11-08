# run_vae_recon_check.py
import torch
from PIL import Image
import torchvision.transforms as T
from infinity.models.bsq_vae.vae import vae_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load("weights/infinity_vae_d32_reg.pth", map_location='cpu')  # 你已經 load 過 d
# build vae_local -- use same defaults as train script (non-patchify)
codebook_dim = 32
codebook_size = 2 ** codebook_dim
vae_local = vae_model(ckpt, schedule_mode="dynamic", codebook_dim=codebook_dim, codebook_size=codebook_size,
                      patch_size=16, encoder_ch_mult=[1,2,4,4,4], decoder_ch_mult=[1,2,4,4,4], test_mode=True).to(device)
vae_local.eval()

# load an image and preprocess to [-1,1]
img_pil = Image.open("output/output5.png").convert("RGB")  # choose a sample from your dataset
transform = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])  # 注意大小可調
img = transform(img_pil).unsqueeze(0).to(device)  # [1,3,H,W]

with torch.no_grad():
    # try repo helper used elsewhere: encode_for_raw_features -> then reconstruct via quantizer/decoder
    try:
        # use same scale schedule logic as repo for a single image: get vae_scale_schedule for your pn if needed
        # Here we just call encode_for_raw_features to get raw features and then attempt decode path
        raw_features, _, _ = vae_local.encode_for_raw_features(img, scale_schedule=[(1, img.shape[-2]//16, img.shape[-1]//16)])
        # if encode_for_raw_features not present/compatible, try vae_local.encoder(img) / vae_local.decode path
    except Exception as e:
        print("encode_for_raw_features failed, try fallback:", e)

    # Simple end-to-end try: if vae_local has direct encode/decode helpers
    try:
        enc = vae_local.encode(img)   # some implementations provide encode; if not, skip
        dec = vae_local.decode(enc)
        recon = dec  # [1,3,H,W] in [-1,1] expected
    except Exception:
        # fallback: if encode/decode not present, try simpler path used in repo: (this part may need minor tweaks for your local vae API)
        print("Fallback: calling vae_local.decoder on summed codes requires token pipeline; skip here.")

# Save recon if produced
if 'recon' in locals():
    import torchvision
    torchvision.utils.save_image((recon.clamp(-1,1)+1)/2.0, "vae_recon_debug.png")
    print("Saved vae_recon_debug.png")
else:
    print("No direct recon produced; use repo encode->quantizer->decoder path (I can help craft exact calls if encode() not available).")