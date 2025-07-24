from PIL import Image
import torch
from torch.nn import functional as F
import numpy as np
import torchvision.transforms as transforms
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
class MaskFeatureProcessor:
    def __init__(self, vae, scale_schedule, device='cuda:0', other_args=None, bitwise_self_correction=None):
        self.vae = vae
        self.scale_schedule = scale_schedule
        self.device = device
        self.apply_spatial_patchify = 0
        self.bitwise_self_correction = bitwise_self_correction
        self.image_size = 1024
    
    def set_mask(self, mask_path, si):
        """設置 mask 圖像並編碼為特徵"""
        mask = Image.open(mask_path).convert('RGB')
        mask = mask.resize((1024, 1024), Image.BICUBIC)
        # Only for black-white txt mask
        # -----------------------------------------------------------------------
        # arr = np.array(mask)  # shape (1024,1024,3), dtype=uint8
        # # 3. 判斷是否「非黑」(任一通道>0) -> 白，否則黑
        # #    keepdims=True 以保留第三維度
        # bw2d = np.where(np.any(arr > 0, axis=2), 255, 0).astype(np.uint8)
        # # 4. 從陣列回到 PIL Image，並確保為 RGB 三通道
        # mask = Image.fromarray(bw2d, mode='L').convert('RGB')
        # -----------------------------------------------------------------------
        black_img = Image.new(mode=mask.mode, size=mask.size, color=(0, 0, 0))
        # 2. 調整大小（如果需要）
        if self.image_size is not None:
            mask = transforms.Resize(self.image_size, interpolation=Image.BICUBIC)(mask)
            black_img = transforms.Resize(self.image_size, interpolation=Image.BICUBIC)(black_img)
        
        # 3. 轉換為 tensor 並歸一化
        mask = transforms.ToTensor()(mask)  # [0, 1]
        mask = mask * 2 - 1  # [-1, 1]
        mask = mask.unsqueeze(0).to(self.device) 
        black_img = transforms.ToTensor()(black_img)  # [0, 1]
        black_img = black_img * 2 - 1  # [-1, 1]
        black_img = black_img.unsqueeze(0).to(self.device) 
        
        # 5. 處理空間 patchify
        if self.apply_spatial_patchify: # False
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in self.scale_schedule]
        else:
            vae_scale_schedule = self.scale_schedule
        
        # 6. 使用 VAE 編碼
        with torch.no_grad():
            # 6.1 處理 mask 將白色區域轉為0（灰）------- only for text mask ------
            # is_black = (mask < -0.9).all(dim=1, keepdim=True)
            # mask = torch.where(is_black, mask, torch.zeros_like(mask))
            # -----------------------------------------------------------

            raw_features, _, _ = self.vae.encode_for_raw_features(mask, scale_schedule=vae_scale_schedule)
            # raw_features - min: -3.464933, max: 3.103741, mean: 0.136124, std: 0.604392
            raw_features_black, _, _ = self.vae.encode_for_raw_features(black_img, scale_schedule=vae_scale_schedule)
            # raw_features_black - min: -1.211359, max: 1.325547, mean: -0.050055, std: 0.569620
            f = raw_features.unsqueeze(2)
            f_black = raw_features_black.unsqueeze(2)

            diff_mask = (f != f_black)
            f_zeroed = f.clone()
            f_zeroed[diff_mask] = 0
        #     # 6.2 使用 bitwise_self_correction 進行量化
        #     x_BLC_wo_prefix, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(
        #         vae_scale_schedule, mask, raw_features, self.device)
            
        #     idx_Bld = gt_ms_idx_Bl[si]
        #     pn = vae_scale_schedule[si]
        #     idx_Bld = idx_Bld.reshape(1, pn[1], pn[2], -1)
        #     idx_Bld = idx_Bld.unsqueeze(1) # [1,1,n,n,32]
        #     # 6.3 轉換為特徵
        #     mask = self.vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')
        
        # self.mask = mask
        return f, f_zeroed

    def apply_mask_to_features(self, features, mask_features, method='weighted', alpha=0.3):
        """將 mask 特徵應用到當前特徵上"""
        if method == 'weighted':
            return features * (1 - alpha) + mask_features * alpha
        elif method == 'residual':
            return features + mask_features * alpha
        elif method == 'gated':
            # 使用 sigmoid 作為閘門
            gate = torch.sigmoid(mask_features)
            return features * (1 - gate) + mask_features * gate
        else:
            raise ValueError(f"Unsupported mask method: {method}")