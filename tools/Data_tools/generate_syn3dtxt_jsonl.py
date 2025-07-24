import os
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# 基礎路徑
BASE_DIR = "/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/SynData/Syn3DTxt_scene_easy"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
NORMALS_DIR = os.path.join(BASE_DIR, "normals")  # 用於 normal_path
METADATA_DIR = os.path.join(BASE_DIR, "metadatas")
OUTPUT_JSONL = os.path.join(BASE_DIR, "syn3dtxt_scene_easy.jsonl")

def get_image_size(image_path):
    """獲取圖片尺寸並計算高寬比"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return height / width
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 1.0  # 如果出錯，返回默認比例

def find_normal_path(image_path):
    """根據圖片路徑查找對應的 normal map 路徑"""
    try:
        # 獲取圖片相對路徑（相對於 IMAGES_DIR）
        rel_path = Path(image_path).relative_to(IMAGES_DIR)
        # 構建 normal map 路徑
        normal_path = Path(NORMALS_DIR) / rel_path
        return str(normal_path) if normal_path.exists() else ""
    except Exception:
        return ""

def load_metadata(image_path):
    """加載圖片的 metadata"""
    try:
        # 獲取圖片名稱（不帶擴展名）
        image_name = Path(image_path).stem
        metadata_path = os.path.join(METADATA_DIR, f"{image_name}.json")
        
        if not os.path.exists(metadata_path):
            print(f"Metadata not found for {image_name}")
            return None
            
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata for {image_path}: {e}")
        return None

def generate_jsonl():
    """生成 JSONL 文件"""
    # 獲取所有原始圖片文件（不包含 _dup 文件）
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    original_image_files = []
    for ext in image_extensions:
        original_image_files.extend([f for f in Path(IMAGES_DIR).rglob(f"*{ext}") if '_dup' not in str(f)])
    
    total_original = len(original_image_files)
    if total_original == 0:
        print("No original images found. Exiting...")
        return
    
    print(f"Found {total_original} original images")
    
    # 計算需要重複的次數
    TARGET_COUNT = 36000
    repeat_times = (TARGET_COUNT + total_original - 1) // total_original  # 向上取整
    print(f"Will generate {repeat_times}x the data to reach at least {TARGET_COUNT} samples")
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    with open(OUTPUT_JSONL, 'w') as f_out:
        # 循環處理重複數據
        for cycle in range(repeat_times):
            print(f"\nProcessing cycle {cycle + 1}/{repeat_times}...")
            
            # 處理當前週期的文件
            for img_path in tqdm(original_image_files, desc=f"Cycle {cycle + 1}", unit="images"):
                if processed_count >= TARGET_COUNT:
                    break
                    
                try:
                    img_path_str = str(img_path)
                    
                    # 如果是重複數據，創建新的文件名
                    if cycle > 0:
                        img_dir = os.path.dirname(img_path_str)
                        img_name, img_ext = os.path.splitext(os.path.basename(img_path_str))
                        new_img_name = f"{img_name}_dup{cycle}{img_ext}"
                        new_img_path = os.path.join(img_dir, new_img_name)
                        
                        # 複製圖片文件
                        if not os.path.exists(new_img_path):
                            shutil.copy2(img_path_str, new_img_path)
                        
                        # 更新路徑為新文件
                        img_path_str = new_img_path
                    
                    # 獲取圖片高寬比
                
                    # 獲取圖片高寬比
                    try:
                        h_div_w = get_image_size(img_path_str)
                    except Exception as e:
                        print(f"Error getting image size for {img_path_str}: {e}")
                        error_count += 1
                        continue
                    
                    # 查找 normal map 路徑
                    try:
                        normal_path = find_normal_path(img_path_str)
                    except Exception as e:
                        print(f"Error finding normal map for {img_path_str}: {e}")
                        normal_path = ""
                    
                    # 加載 metadata
                    try:
                        metadata = load_metadata(img_path_str)
                        if not metadata:
                            print(f"No metadata found for {img_path_str}")
                            error_count += 1
                            continue
                    except Exception as e:
                        print(f"Error loading metadata for {img_path_str}: {e}")
                        error_count += 1
                        continue
                
                    # 獲取 caption 和 text
                    try:
                        long_caption = metadata.get('caption', '')
                        text = metadata.get('text', '')
                        
                        # 如果是重複數據，可以添加後綴以區分
                        if cycle > 0:
                            suffix = f" (variant {cycle})"
                            if long_caption and not long_caption.endswith(suffix):
                                long_caption += suffix
                            if text and not text.endswith(suffix):
                                text += suffix
                        
                        # 構建輸出數據
                        data = {
                            "image_path": img_path_str,
                            "normal_path": normal_path,
                            "h_div_w": h_div_w,
                            "long_caption": long_caption,
                            "long_caption_type": "caption-RolmOCR",
                            "text": text,
                            "short_caption_type": "caption-RolmOCR"
                        }
                        
                        # 寫入 JSONL
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        processed_count += 1
                        
                        # 如果達到目標數量，立即返回
                        if processed_count >= TARGET_COUNT:
                            break
                            
                    except Exception as e:
                        print(f"Error processing metadata for {img_path_str}: {e}")
                        error_count += 1
                        continue
                        
                except Exception as e:
                    print(f"Error processing {img_path_str}: {e}")
                    error_count += 1
                    continue
                
                # 再次檢查是否達到目標數量（針對內部循環）
                if processed_count >= TARGET_COUNT:
                    break
            
            # 檢查是否達到目標數量（針對外部循環）
            if processed_count >= TARGET_COUNT:
                break
    
    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Original images: {total_original}")
    print(f"Target count: {TARGET_COUNT}")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors: {error_count}")
    if total_original > 0:
        success_rate = (processed_count / min(TARGET_COUNT, total_original * repeat_times)) * 100
        print(f"Success rate: {success_rate:.2f}%")
    print(f"\nOutput file: {OUTPUT_JSONL}")
    print("="*50)

if __name__ == "__main__":
    print("Starting to generate JSONL file...")
    generate_jsonl()
    print("\nDone!")
