import os
import json
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

# 定義圖片源目錄映射（相對路徑 -> 絕對路徑）
IMAGE_SOURCE_DIRS = {
    "data/real_data/Total-text/images": "/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/SynData/real_data/Total-text/images",
    "data/real_data/TextOCR/images": "/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/SynData/real_data/TextOCR/images",
    "data/real_data/SVT/images": "/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/SynData/real_data/SVT/images"
}

# 定義輸入JSONL目錄
INPUT_JSONL_DIR = "/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/SynData/real_data/combined_splits_by_ratio"

def convert_to_absolute_path(rel_path):
    """將相對路徑轉換為絕對路徑"""
    for rel_dir, abs_dir in IMAGE_SOURCE_DIRS.items():
        if rel_path.startswith(rel_dir):
            return rel_path.replace(rel_dir, abs_dir, 1)
    return None

def resize_image(image_path, target_ratio):
    """調整圖片大小到目標比例並覆蓋原始檔案"""
    try:
        with Image.open(image_path) as img:
            # 獲取當前寬高
            width, height = img.size
            current_ratio = height / width
            
            # 計算目標尺寸
            if current_ratio > target_ratio:
                # 圖片比目標更"高"，調整高度
                new_height = int(width * target_ratio)
                new_width = width
            else:
                # 圖片比目標更"寬"，調整寬度
                new_width = int(height / target_ratio)
                new_height = height
            
            # 調整大小並覆蓋原始檔案
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(image_path)
            return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def process_jsonl_file(jsonl_path):
    """處理單個jsonl文件並更新image_path"""
    # 從文件名中提取ratio
    base_name = os.path.basename(jsonl_path)
    ratio_str = base_name.split('_')[0]
    try:
        target_ratio = float(ratio_str)
    except ValueError:
        print(f"Invalid ratio in filename: {base_name}")
        return
    
    print(f"\nProcessing ratio: {target_ratio}")
    
    # 讀取原始jsonl文件
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    # 用於存儲更新後的行
    updated_lines = []
    processed_count = 0
    error_count = 0
    
    # 處理每張圖片
    for line in tqdm(lines, desc=f"Ratio {ratio_str}"):
        try:
            data = json.loads(line)
            image_path = data.get('image_path')
            if not image_path:
                updated_lines.append(line)
                error_count += 1
                continue
                
            # 轉換為絕對路徑
            abs_image_path = convert_to_absolute_path(image_path)
            if not abs_image_path or not os.path.exists(abs_image_path):
                print(f"Image not found: {image_path}")
                updated_lines.append(line)
                error_count += 1
                continue
                
            # 調整圖片大小（直接覆蓋原始檔案）
            if resize_image(abs_image_path, target_ratio):
                # 更新為絕對路徑
                data['image_path'] = abs_image_path
                updated_lines.append(json.dumps(data) + '\n')
                processed_count += 1
            else:
                updated_lines.append(line)
                error_count += 1
                
        except Exception as e:
            print(f"Error processing line: {e}")
            updated_lines.append(line)
            error_count += 1
    
    # 將更新後的內容寫回文件
    with open(jsonl_path, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"Processed: {processed_count} images, Errors: {error_count}")

def main():
    # 獲取所有jsonl文件
    jsonl_files = glob.glob(os.path.join(INPUT_JSONL_DIR, "*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {INPUT_JSONL_DIR}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    # 處理每個jsonl文件
    for jsonl_file in sorted(jsonl_files):
        process_jsonl_file(jsonl_file)
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    main()