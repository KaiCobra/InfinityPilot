import os
import json
import glob
from PIL import Image
import numpy as np

# 定義可用的 h_div_w templates
h_div_w_templates = np.array([
    1.000, 1.250, 1.333, 1.500, 1.750, 2.000, 2.500, 3.000,
    0.500, 0.667, 0.714, 0.667, 0.571, 0.500, 0.400, 0.333
])

def find_nearest_template(h_div_w):
    """找到最接近的 h_div_w template"""
    # 計算與所有模板的差異
    diffs = np.abs(h_div_w_templates - h_div_w)
    # 找到最小差異的索引
    min_idx = np.argmin(diffs)
    # 返回最接近的模板值
    return h_div_w_templates[min_idx]

def create_output_dir():
    output_dir = '/home/avlab/SceneTxtVAR/data/real_data/combined_splits_by_ratio'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def verify_batches(output_dir):
    """驗證所有 JSONL 文件的批次數"""
    print(f"\nVerifying batches in {output_dir}")
    total_batches = 0
    
    for filename in os.listdir(output_dir):
        if not filename.endswith('.jsonl'):
            continue
            
        filepath = os.path.join(output_dir, filename)
        
        # 從文件名中提取 h_div_w 和 num_samples
        # 移除 .jsonl 擴展名
        base_name = filename[:-5]
        # 分割 h_div_w 和 num_samples
        h_div_w, num_samples_str = base_name.split('_')
        # 確保 num_samples_str 是純數字
        num_samples_str = ''.join(filter(str.isdigit, num_samples_str))
        num_samples = int(num_samples_str)
        
        # 驗證文件中的實際樣本數量
        actual_num_samples = 0
        with open(filepath, 'r') as f:
            for line in f:
                actual_num_samples += 1
        
        # 驗證樣本數是否相符
        if actual_num_samples != num_samples:
            print(f"Warning: File {filename} has {actual_num_samples} samples but filename indicates {num_samples}")
            # 重命名文件以反映實際樣本數量
            new_filename = f"{h_div_w}_{actual_num_samples:07d}.jsonl"
            os.rename(filepath, os.path.join(output_dir, new_filename))
            print(f"Renamed to: {new_filename}")
            
            # 更新 num_samples
            num_samples = actual_num_samples
            filename = new_filename
            
        # 計算批次數
        samples_per_batch = 8 * 4  # 8 workers * 4 batch_size = 32 samples per batch
        num_batches = max(1, num_samples // 8 // 4)
        
        print(f"File: {filename}")
        print(f"    Samples: {num_samples}")
        print(f"    Batches: {num_batches}")
        print(f"    Samples per batch: {samples_per_batch}")
        print(f"    Last batch samples: {num_samples % samples_per_batch}")
        
        total_batches += num_batches
    
    print(f"\nTotal batches: {total_batches}")
    return total_batches

def ensure_min_samples(samples, min_samples=9000):
    """確保每個檔案的樣本數量都是精確的 9000"""
    if len(samples) >= min_samples:
        # 如果樣本數量超過 9000，只取前 9000 個
        return samples[:min_samples]
    
    # 計算需要重複幾次
    num_repeats = min_samples // len(samples)
    
    print(f"Not enough samples ({len(samples)}), repeating {num_repeats} times to reach {min_samples} samples")
    
    # 重複資料
    repeated_samples = samples * num_repeats
    
    # 補充到 9000 個
    needed_more = min_samples - len(repeated_samples)
    repeated_samples.extend(samples[:needed_more])
    
    # 確保最終數量是 9000
    assert len(repeated_samples) == min_samples, f"Expected {min_samples} samples, got {len(repeated_samples)}"
    
    return repeated_samples

def process_combined_split():
    input_file = '/home/avlab/SceneTxtVAR/data/real_data/splits/combined_split.jsonl'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return
    
    output_dir = create_output_dir()
    print(f"\nProcessing combined split...")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # 統計不同高寬比的圖片數量
    ratio_count = {}
    
    # 計數器
    processed_count = 0
    failed_count = 0
    
    print("Reading combined split file...")
    with open(input_file, 'r') as f:
        for line in f:
            try:
                sample = json.loads(line)
                h_div_w = sample.get('h_div_w', None)
                if h_div_w is None:
                    print(f"Warning: Missing h_div_w in sample: {sample}")
                    failed_count += 1
                    continue
                
                # 確保 h_div_w 是浮點數
                try:
                    h_div_w = float(h_div_w)
                except ValueError:
                    print(f"Warning: Invalid h_div_w value: {h_div_w}")
                    failed_count += 1
                    continue
                
                # 找到最接近的 template
                nearest_template = find_nearest_template(h_div_w)
                
                # 統計每個高寬比的圖片數量
                if nearest_template not in ratio_count:
                    ratio_count[nearest_template] = []
                
                ratio_count[nearest_template].append(sample)
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} samples...")
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                failed_count += 1
                continue
    
    print(f"\nSummary:")
    print(f"Total samples: {processed_count + failed_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed to process: {failed_count}")
    
    if processed_count == 0:
        print("Warning: No samples were successfully processed")
        return
    
    # 顯示每個比率的原始樣本數量
    print(f"\nOriginal samples per ratio:")
    for ratio, samples in ratio_count.items():
        print(f"Ratio {ratio}: {len(samples)} samples")
    
    # 處理所有高寬比的圖片
    print(f"\nProcessing ratios:")
    for ratio, samples in list(ratio_count.items()):
        print(f"Processing ratio {ratio} with {len(samples)} samples")
        
        # 確保每個檔案至少有 3000 個樣本
        samples = ensure_min_samples(samples)
        
        # 創建輸出文件名
        output_file = os.path.join(output_dir, f"{ratio:.3f}_{len(samples):07d}.jsonl")
        print(f"Creating file: {output_file}")
        
        # 寫入 JSONL 文件
        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Wrote {len(samples)} samples for ratio {ratio}")
    
    # 驗證所有文件的批次數
    total_batches = verify_batches(output_dir)
    print(f"\nTotal batches after verification: {total_batches}")

def main():
    process_combined_split()

if __name__ == '__main__':
    main()
