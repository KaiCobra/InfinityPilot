import os
import json
import glob
from tqdm import tqdm

# 定義輸入JSONL目錄
INPUT_JSONL_DIR = "/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/SynData/real_data/combined_splits_by_ratio"

def update_h_div_w_in_jsonl(jsonl_path):
    """更新JSONL文件中的h_div_w值為文件名中的比例"""
    # 從文件名中提取ratio
    base_name = os.path.basename(jsonl_path)
    try:
        ratio_str = base_name.split('_')[0]
        target_ratio = float(ratio_str)
    except (ValueError, IndexError):
        print(f"Invalid ratio in filename: {base_name}")
        return
    
    print(f"\nUpdating h_div_w to {target_ratio} in {base_name}")
    
    # 讀取原始jsonl文件
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    # 用於存儲更新後的行
    updated_lines = []
    updated_count = 0
    
    # 處理每一行
    for line in tqdm(lines, desc=f"Processing {base_name}"):
        try:
            data = json.loads(line)
            # 更新h_div_w
            data['h_div_w'] = target_ratio
            updated_lines.append(json.dumps(data) + '\n')
            updated_count += 1
        except Exception as e:
            print(f"Error processing line: {e}")
            updated_lines.append(line)
    
    # 將更新後的內容寫回文件
    with open(jsonl_path, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"Updated {updated_count} records in {base_name}")

def main():
    # 獲取所有jsonl文件
    jsonl_files = glob.glob(os.path.join(INPUT_JSONL_DIR, "*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {INPUT_JSONL_DIR}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    # 處理每個jsonl文件
    for jsonl_file in sorted(jsonl_files):
        update_h_div_w_in_jsonl(jsonl_file)
    
    print("\nAll updates completed!")

if __name__ == "__main__":
    main()
