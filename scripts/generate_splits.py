import json
import os
from pathlib import Path

def process_dataset(dataset_name, output_dir):
    """
    Process a single dataset and generate its split file.
    """
    dataset_path = Path("data/real_data") / dataset_name
    metadata_dir = dataset_path / "metadatas"
    images_dir = dataset_path / "images"
    normal_vis_dir = dataset_path / "normal_vis"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read all metadata files
    metadata_files = list(metadata_dir.glob("*.json"))
    
    # Process each metadata file
    split_data = []
    for meta_file in metadata_files:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            
        # Get image filenames from paths
        image_name = os.path.basename(meta["image_path"])
        normal_vis_name = os.path.splitext(image_name)[0] + ".jpg"  # Assuming .jpg extension
        
        # Create split entry with exact fields and order as example
        split_entry = {
            "image_path": str(images_dir / image_name),
            "normal_path": str(normal_vis_dir / normal_vis_name),
            "h_div_w": meta["h_div_w"],
            "long_caption": meta["long_caption"],
            "long_caption_type": meta["long_caption_type"],
            "text": meta["text"],
            "short_caption_type": meta["short_caption_type"]
        }
        split_data.append(split_entry)
    
    # Write to JSONL file
    output_file = os.path.join(output_dir, f"{dataset_name}_split.jsonl")
    with open(output_file, 'w') as f:
        for entry in split_data:
            f.write(json.dumps(entry) + "\n")
    
    return split_data

def generate_splits():
    """
    Generate split files for all datasets and a combined split.
    """
    datasets = ["SVT", "TextOCR", "Total-text"]
    output_dir = "data/splits"
    
    # Generate splits for each dataset
    all_splits = []
    for dataset in datasets:
        print(f"Processing {dataset}...")
        splits = process_dataset(dataset, output_dir)
        all_splits.extend(splits)
    
    # Generate combined split
    combined_file = os.path.join(output_dir, "combined_split.jsonl")
    with open(combined_file, 'w') as f:
        for entry in all_splits:
            f.write(json.dumps(entry) + "\n")
    
    print(f"All splits generated successfully in {output_dir}")

if __name__ == "__main__":
    generate_splits()
