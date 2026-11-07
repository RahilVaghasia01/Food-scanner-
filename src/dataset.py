import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import ViTImageProcessor

class FoodDataset(Dataset):
    def __init__(self, csv_file, img_dir, processor_name="google/vit-base-patch16-224-in21k"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            processor_name (string): Name of the ViT processor.
        """
        # Read CSV. Assumes columns are 'file_name' and 'label' (or adjust accordingly)
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = ViTImageProcessor.from_pretrained(processor_name)
        
        # Ensure label mapping works if labels are integers or strings
        self.labels = self.df.iloc[:, 1].tolist()
        self.filenames = self.df.iloc[:, 0].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get image path
        img_name = str(self.filenames[idx])
        # Append extension if missing in your CSV file names (e.g., if CSV says '001' instead of '001.jpg')
        if not (img_name.endswith('.jpg') or img_name.endswith('.png')):
            img_name += '.jpg'
            
        img_path = os.path.join(self.img_dir, img_name)
        
        # 2. Load image safely
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to a blank image if a file is corrupted
            image = Image.new("RGB", (224, 224), color=0)
            print(f"Error loading image {img_path}: {e}")

        # 3. Process image for ViT (resizes to 224x224 and normalizes)
        inputs = self.processor(image, return_tensors="pt")
        
        # 4. Format outputs
        item = {
            "pixel_values": inputs["pixel_values"].squeeze(0), 
            "labels": torch.tensor(int(self.labels[idx])) 
        }
        return item

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }