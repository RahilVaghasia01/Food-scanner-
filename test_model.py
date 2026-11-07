import os
import torch
import pandas as pd
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# 1. Paths configuration
MODEL_PATH = "./checkpoints/best_model"
CSV_PATH = "./data/train_flat/metadata.csv" 

# Put the name of a real food image file you have inside your data folder to test
# OR drag a random food image from Google into this folder and type its name here!
TEST_IMAGE_PATH = "./test.jpg" 

def main():
    # 2. Check if model exists locally
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Could not find the '{MODEL_PATH}' folder.")
        print("Please make sure you downloaded your checkpoint from Kaggle and renamed it to 'food_model'.")
        return

    print("📦 Loading your custom trained ViT model and processor...")
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    model.eval()

    # 3. Load the image
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Error: Cannot find test image at {TEST_IMAGE_PATH}")
        print("Please update TEST_IMAGE_PATH with a real image path on your computer.")
        return

    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    
    # 4. Process image and predict
    print("🔮 Running inference locally...")
    inputs = processor(image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()

    print("\n" + "="*40)
    print(f"🎯 Predicted Class ID: {predicted_class_idx}")
    
    # 5. Look up the human-readable food name from your CSV
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            # Find the rows matching this index to discover the food name
            # Assuming your CSV column 1 contains the class labels
            matches = df[df.iloc[:, 1] == predicted_class_idx]
            if not matches.empty:
                # Assuming your CSV might have a 'label_name' column, or let's print the match
                print(f" Food Category Match Found in Dataset!")
                print(f"  Sample File with this label: {matches.iloc[0, 0]}")
        except Exception as e:
            print(f"Could not parse CSV names: {e}")
            
    print("="*40)

if __name__ == "__main__":
    main()