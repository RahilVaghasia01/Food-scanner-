import os
import evaluate
import numpy as np
from dataset import FoodDataset, collate_fn
from model import initialize_vit_model
from transformers import TrainingArguments, Trainer

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    # --- 1. SET UP PATHS ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Point directly to the new metadata files inside the flat folders
    TRAIN_CSV = os.path.join(BASE_DIR, "data", "train_flat", "metadata.csv")
    TRAIN_IMG_DIR = os.path.join(BASE_DIR, "data", "train_flat")
    
    VAL_CSV = os.path.join(BASE_DIR, "data", "val_flat", "metadata.csv")
    VAL_IMG_DIR = os.path.join(BASE_DIR, "data", "val_flat")
    
    # --- 2. INSTANTIATE DATASETS ---
    print("Loading datasets...")
    train_dataset = FoodDataset(csv_file=TRAIN_CSV, img_dir=TRAIN_IMG_DIR)
    val_dataset = FoodDataset(csv_file=VAL_CSV, img_dir=VAL_IMG_DIR)
    
    # Safely extract dynamic number of unique classes from your training label series
    num_classes = len(train_dataset.df.iloc[:, 1].unique())
    print(f"Detected {num_classes} distinct food categories.")

    # --- 3. INITIALIZE MODEL ---
    model = initialize_vit_model(num_classes=num_classes)

    # --- 4. CONFIG TRAINING HYPERPARAMETERS ---
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=32,   
        per_device_eval_batch_size=32,
        eval_strategy="epoch",            
        save_strategy="epoch",            
        learning_rate=5e-5,               
        num_train_epochs=5,               
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        use_cpu=False,                  
        dataloader_num_workers=0, 
    )

    # --- 5. RUN TRAINING ENGINE ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn, 
        compute_metrics=compute_metrics,
    )

    print("Beginning training session...")
    trainer.train()
    
    # Save the polished champion configurations
    print("Saving optimized food classification weights...")
    trainer.save_model("./checkpoints/best_model")

if __name__ == "__main__":
    main()