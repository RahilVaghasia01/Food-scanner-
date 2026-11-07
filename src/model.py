from transformers import ViTForImageClassification

def initialize_vit_model(num_classes, model_name="google/vit-base-patch16-224-in21k"):
    """
    Loads a pre-trained Vision Transformer from Hugging Face 
    and replaces the top head layer with our custom class count.
    """
    print(f"Initializing {model_name} with {num_classes} output classes...")
    
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True # Ignores the default 21k ImageNet head
    )
    
    return model