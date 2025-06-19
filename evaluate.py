import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import timm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
import warnings
import os
import random
import csv
import zipfile

# This script reuses the class definitions and functions from the training script.
# Make sure they are consistent.

# --- Class Definitions (Copied from training script) ---
class VisionEncoderWrapper(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()
    def forward(self, x): return self.model(x)
    @property
    def feature_dim(self): return self.model.num_features

class LanguageModelWrapper:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
    def get_input_embeddings(self, texts, device):
        if isinstance(texts, str): texts = [texts]
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=77)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            if hasattr(self.model, 'transformer'):
                token_embeds = self.model.transformer.wte(tokens['input_ids'])
                seq_len = tokens['input_ids'].size(1)
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand_as(tokens['input_ids'])
                pos_embeds = self.model.transformer.wpe(position_ids)
                full_embeddings = token_embeds + pos_embeds
            else:
                full_embeddings = self.model.get_input_embeddings()(tokens['input_ids'])
                print(f"Warning: Cannot add positional embeddings for {self.model.config.model_type}")
        if 'attention_mask' in tokens:
            mask = tokens['attention_mask'].unsqueeze(-1).float()
            return ( (full_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) )
        else: return full_embeddings.mean(dim=1)
    @property
    def embedding_dim(self): return self.model.config.hidden_size

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.projection(x)

# --- Helper Functions (Copied from training script) ---
def load_evaluation_data(num_samples=2000, image_dir="./flickr8k/Images", caption_file="./flickr8k/captions.txt"):
    if not os.path.exists(image_dir) or not os.path.exists(caption_file):
        raise FileNotFoundError("Flickr8k dataset files not found.")
    caption_dict = {}
    with open(caption_file, newline='', encoding='utf-8') as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            caption_dict.setdefault(row[0], []).append(row[1])
    image_names = list(caption_dict.keys())
    random.shuffle(image_names)
    image_names = image_names[:num_samples]
    samples = []
    for name in image_names:
        image_path = os.path.join(image_dir, name)
        if not os.path.exists(image_path): continue
        try:
            image = Image.open(image_path).convert("RGB")
            samples.append((image, caption_dict[name][0]))
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            continue
    print(f"Successfully loaded {len(samples)} image-caption pairs for evaluation.")
    return samples

def preprocess_image(image, size=224):
    if image.mode != 'RGB': image = image.convert('RGB')
    transform = timm.data.create_transform(input_size=size, is_training=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transform(image).unsqueeze(0)

def evaluate_similarity(vision_model, language_model, projection_layer, test_samples, device='cpu'):
    similarities = []
    projection_layer.eval()
    with torch.no_grad():
        for image, caption in test_samples:
            try:
                img_tensor = preprocess_image(image).to(device)
                vision_features = vision_model(img_tensor)
                projected_vision = projection_layer(vision_features)
                lang_embeddings = language_model.get_input_embeddings([caption], device=device)
                similarity = F.cosine_similarity(projected_vision, lang_embeddings, dim=1)
                similarities.append(similarity.item())
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    return similarities

def load_projection_layer(model, v_name, l_name, save_dir="checkpoints"):
    """Loads the projection layer's state dictionary."""
    file_path = os.path.join(save_dir, f"{v_name}_to_{l_name}.pt")
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")
    else:
        print(f"Warning: No checkpoint found for {v_name}_to_{l_name} at {file_path}. Using randomly initialized weights.")
    model.eval()

# --- Main Evaluation Logic ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize all models (same as training script)
    vision_models = {
        #'vit_small': VisionEncoderWrapper('vit_small_patch16_224'),
        'vit_base': VisionEncoderWrapper('vit_base_patch16_224'),
        'vit_large': VisionEncoderWrapper('vit_large_patch16_224'),
        'vit_huge': VisionEncoderWrapper('vit_huge_patch14_224'),
    }
    language_models = {
        'gpt2': LanguageModelWrapper('gpt2'),
        'gpt2-medium': LanguageModelWrapper('gpt2-medium'),
        'gpt2-large': LanguageModelWrapper('gpt2-large')
    }
    for model in list(vision_models.values()) + [m.model for m in language_models.values()]:
        model.to(device)

    # Create and load projection layers
    projection_layers = {}
    for v_name, v_model in vision_models.items():
        for l_name, l_model in language_models.items():
            proj_name = f"{v_name}_to_{l_name}"
            proj_layer = ProjectionLayer(v_model.feature_dim, l_model.embedding_dim).to(device)
            load_projection_layer(proj_layer, v_name, l_name) # Load trained weights
            projection_layers[proj_name] = proj_layer
    
    # Load test data
    all_data = load_evaluation_data(2000)
    test_samples = all_data[1800:] # Use the same test split as in training

    # Evaluate all combinations
    print("\nEvaluating all combinations with loaded models...")
    results = {}
    for proj_name, proj_layer in projection_layers.items():
        v_name, l_name = proj_name.split('_to_')
        similarities = evaluate_similarity(
            vision_models[v_name],
            language_models[l_name],
            proj_layer,
            test_samples,
            device=device
        )
        results[proj_name] = {
            'similarities': similarities,
            'mean': np.mean(similarities) if similarities else 0,
            'std': np.std(similarities) if similarities else 0
        }
        print(f"  {proj_name}: Mean Similarity = {results[proj_name]['mean']:.4f} ± {results[proj_name]['std']:.4f}")
    
    # Visualize and analyze results (same as before)
    print("\nGenerating results heatmap...")
    vision_names = list(vision_models.keys())
    language_names = list(language_models.keys())
    similarity_matrix = np.zeros((len(vision_names), len(language_names)))

    for i, v_name in enumerate(vision_names):
        for j, l_name in enumerate(language_names):
            proj_name = f"{v_name}_to_{l_name}"
            similarity_matrix[i, j] = results[proj_name]['mean']

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=language_names, yticklabels=vision_names, annot=True, fmt='.4f', cmap='YlOrRd')
    plt.title('Cosine Similarity Matrix: Vision Encoder vs. Language Model (Linear Projection + InfoNCE)')
    plt.xlabel('Language Model')
    plt.ylabel('Vision Encoder')
    plt.tight_layout()
    plt.savefig("evaluation_heatmap.png")
    print("\nHeatmap saved as 'evaluation_heatmap.png'")
    plt.show()

    print("\n" + "="*50)
    print("Results Analysis")
    print("="*50)
    best_combo = max(results, key=lambda x: results[x]['mean'])
    worst_combo = min(results, key=lambda x: results[x]['mean'])
    print(f"\nBest Combination: {best_combo} (Mean Similarity: {results[best_combo]['mean']:.4f})")
    print(f"Worst Combination: {worst_combo} (Mean Similarity: {results[worst_combo]['mean']:.4f})")

if __name__ == "__main__":
    main()
