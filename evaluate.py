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
from tqdm import tqdm 

# --- Class Definitions  ---
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

# --- Helper Functions  ---
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
            # For simplicity, we use the first caption as the ground truth for each image
            samples.append((Image.open(image_path).convert("RGB"), caption_dict[name][0]))
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            continue
    print(f"Successfully loaded {len(samples)} image-caption pairs for evaluation.")
    return samples

def preprocess_image(image, size=224):
    if image.mode != 'RGB': image = image.convert('RGB')
    transform = timm.data.create_transform(input_size=size, is_training=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transform(image).unsqueeze(0)

def load_projection_layer(model, v_name, l_name, save_dir="checkpoints"):
    file_path = os.path.join(save_dir, f"{v_name}_to_{l_name}.pt")
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu'))) # Load and adapt to CPU
        print(f"Model loaded from {file_path}")
    else:
        print(f"Warning: No checkpoint found for {v_name}_to_{l_name} at {file_path}. Using randomly initialized weights.")
    model.eval()


# --- NEW: Recall@k Evaluation Function ---
def calculate_recall_at_k(vision_model, language_model, projection_layer, test_samples, k, device='cpu'):
    """
    Calculates Image-to-Text and Text-to-Image Recall@k.
    """
    projection_layer.eval()
    
    all_image_embeddings = []
    all_text_embeddings = []

    print(f"  > Pre-computing embeddings for {len(test_samples)} samples...")
    # 1. Pre-compute all embeddings for the test set
    with torch.no_grad():
        for image, caption in tqdm(test_samples, desc="Embedding"):
            # Image embedding
            img_tensor = preprocess_image(image).to(device)
            vision_features = vision_model(img_tensor)
            projected_vision = projection_layer(vision_features)
            all_image_embeddings.append(projected_vision)
            
            # Text embedding
            lang_embeddings = language_model.get_input_embeddings([caption], device=device)
            all_text_embeddings.append(lang_embeddings)

    # Convert lists to tensors and normalize (L2 norm)
    # Normalization makes dot product equivalent to cosine similarity, which is more stable
    image_embeds = F.normalize(torch.cat(all_image_embeddings), p=2, dim=-1)
    text_embeds = F.normalize(torch.cat(all_text_embeddings), p=2, dim=-1)
    
    num_samples = len(test_samples)

    # 2. Calculate full similarity matrix
    sim_matrix = image_embeds @ text_embeds.T
    
    # 3. Calculate Image-to-Text Recall@k (I2T)
    # For each image, find the rank of its corresponding text
    i2t_scores = sim_matrix
    _, i2t_topk_indices = torch.topk(i2t_scores, k, dim=1)
    
    # The correct text for image `i` is text `i`. We create a tensor of correct indices [0, 1, 2, ...]
    correct_indices = torch.arange(num_samples, device=device).view(-1, 1)
    
    # Check if the correct index is in the top k results for each row
    i2t_hits = (i2t_topk_indices == correct_indices).any(dim=1).sum().item()
    i2t_recall = (i2t_hits / num_samples) * 100

    # 4. Calculate Text-to-Image Recall@k (T2I)
    # For each text, find the rank of its corresponding image
    t2i_scores = sim_matrix.T # Transpose the matrix for text-to-image retrieval
    _, t2i_topk_indices = torch.topk(t2i_scores, k, dim=1)

    # Correct indices are still [0, 1, 2, ...], logic remains the same
    t2i_hits = (t2i_topk_indices == correct_indices).any(dim=1).sum().item()
    t2i_recall = (t2i_hits / num_samples) * 100

    return i2t_recall, t2i_recall


# --- Main Evaluation Logic (MODIFIED) ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize all models
    vision_models = {
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
    all_data = load_evaluation_data(2000) # Use 2000 samples from the dataset
    test_samples = all_data[1800:] # Use 200 samples for the final test set

    # Evaluate all combinations for Recall@5
    RECALL_K = 5
    print(f"\nEvaluating all combinations for Recall@{RECALL_K}...")
    results = {}
    for proj_name, proj_layer in projection_layers.items():
        print(f"\n--- Evaluating {proj_name} ---")
        v_name, l_name = proj_name.split('_to_')
        
        i2t_recall, t2i_recall = calculate_recall_at_k(
            vision_models[v_name],
            language_models[l_name],
            proj_layer,
            test_samples,
            k=RECALL_K,
            device=device
        )
        
        avg_recall = (i2t_recall + t2i_recall) / 2
        results[proj_name] = {
            'i2t_r5': i2t_recall,
            't2i_r5': t2i_recall,
            'avg_r5': avg_recall
        }
        print(f"  > Results for {proj_name}: I2T R@5 = {i2t_recall:.2f}%, T2I R@5 = {t2i_recall:.2f}%, Avg R@5 = {avg_recall:.2f}%")
    
    # Visualize and analyze results using Average Recall@5
    print("\nGenerating Recall@5 heatmap...")
    vision_names = list(vision_models.keys())
    language_names = list(language_models.keys())
    recall_matrix = np.zeros((len(vision_names), len(language_names)))

    for i, v_name in enumerate(vision_names):
        for j, l_name in enumerate(language_names):
            proj_name = f"{v_name}_to_{l_name}"
            recall_matrix[i, j] = results[proj_name]['avg_r5']

    plt.figure(figsize=(10, 8))
    sns.heatmap(recall_matrix, xticklabels=language_names, yticklabels=vision_names, annot=True, fmt='.2f', cmap='YlOrRd', annot_kws={"size": 12})
    plt.title(f'Average Recall@{RECALL_K} (%): Vision Encoder vs. Language Model')
    plt.xlabel('Language Model')
    plt.ylabel('Vision Encoder')
    plt.tight_layout()
    plt.savefig(f"evaluation_recall_at_{RECALL_K}_heatmap.png")
    print(f"\nHeatmap saved as 'evaluation_recall_at_{RECALL_K}_heatmap.png'")
    plt.show()

    print("\n" + "="*50)
    print(f"Recall@{RECALL_K} Results Analysis")
    print("="*50)
    best_combo = max(results, key=lambda x: results[x]['avg_r5'])
    worst_combo = min(results, key=lambda x: results[x]['avg_r5'])
    print(f"\nBest Combination: {best_combo} (Avg R@5: {results[best_combo]['avg_r5']:.2f}%)")
    print(f"Worst Combination: {worst_combo} (Avg R@5: {results[worst_combo]['avg_r5']:.2f}%)")

if __name__ == "__main__":
    main()
