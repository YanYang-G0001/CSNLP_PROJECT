#linear
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

# Ignore unnecessary warnings
warnings.filterwarnings('ignore')

# --- Main Functions and Class Definitions ---

# 2. Model Wrapper Classes
class VisionEncoderWrapper(nn.Module):
    """Wrapper class for the vision encoder to extract image features"""
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

    @property
    def feature_dim(self):
        return self.model.num_features

class LanguageModelWrapper:
    """Wrapper class for the language model to extract text embeddings with positional information"""
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_input_embeddings(self, texts, device):
        """Get the full input embeddings including positional embeddings"""
        if isinstance(texts, str):
            texts = [texts]

        # Convert text to tokens and move to the specified device
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=77)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            if hasattr(self.model, 'transformer'):
                token_embeds = self.model.transformer.wte(tokens['input_ids'])
                seq_len = tokens['input_ids'].size(1)
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand_as(tokens['input_ids'])
                pos_embeds = self.model.transformer.wpe(position_ids)
                full_embeddings = token_embeds + pos_embeds
            else:
                full_embeddings = self.model.get_input_embeddings()(tokens['input_ids'])
                print(f"Warning: Cannot add positional embeddings for {self.model.config.model_type}")
        
        if 'attention_mask' in tokens:
            mask = tokens['attention_mask'].unsqueeze(-1).float()
            masked_embeddings = full_embeddings * mask
            lengths = mask.sum(dim=1)
            return (masked_embeddings.sum(dim=1) / lengths.clamp(min=1))
        else:
            return full_embeddings.mean(dim=1)

    @property
    def embedding_dim(self):
        return self.model.config.hidden_size

class ProjectionLayer(nn.Module):
    """A linear layer to project vision features into the language embedding space"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

# 3. Data Processing Functions
class ImageCaptionDataset(Dataset):
    """Custom dataset class for DataLoader"""
    def __init__(self, samples, preprocess_fn):
        self.samples = samples
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, caption = self.samples[idx]
        # squeeze(0) is to remove the batch dimension added by preprocess_fn
        image_tensor = self.preprocess_fn(image).squeeze(0) 
        return image_tensor, caption

def download_and_unzip_flickr8k():
    """Download and unzip the Flickr8k dataset"""
    url = "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
    zip_path = "flickr8k.zip"
    extract_path = "./flickr8k"

    if os.path.exists(extract_path):
        print("Flickr8k dataset already exists.")
        return

    print("Downloading Flickr8k dataset...")
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete. Unzipping...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Unzipping complete.")

    os.remove(zip_path)

def load_evaluation_data(num_samples=1000, image_dir="./flickr8k/Images", caption_file="./flickr8k/captions.txt"):
    """Load image-caption pairs from the local Flickr8k directory"""
    print("Loading evaluation dataset from local Flickr8k...")
    if not os.path.exists(image_dir) or not os.path.exists(caption_file):
        raise FileNotFoundError("Flickr8k dataset files not found. Please ensure it has been downloaded and unzipped.")

    caption_dict = {}
    with open(caption_file, newline='', encoding='utf-8') as csvfile:
        next(csvfile) # Skip header row
        reader = csv.reader(csvfile)
        for row in reader:
            img_name, caption = row[0], row[1]
            caption_dict.setdefault(img_name, []).append(caption)

    print(f"Found {len(caption_dict)} images with captions.")

    image_names = list(caption_dict.keys())
    random.shuffle(image_names)
    image_names = image_names[:num_samples]

    samples = []
    for name in image_names:
        image_path = os.path.join(image_dir, name)
        if not os.path.exists(image_path): continue
        try:
            image = Image.open(image_path).convert("RGB")
            caption = caption_dict[name][0]
            samples.append((image, caption))
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            continue

    print(f"Successfully loaded {len(samples)} image-caption pairs.")
    return samples

def preprocess_image(image, size=224):
    """Preprocess the image for the vision model"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = timm.data.create_transform(
        input_size=size,
        is_training=False,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return transform(image).unsqueeze(0)

# 4. Training and Evaluation Functions (Updated to InfoNCE)
def train_projection_layer_infonce(vision_model, language_model, projection_layer,
                                   train_loader, epochs=10, lr=1e-4, device='cpu'):
    """Train the linear projection layer using the InfoNCE contrastive loss method"""
    optimizer = torch.optim.AdamW(projection_layer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    projection_layer.train()

    for epoch in range(epochs):
        total_loss = 0
        for images, captions in train_loader:
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Get features
            with torch.no_grad():
                vision_features = vision_model(images)
            
            projected_vision = projection_layer(vision_features)
            
            with torch.no_grad():
                lang_embeddings = language_model.get_input_embeddings(list(captions), device=device)
                
            # InfoNCE loss calculation
            projected_vision_norm = F.normalize(projected_vision, p=2, dim=1)
            lang_embeddings_norm = F.normalize(lang_embeddings, p=2, dim=1)
            
            temperature = 0.07  # Temperature parameter commonly used in InfoNCE
            logits = (projected_vision_norm @ lang_embeddings_norm.T) / temperature
            
            labels = torch.arange(len(images), device=device)
            
            # Calculate symmetric loss from image-to-text and text-to-image
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.T, labels)
            
            loss = (loss_i + loss_t) / 2
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader) if train_loader else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    projection_layer.eval()

# --- [NEW] Model Saving Function ---
def save_projection_layer(model, v_name, l_name, save_dir="checkpoints"):
    """Saves the projection layer's state dictionary."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{v_name}_to_{l_name}.pt")
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

# 5. Main Execution Logic
def main():
    """Main function to execute the entire experiment workflow"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize vision models
    print("Initializing vision models...")
    vision_models = {
        'vit_small': VisionEncoderWrapper('vit_small_patch16_224'),
        'vit_base': VisionEncoderWrapper('vit_base_patch16_224'),
        'vit_large': VisionEncoderWrapper('vit_large_patch16_224'),
        'vit_huge': VisionEncoderWrapper('vit_huge_patch14_224'),
    }
    for name, model in vision_models.items():
        model.to(device)
        print(f"  {name}: feature_dim = {model.feature_dim}")

    # Initialize language models
    print("\nInitializing language models...")
    language_models = {
        'gpt2': LanguageModelWrapper('gpt2'),
        'gpt2-medium': LanguageModelWrapper('gpt2-medium'),
        'gpt2-large': LanguageModelWrapper('gpt2-large')
    }
    for name, model in language_models.items():
        model.model.to(device)
        print(f"  {name}: embedding_dim = {model.embedding_dim}")

    # Create linear projection layers
    print("\nCreating linear projection layers...")
    projection_layers = {}
    for v_name, v_model in vision_models.items():
        for l_name, l_model in language_models.items():
            proj_name = f"{v_name}_to_{l_name}"
            projection_layers[proj_name] = ProjectionLayer(
                v_model.feature_dim,
                l_model.embedding_dim
            ).to(device)
            print(f"  Created projection layer: {proj_name} ({v_model.feature_dim} -> {l_model.embedding_dim})")

    # Download and load data
    download_and_unzip_flickr8k()
    sample_data = load_evaluation_data(2000)
    
    # Increase the amount of training data
    train_samples = sample_data[:1800]
    # Note: test_samples are not used in this script anymore, but loaded in evaluate.py
    
    # Create DataLoader
    train_dataset = ImageCaptionDataset(train_samples, preprocess_image)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # Train all projection layers (using InfoNCE) and save them
    print("\nStarting training (using InfoNCE loss)...")
    for proj_name, proj_layer in projection_layers.items():
        v_name, l_name = proj_name.split('_to_')
        print(f"\n--- Training {proj_name} ---")
        
        train_projection_layer_infonce(
            vision_models[v_name],
            language_models[l_name],
            proj_layer,
            train_loader,
            epochs=20,
            lr=1e-4,   
            device=device
        )
        
        # --- [NEW] Save the trained model ---
        save_projection_layer(proj_layer, v_name, l_name)

    print("\n" + "="*50)
    print("All models have been trained and saved successfully.")
    print("You can now run the `evaluate.py` script to see the results.")
    print("="*50)


if __name__ == "__main__":
    main()
