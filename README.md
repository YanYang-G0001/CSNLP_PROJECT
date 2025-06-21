<div align="center">
  <h1>Which Side Should We Scale? Revisiting Parameter Balance in Vision-Language Models</h1>

  <a href="#"><img src="https://img.shields.io/badge/Paper-ComingSoon-lightgrey" alt="Paper PDF"></a>
  <a href="https://github.com/YanYang-G0001/CSNLP_PROJECT"><img src="https://img.shields.io/badge/Code-GitHub-green" alt="GitHub Code"></a>
  <a href="https://yanyang-g0001.github.io/CSNLP_PROJECT//"><img src="https://img.shields.io/badge/Project_Page-Live-blue" alt="Project Page"></a>
  <a href="#"><img src="https://img.shields.io/badge/Poster-GoogleSlides-orange" alt="Poster"></a>

  <br><br>
  <strong>Hepeng Fan Yanyang Gong Qingcheng Wang</strong><br>
  <em>ETH Zurich</em>
</div>

---
This repository contains the implementation and experiments for our research on parameter balance in dual-encoder vision-language models.

## Abstract

Dual-encoder architectures are key to achieving efficient image-text retrieval. This work investigates a core question within this framework: under a limited parameter budget, which modality, vision or language, should be preferentially scaled? We construct a dual-tower model composed of a pre-trained ViT and a text encoder built upon averaged token embeddings from GPT-2. By systematically testing combinations of encoders at different scales, our results show that the model's parameter balance is more critical to performance than the total parameter count.

## Architecture

Our dual-encoder framework consists of:
- **Vision Encoder**: Pre-trained Vision Transformers (ViT) of different scales
- **Language Encoder**: GPT-2 based text encoders with averaged token embeddings
- **Projection Layer**: Linear transformation to align vision and language embeddings
- **Training Objective**: InfoNCE contrastive loss for cross-modal alignment

## Quick Start

### Prerequisites

```bash
pip install torch torchvision transformers timm pillow numpy matplotlib seaborn requests tqdm
```

### Dataset

The code automatically downloads the Flickr8k dataset. No manual setup required!

### Training

Train all model combinations:

```bash
python train_linear.py
```

This will:
- Download and extract the Flickr8k dataset
- Initialize vision models (ViT-Small/Base/Large/Huge) and language models (GPT-2/Medium/Large)
- Train linear projection layers using InfoNCE loss
- Save trained models to `checkpoints/` directory

### Evaluation

Evaluate trained models:

```bash
python evaluate.py
```

This will:
- Load all trained projection layers
- Compute Recall@5 metrics for image-to-text and text-to-image retrieval
- Generate performance heatmaps
- Analyze parameter balance effects

## Model Combinations

We systematically evaluate combinations of:

**Vision Encoders:**
- ViT-Small (22M parameters)
- ViT-Base (86M parameters) 
- ViT-Large (304M parameters)
- ViT-Huge (632M parameters)

**Language Models:**
- GPT-2 (124M parameters)
- GPT-2-Medium (355M parameters)
- GPT-2-Large (774M parameters)

## Results

Our key findings:
- **Parameter balance matters more than total count**: A balanced 660M parameter model outperforms imbalanced 760M and 860M models
- **No clear modality preference**: Vision-heavy and language-heavy configurations show similar performance
- **Balanced scaling is optimal**: Avoiding bottlenecks in either modality yields best results

Results are visualized in heatmaps showing Recall@5 performance across all model combinations.

## File Structure

```
CSNLP_PROJECT/
├── train_linear.py      # Training script with InfoNCE loss
├── evaluate.py          # Evaluation script with Recall@k metrics
├── index.html          # Project webpage
├── README.md           # This file
└── *.png              # Generated result visualizations
```

## Technical Details

### Training Configuration
- **Loss Function**: InfoNCE contrastive loss with temperature τ=0.07
- **Optimizer**: AdamW with learning rate 1e-4
- **Scheduler**: Cosine annealing
- **Batch Size**: 256
- **Epochs**: 20
- **Training Data**: 1,800 image-caption pairs from Flickr8k

### Evaluation Metrics
- **Recall@5**: Percentage of queries where the correct match appears in top-5 results
- **Image-to-Text (I2T)**: Given an image, retrieve its caption
- **Text-to-Image (T2I)**: Given a caption, retrieve its image
- **Test Set**: 200 held-out image-caption pairs

### Key Components

**VisionEncoderWrapper**: Wraps timm vision models for feature extraction
```python
class VisionEncoderWrapper(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
```

**LanguageModelWrapper**: Extracts contextualized embeddings from GPT-2
```python
def get_input_embeddings(self, texts, device):
    # Combines token + positional embeddings, then averages
    full_embeddings = token_embeds + pos_embeds
    return (full_embeddings * mask).sum(dim=1) / mask.sum(dim=1)
```

**ProjectionLayer**: Simple linear transformation for modality alignment
```python
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
```

## Key Insights

1. **Balanced Scaling Strategy**: Rather than heavily scaling one modality, balanced parameter allocation across vision and language components yields optimal performance.

2. **Parameter Count vs. Balance**: Total parameter count is less important than how parameters are distributed between modalities.

3. **No Clear Winner**: Neither vision-heavy nor language-heavy configurations consistently outperform the other, suggesting balanced approaches are more robust.

## Team

- **Hepeng Fan** - ETH Zurich
- **Yanyang Gong** - ETH Zurich  
- **Qingcheng Wang** - ETH Zurich

## Acknowledgements

We thank **Yifan Hou** for his patient guidance and continuous support throughout this project, and **Prof. Mrinmaya Sachan** for designing this excellent course.

## Citation

```bibtex
@misc{fan2024which,
  title={Which Side Should We Scale? Revisiting Parameter Balance in Vision-Language Models},
  author={Hepeng Fan and Yanyang Gong and Qingcheng Wang},
  year={2024},
  institution={ETH Zurich}
}
```

---