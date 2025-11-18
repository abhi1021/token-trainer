---
license: mit
tags:
- pytorch
- causal-lm
- gpt2
---

# GPT-2 Style Model

This is a GPT-2 style language model trained from scratch.

## Model Details

- **Architecture**: GPT-2
- **Parameters**:
  - Layers: 12
  - Heads: 12
  - Embedding dimension: 768
  - Vocabulary size: 50257
  - Block size: 1024
- **Training steps**: 5000
- **Final loss**: 6.9023

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(repo_id="agileabhi/gpt_tokeniser", filename="model.pt")

# Load the checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

# Recreate the model (you'll need the GPT class definition)
config = GPTConfig(**checkpoint['config'])
model = GPT(config)
model.load_state_dict(checkpoint['model_state_dict'])
```
