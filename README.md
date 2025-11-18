# Token Trainer - GPT-2 from Scratch

A PyTorch implementation of GPT-2 training from scratch, featuring custom transformer architecture with causal self-attention, multi-layer perceptron blocks, and efficient data loading.

## Overview

This project implements a GPT-2 language model trained from scratch using PyTorch. The implementation includes:

- **Custom Transformer Architecture**: Built from the ground up with attention mechanisms, feed-forward networks, and layer normalization
- **Efficient Training**: Optimized training loop with gradient accumulation and learning rate scheduling
- **Flexible Configuration**: Support for different model sizes (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- **Pretrained Weight Loading**: Option to initialize from HuggingFace pretrained models

## Features

### Model Architecture

- **Causal Self-Attention**: Implements masked multi-head attention for autoregressive generation
- **Multi-Layer Perceptron**: GELU activation with projection layers
- **Block Structure**: Stacked transformer blocks with residual connections
- **Layer Normalization**: Pre-normalization for stable training

### Training Features

- **Mixed Precision Training**: Automatic mixed precision (AMP) support for faster training
- **Gradient Accumulation**: Enables effective large batch sizes on limited hardware
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **DataLoader**: Custom implementation with efficient batching
- **Multi-device Support**: Automatic detection and utilization of CUDA, MPS (Apple Silicon), or CPU

## Model Configurations

```python
'gpt2':        block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embed=768
'gpt2-medium': block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embed=1024
'gpt2-large':  block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embed=1280
'gpt2-xl':     block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embed=1600
```

## Installation

### Prerequisites

```bash
pip install torch tiktoken
```

### Clone the Repository

```bash
git clone https://github.com/abhi1021/token-trainer.git
cd token-trainer
```

## Usage

### Training from Scratch

```bash
python train_gpt2-8.py
```

### Training Parameters

- **Batch Size**: 8 (configurable via gradient accumulation)
- **Learning Rate**: 3e-5
- **Max Iterations**: 50 (adjust as needed)
- **Sequence Length**: 1024 tokens
- **Gradient Accumulation**: 5 steps

### Input Data

Place your training text in `input.txt`. The model uses the GPT-2 tokenizer (tiktoken) to encode the text.

## Model Components

### CausalSelfAttention

Implements scaled dot-product attention with causal masking:
- Query, Key, Value projections
- Multi-head attention mechanism
- Attention dropout for regularization
- Causal mask to prevent attending to future tokens

### MLP (Multi-Layer Perceptron)

Feed-forward network with:
- Linear transformation to 4x hidden dimension
- GELU activation function
- Projection back to model dimension
- Dropout for regularization

### Block

Transformer block combining:
- Layer normalization before attention and MLP
- Residual connections around each sub-layer
- Causal self-attention
- Position-wise feed-forward network

### GPT Model

Complete model architecture:
- Token and position embeddings
- Stack of transformer blocks
- Language modeling head for next-token prediction
- Weight sharing between token embeddings and output layer

## Training Process

1. **Data Loading**: Loads text from `input.txt` and encodes using GPT-2 tokenizer
2. **Batch Generation**: Creates batches with shape (B, T) where B=batch size, T=sequence length
3. **Forward Pass**: Computes logits and cross-entropy loss
4. **Backward Pass**: Gradient computation with mixed precision
5. **Optimization**: AdamW optimizer with weight decay
6. **Loss Tracking**: Prints training loss every iteration

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB+ VRAM, CUDA-capable GPU
- **Apple Silicon**: MPS acceleration supported

## Device Selection

The script automatically selects the best available device:
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon)
3. CPU (fallback)

## File Structure

```
token-trainer/
├── train_gpt2-8.py          # Main training script
├── input.txt                # Training data
└── README.md                # This file
```

## Technical Details

### Attention Mechanism

- Uses scaled dot-product attention: `softmax(QK^T / √d_k)V`
- Causal masking prevents information leakage from future tokens
- Multi-head attention allows the model to attend to different positions

### Optimization

- **Optimizer**: AdamW with weight decay (0.1)
- **Learning Rate**: 3e-5 with cosine annealing
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Uses float16 for faster computation

### Memory Efficiency

- Gradient accumulation reduces memory requirements
- Efficient data loading with custom DataLoader
- Option to use smaller model configurations

## Loading Pretrained Weights

The script supports loading pretrained GPT-2 weights from HuggingFace:

```python
model = GPT.from_pretrained('gpt2')
```

Supported models: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`

## Output

During training, the script outputs:
- Current iteration number
- Training loss
- Estimated time per iteration

## Future Improvements

- [ ] Validation set evaluation
- [ ] Model checkpointing
- [ ] WandB/TensorBoard integration
- [ ] Distributed training support
- [ ] Text generation inference script
- [ ] Fine-tuning capabilities

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al.
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This is an educational implementation for learning purposes. For production use, consider using official implementations from HuggingFace Transformers.
