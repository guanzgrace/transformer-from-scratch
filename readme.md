# GPT-2 From Scratch

A from-scratch implementation of GPT-2 in PyTorch, built to deeply understand every component of a decoder-only transformer end-to-end — from attention heads to training loops to sampling strategies.

## Overview

This project implements GPT-2 from scratch in PyTorch, based on the TransformerLens architecture from the ARENA curriculum. The implementation is decoupled from TransformerLens dependencies (specifically `HookedTransformer`) and uses Hugging Face's tokenizer and pretrained weights for validation (`from transformers import GPT2TokenizerFast, GPT2LMHeadModel`). It includes a complete training pipeline for learning GPT-2 style transformers from scratch on custom datasets, and a sampling module with multiple decoding strategies (greedy, temperature, top-k, top-p, and frequency penalty).

I did this as an educational exercise to understand transformer architecture more deeply.


## Features

- **Numerically Validated**: Matches HuggingFace GPT-2 output with logit differences < 1e-3
- **Pretrained Weights**: Loads and uses official GPT-2 pretrained weights
- **Training from Scratch**: Complete training pipeline for learning GPT-2 style models on custom datasets
- **Comprehensive Testing**: Includes validation tests comparing against HuggingFace's implementation
- **Usage Examples**: 
  - Generate top-k next token predictions
  - Compare outputs between this and HuggingFace models
  - Generate sequences (50+ tokens) from a starting prompt
  - (`training.py`) Train small transformers on TinyStories dataset
  - (`sampling.py`) Sample with greedy, top-k, top-p, temperature, and frequency penalty strategies

## Quickstart
```
pip install -r requirements.txt
```

```
# Validate the model against HuggingFace's GPT-2
python transformer_test.py

# Train a small model on TinyStories
python training_test.py

# Test sampling strategies with pretrained GPT-2
python sampling_test.py
```

**Note:** All `.py` files include `# %%` cell markers and can be run interactively as Jupyter notebooks in VS Code.

## Project Structure
```
.
├── transformer.py          # Core GPT-2 implementation (model architecture)
├── transformer_test.py     # Model validation tests (compare with HuggingFace)
├── training.py             # Training pipeline
├── training_test.py        # Test a training run on TinyStories dataset
├── sampling.py             # Sampling strategies (greedy, top-k, top-p, temperature, frequency penalty)
├── sampling_test.py        # Sampling strategy tests with pretrained GPT-2
├── requirements.txt        # Python dependencies
├── outputs/                # Training logs and sample outputs
│   ├── transformer_output.txt
│   ├── training_logs.txt
│   └── sampling_output.txt
└── README.md
```


## Results

Training a small model (2 layers, 64 dim) on TinyStories for 5 epochs (~15 min on MPS):
- Final loss: ~3.17
- Final accuracy: ~35.5%
- The output progressed from nonsense to coherent sentences.

See `outputs/training_logs.txt` for the full training log.


## Things that I learned

From building the transformer architecture:
- **`einops.einsum`** is really useful for readable tensor operations
- Understanding the **forward pass dimensions** of each layer is crucial
- **Numerical precision** matters when validating against reference implementations

From training:
- Even training a small transformer can be slow. I had to reduce the number of layers and create a smaller model to make the training be reasonable. I also had to increase the batch size.
- As long as loss is steadily decreasing and accuracy is increasing, it should be ok. Generation quality improves gradually.
- For example:
  - At the end of Epoch 2, the generated story is: "Once upon a time, "I'm a big, you. I can't know what you can't know what you are you [truncated]"
  - At the end of Epoch 5, the generated story is: "Once upon a time, there was a little girl named Lily. She loved to play with her friends. One day, she saw a big, there was a little girl named Lily. She loved to play with her friends. One day, she saw a big, there"


## Acknowledgments

I would like to acknowledge the ARENA curriculum's Day 1.1: Transformers From Scratch.