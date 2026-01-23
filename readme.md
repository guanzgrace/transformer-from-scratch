# GPT-2 From Scratch

A clean implementation of GPT-2 that achieves numerical equivalence with Hugging Face's model.

## Overview

This project implements GPT-2 from scratch in PyTorch, based on the TransformerLens architecture from the ARENA curriculum. The implementation is decoupled from TransformerLens dependencies (specifically `HookedTransformer`) and uses Hugging Face's tokenizer and pretrained weights for validation (`from transformers import GPT2TokenizerFast, GPT2LMHeadModel`).

I did this as an educational exercise to understand transformer architecture more deeply.

## Features

- **Numerically Validated**: Matches HuggingFace GPT-2 output with logit differences < 1e-3
- **Pretrained Weights**: Loads and uses official GPT-2 pretrained weights
- **Comprehensive Testing**: Includes validation tests comparing against HuggingFace's implementation
- **Usage Examples**: 
  - Generate top-k next token predictions
  - Compare outputs between this and HuggingFace models
  - Generate sequences (50+ tokens) from a starting prompt

## Project Structure
```
.
├── transformer.py          # Core GPT-2 implementation + tests
├── transformer_output.txt  # Sample output logs (python transformer.py > transformer_output.txt)
└── README.md
```

## Things that I learned

- **`einops.einsum`** is really useful for readable tensor operations
- Understanding the **forward pass dimensions** of each layer is crucial
- **Numerical precision** matters when validating against reference implementations

## TODO
- [ ] Add `training.py` to train the model locally
- [ ] Add `sampling.py` to showcase different methods to sample from a transformer

## Acknowledgments

I would like to acknowledge the ARENA curriculum's TransformerLens implementation.