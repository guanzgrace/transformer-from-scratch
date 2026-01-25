# GPT-2 From Scratch

A clean implementation of GPT-2 that achieves numerical equivalence with Hugging Face's model.


## Overview

This project implements GPT-2 from scratch in PyTorch, based on the TransformerLens architecture from the ARENA curriculum. The implementation is decoupled from TransformerLens dependencies (specifically `HookedTransformer`) and uses Hugging Face's tokenizer and pretrained weights for validation (`from transformers import GPT2TokenizerFast, GPT2LMHeadModel`). It includes a complete training pipeline for learning GPT-2 style transformers from scratch on custom datasets.

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
  - Train small transformers on TinyStories dataset


## Project Structure
```
.
├── transformer.py          # Core GPT-2 implementation (model architecture)
├── transformer_test.py     # Model validation tests (compare with HuggingFace)
├── training.py             # Training pipeline
├── training_test.py        # Test a training run on TinyStories dataset
├── outputs/                # Training logs and sample outputs
│   ├── transformer_output.txt
│   └── training_logs.txt
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


## TODO
- [ ] Add `sampling.py` to showcase different methods to sample from a transformer


## Acknowledgments

I would like to acknowledge the ARENA curriculum's Day 1.1: Transformers From Scratch.