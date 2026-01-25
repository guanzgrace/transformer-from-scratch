"""
Simple Training for DemoTransformer

Tests the training implementation by training a small transformer from scratch
on the TinyStories dataset and monitoring learning progress.

This test suite:
- Trains a small GPT-2 style model from random initialization
- Monitors loss and next-token prediction accuracy
- Generates sample text at each epoch to visualize learning progress
- Validates that the model learns basic language patterns over time

Note: This trains from scratch (not pretrained weights).
Expect gradual improvement from nonsense → simple phrases → coherent sentences over ~5 epochs.
5 epochs with the current configs takes ~15 min to train locally on my laptop.
"""
# %%
from torch.utils.data import DataLoader
import datasets

from transformer import DemoTransformer, Config, device
from training import TransformerTrainer, TransformerTrainingArgs, tokenize_and_concatenate
from transformers import GPT2TokenizerFast

# %%
########################################################################################################
# Test this training implementation
########################################################################################################

if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model configuration (smaller for faster training)
    model_cfg = Config(
        debug=False,
        d_model=32,
        n_heads=4,
        d_head=8,
        d_mlp=32 * 4,
        n_layers=2,
        n_ctx=128,
        d_vocab=50257,
    )
    print(model_cfg)
    
    # Initialize model
    model = DemoTransformer(model_cfg).to(device)
    
    # Load and prepare dataset
    print("\nLoading dataset...")
    dataset = datasets.load_dataset("roneneldan/TinyStories", split="train[:10000]")  # Subset for faster training
    
    tokenized_dataset = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=model_cfg.n_ctx,
        column_name="text",
        add_bos_token=True,
        num_proc=4,
    )
    
    # Split into train/test
    tokenized_dataset.set_format(type="torch", columns=["tokens"])
    dataset_dict = tokenized_dataset.train_test_split(test_size=1000, seed=23)
    
    # Create data loaders
    args = TransformerTrainingArgs(
        batch_size=64,
        epochs=5,
        max_steps_per_epoch=500,
        lr=1e-3,
        weight_decay=1e-2,
        eval_every=100,
        print_every=50,
    )
    
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        dataset_dict["train"], 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        dataset_dict["test"], 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    # Train the model
    trainer = TransformerTrainer(args, model, tokenizer)
    train_losses, eval_accuracies = trainer.train(train_loader, test_loader)
    
    # See outputs on a couple other prompts 
    test_prompts = [
        "Once upon a time",
        "The cat sat on the",
        "In a land far away",
    ]
    
    for prompt in test_prompts:
        sample_text = trainer.generate_sample(prompt, max_tokens=50)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {sample_text}\n")
# %%
