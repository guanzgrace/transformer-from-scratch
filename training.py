"""
Transformer Training Pipeline

A training pipeline for GPT-2 style transformers from scratch, including:
- Dataset tokenization and preprocessing
- Training loop with loss tracking and periodic evaluation
- Text generation sampling during training
- Configurable training hyperparameters

Key components:
- TransformerTrainingArgs: Dataclass for training configuration
- tokenize_and_concatenate: Prepares text data for language modeling
- TransformerTrainer: Handles training loop, evaluation, and generation

Example usage:
    model = DemoTransformer(config).to(device)
    trainer = TransformerTrainer(args, model, tokenizer)
    train_losses, eval_accuracies = trainer.train(train_loader, test_loader)

I based the code on the ARENA curriculum (TransformerLens implementation). Then, I 
decoupled the implementation from TransformerLens, using only standard PyTorch and
HuggingFace's tokenizer. 
"""

# %%
from dataclasses import dataclass
import torch as t
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformer import DemoTransformer, Config, get_log_probs, device
from transformers import GPT2TokenizerFast

# %%
@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    eval_every: int = 100  # Evaluate every N steps
    print_every: int = 50   # Print loss every N steps

# %%
def tokenize_and_concatenate(
    dataset,
    tokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 4,
):
    """
    Tokenize and concatenate a dataset.
    Replacement for TransformerLens' tokenize_and_concatenate.
    """
    def tokenize_function(examples):
        text = examples[column_name]
        tokens = tokenizer(text, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        
        if add_bos_token:
            tokens = [[tokenizer.bos_token_id] + seq for seq in tokens]
        
        # Concatenate all sequences with EOS tokens between them
        concatenated = []
        for seq in tokens:
            concatenated.extend(seq + [tokenizer.eos_token_id])
        
        # Split into chunks of max_length
        total_length = len(concatenated)
        total_length = (total_length //max_length) * max_length
        
        result = {
            "tokens": [
                concatenated[i : i + max_length]
                for i in range(0, total_length, max_length)
            ]
        }
        return result
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc if not streaming else None,
    )
    
    return tokenized

# %%
class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        super().__init__()
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.step = 0
        self.train_losses = []
        self.eval_accuracies = []

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        loss = -get_log_probs(logits, tokens).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        self.train_losses.append(loss.item())
        return loss

    @t.inference_mode()
    def evaluate(self, test_loader: DataLoader) -> float:
        """
        Evaluate the model on the test set and return the accuracy.
        """
        self.model.eval()
        total_correct, total_samples = 0, 0

        for batch in tqdm(test_loader, desc="Evaluating"):
            tokens = batch["tokens"].to(device)
            logits: Tensor = self.model(tokens)[:, :-1]
            predicted_tokens = logits.argmax(dim=-1)
            total_correct += (predicted_tokens == tokens[:, 1:]).sum().item()
            total_samples += tokens.size(0) * (tokens.size(1) - 1)

        accuracy = total_correct / total_samples
        self.model.train()
        return accuracy

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        """
        Trains the model for `self.args.epochs` epochs.
        """
        print(f"Starting training for {self.args.epochs} epochs")
        print(f"Total steps per epoch: {min(len(train_loader), self.args.max_steps_per_epoch)}")
        print(f"Device: {device}")
        print("-" * 70)
        
        self.model.train()
        accuracy = 0.0

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs, desc="Training")

        for epoch in range(self.args.epochs):
            epoch_losses = []
            
            for i, batch in enumerate(train_loader):
                loss = self.training_step(batch)
                epoch_losses.append(loss.item())
                
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch + 1}/{self.args.epochs} | Loss: {loss:.4f} | Acc: {accuracy:.3f}"
                )
                
                # Print loss periodically
                if (i + 1) % self.args.print_every == 0:
                    avg_loss = np.mean(epoch_losses[-self.args.print_every:])
                    print(f"Step {self.step}: avg loss = {avg_loss:.4f}")
                
                # Evaluate periodically
                if (i + 1) % self.args.eval_every == 0:
                    accuracy = self.evaluate(test_loader)
                    self.eval_accuracies.append(accuracy)  # Track all evaluations
                    print(f"Step {self.step}: accuracy = {accuracy:.4f}")
                
                if i >= self.args.max_steps_per_epoch:
                    break

            # End of epoch evaluation and sample generation
            print(f"\n{'='*70}")
            print(f"End of Epoch {epoch + 1}")
            print(f"{'='*70}")
            
            accuracy = self.evaluate(test_loader)
            self.eval_accuracies.append(accuracy)
            avg_epoch_loss = np.mean(epoch_losses)
            
            print(f"Average Loss: {avg_epoch_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            
            sample_text = self.generate_sample("Once upon a time", max_tokens=50)
            print(f"\nGenerated Sample:\n{sample_text}\n")
            print("="*70 + "\n")

        progress_bar.close()
        print("Training complete!")
        return self.train_losses, self.eval_accuracies

        
    @t.inference_mode()
    def generate_sample(self, prompt: str = "Once upon a time", max_tokens: int = 50) -> str:
        """
        Generates a sample text from the model using greedy decoding.
        """
        self.model.eval()
        
        # Encode prompt once
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate tokens one at a time
        for _ in range(max_tokens):
            # Truncate to context length if needed
            if input_ids.shape[1] > self.model.cfg.n_ctx:
                input_ids = input_ids[:, -self.model.cfg.n_ctx:]
            
            logits = self.model(input_ids)
            next_token_id = logits[0, -1].argmax().item()
            
            # Stop at EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            next_token = t.tensor([[next_token_id]], device=device)
            input_ids = t.cat([input_ids, next_token], dim=1)
        
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.model.train()
        return generated_text
