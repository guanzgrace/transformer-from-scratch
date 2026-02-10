"""
Sampling strategy tests for GPT-2.

Tests the TransformerSampler using pretrained GPT-2 weights to verify that
each sampling method produces expected behavior:
- Greedy decoding: deterministic, most-likely token selection
- Temperature scaling: controls randomness of predictions
- Top-k sampling: restricts sampling to k most likely tokens
- Top-p (nucleus) sampling: restricts sampling to cumulative probability threshold
- Frequency penalty: discourages repetition of already-used tokens
"""
# %%
import torch as t
from transformer import DemoTransformer, Config, load_gpt2_weights, device
from transformers import GPT2TokenizerFast
from sampling import TransformerSampler
# %%
########################################################################################################
# Test sampling implementation
########################################################################################################

if __name__ == "__main__":
    # Disable gradients for inference
    t.set_grad_enabled(False)

    print("Loading model...")
    cfg = Config(debug=False)
    model = DemoTransformer(cfg).to(device)
    model = load_gpt2_weights(cfg, model)
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    sampler = TransformerSampler(model, tokenizer)

    prompt = "The meaning of life is"

    print("\n" + "="*70)
    print("SAMPLING TESTS")
    print("="*70)
    print(f"Prompt: {prompt!r}\n")

    # Test 1: Greedy decoding
    print("1. Greedy decoding (temperature=0.0)")
    output = sampler.sample(prompt, max_tokens_generated=8, temperature=0.0)
    print(f"   {output!r}\n")

    # Test 2: Temperature sampling
    print("2. Temperature sampling (temperature=0.7)")
    output = sampler.sample(prompt, max_tokens_generated=20, temperature=0.7, seed=42)
    print(f"   {output!r}\n")

    # Test 3: Top-k sampling
    print("3. Top-k sampling (k=50)")
    output = sampler.sample(prompt, max_tokens_generated=15, temperature=1.0, top_k=50, seed=42)
    print(f"   {output!r}\n")

    # Test 4: Top-p sampling
    print("4. Top-p sampling (p=0.9)")
    output = sampler.sample(prompt, max_tokens_generated=20, temperature=1.0, top_p=0.9, seed=42)
    print(f"   {output!r}\n")

    # Test 5: Frequency penalty
    print("5. Frequency penalty (penalty=2.0)")
    output = sampler.sample(prompt, max_tokens_generated=15, temperature=1.0, frequency_penalty=2.0, seed=42)
    print(f"   {output!r}\n")

    print("="*70)
    print("All sampling tests complete!")