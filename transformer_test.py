"""
Lightweight tests for DemoTransformer

Tests the custom GPT-2 implementation against HuggingFace's pretrained model
to verify numerical equivalence (logits difference < 1e-3).

This test suite:
- Compares top-k predictions between DemoTransformer and HF's GPT-2
- Validates logits match within acceptable tolerances
- Tests autoregressive text generation
"""
# %%
import torch as t
from tqdm.notebook import tqdm

from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from transformer import DemoTransformer, Config, load_gpt2_weights, device
# %%
########################################################################################################
# Setup
########################################################################################################
if __name__ == "__main__":
    # Load the standard GPT-2 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Set the padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token 
    # Usage Example:
    # reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer."
    # tokens = tokenizer.encode(reference_text, return_tensors="pt").to(device)

    cfg = Config(debug=False)
    print(cfg)

    # %%
    ########################################################################################################
    # Test this transformer implementation
    ########################################################################################################

    def print_top_k_predictions(logits, tokenizer, k, model_name):
        """
        Print top-k predictions from logits.
        """
        last_token_logits = logits[0, -1]
        _, top_k_indices = t.topk(last_token_logits, k=k)
        probs = t.softmax(last_token_logits, dim=-1)
        
        print(f"\n🔹 {model_name} Top 5 Predictions:")
        for i in range(k):
            token_id = top_k_indices[i].item()
            token_text = tokenizer.decode(token_id)
            prob = probs[token_id].item()
            print(f"  {i+1}: ID {token_id:5} | Prob: {prob:6.2%} | Word: '{token_text}'")


    # Usage Example:
    # reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer."
    # tokens = tokenizer.encode(reference_text, return_tensors="pt").to(device)

    # Create and Load Model to compare against HuggingFace's official GPT-2
    demo_gpt2 = DemoTransformer(cfg).to(device)
    demo_gpt2 = load_gpt2_weights(cfg, demo_gpt2)
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    hf_model.eval()

    # %%
    # Test just one example of getting the top k predictions for DemoTransformer
    str_to_test = "The capital of France is Paris. The capital of Germany is"
    print(f"Getting top 5 tokens after: {str_to_test}")
    test_input = tokenizer.encode(str_to_test, return_tensors="pt").to(device)

    with t.inference_mode():
        logits = demo_gpt2(test_input)
        print_top_k_predictions(logits, tokenizer, k=5, model_name='DemoTransformer')

    # %%
    # Test DemoTransformer against HuggingFace's transformer
    def test_prediction(prompt: str, model_demo=None, model_hf=None, k: int = 5):
        """
        Test predictions for a given prompt on both your model and HuggingFace's model.
        
        Args:
            prompt: The text prompt to test
            model_demo: DemoTransformer model (optional)
            model_hf: HuggingFace GPT2LMHeadModel (optional)
            k: Number of top predictions to show
        """
        print(f"\n{'='*70}")
        print(f"Testing: '{prompt}'")
        print(f"{'='*70}")
        
        test_input = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with t.inference_mode():
            if model_demo is not None:
                logits_demo = model_demo(test_input)
                print_top_k_predictions(logits_demo, tokenizer, k=5, model_name='DemoTransformer')
            
            if model_hf is not None:
                logits_hf = model_hf(test_input).logits
                print_top_k_predictions(logits_hf, tokenizer, k=5, model_name='HuggingFace Transformer')
            
            # Compare logits if both models provided
            if model_demo is not None and model_hf is not None:
                max_diff = (logits_demo - logits_hf).abs().max().item()
                mean_diff = (logits_demo - logits_hf).abs().mean().item()
                
                print(f"\n📊 LOGITS COMPARISON:")
                print(f"  Max diff:  {max_diff:.6f}")
                print(f"  Mean diff: {mean_diff:.6f}")
                print(f"  Status: {'✅ PASS' if max_diff < 1e-3 else '❌ FAIL'}")

    test_prediction("The word is Apple. The word is", model_demo=demo_gpt2, model_hf=hf_model)
    test_prediction("The capital of France is", model_demo=demo_gpt2, model_hf=hf_model)
    test_prediction("1, 2, 3, 4,", model_demo=demo_gpt2, model_hf=hf_model)
    test_prediction("Barack Obama was born in", model_demo=demo_gpt2, model_hf=hf_model)
    test_prediction("The likelihood that a deceased donor kidney will be used", model_demo=demo_gpt2, model_hf=hf_model)


    # %%
    # Test generation of the following 50 tokens given a starting string.
    str_to_test = "The likelihood that a deceased donor kidney will be used"
    print(f"Getting next 50 tokens after: {str_to_test}")

    for i in tqdm(range(50)):
        input_ids = tokenizer.encode(str_to_test, return_tensors="pt").to(device)
        
        with t.inference_mode():
            logits = demo_gpt2(input_ids)
        
        next_token_id = logits[0, -1].argmax().item()
        
        # Append to string
        next_word = tokenizer.decode(next_token_id)
        str_to_test += next_word

    print(str_to_test)
    # %%
