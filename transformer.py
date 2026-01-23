"""
GPT-2 Transformer Implementation from Scratch

A clean implementation of the GPT-2 architecture that achieves numerical equivalence
with HuggingFace's pretrained model (logits difference < 1e-3).

I based the code on the ARENA curriculum (TransformerLens implementation). Then, I 
decoupled the implementation from TransformerLens, using only standard PyTorch and
HuggingFace's tokenizer. 
"""
# %%
from dataclasses import dataclass

import einops
import torch as t
import torch.nn as nn
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.notebook import tqdm

# Standard Hugging Face replacements for TransformerLens
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# Set device
device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# Load the standard GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# Set the padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token 
# Usage Example:
# reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer."
# tokens = tokenizer.encode(reference_text, return_tensors="pt").to(device)

MAIN = __name__ == "__main__"

# %%
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()
print(cfg)
# %%
########################################################################################################
# Model Architecture
########################################################################################################
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # take mean, variance
        res_mean = residual.mean(dim=-1, keepdim=True)
        res_var = residual.var(dim=-1, unbiased=False, keepdim=True)

        # for std, add small layer_norm_eps so we avoid division by 0
        res_std = t.sqrt(res_var + self.cfg.layer_norm_eps)

        # scale residual
        residual = (residual - res_mean) / res_std

        # take linear transformation of scaled residual
        return residual * self.w + self.b
    
class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        # Get the positional embeddings (every word at position 1 has the same PosEmbed, etc)
        batch, seq = tokens.shape
        return einops.repeat(self.W_pos[:seq], 'seq d_model -> batch seq d_model', batch=batch)

class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # b = batch, s = sequence position, d = d_model, n = n_heads, h = d_head
        # calculate Query, Key, and Value
        Q = einops.einsum(normalized_resid_pre, self.W_Q, 'b s d, n d h -> b s n h') + self.b_Q
        K = einops.einsum(normalized_resid_pre, self.W_K, 'b s d, n d h -> b s n h') + self.b_K
        V = einops.einsum(normalized_resid_pre, self.W_V, 'b s d, n d h -> b s n h') + self.b_V

        # both q=seq_q and k=seq_k are the same dim
        # using einsum we don't have to worry about transposing K
        QKT = einops.einsum(Q, K, 'b q n h, b k n h -> b n q k')

        # scale and mask the attention scores, apply softmax to get probabilities
        A = self.apply_causal_mask(QKT / self.cfg.d_head**0.5)
        attn_pattern = A.softmax(dim=-1)

        # take attention-weighted sum of value vectors
        z = einops.einsum(V, attn_pattern, 'b k n h, b n q k -> b q n h')

        # last linear transformation
        output = einops.einsum(z, self.W_O, 'b q n h, n h d -> b q d') + self.b_O
        return output
    
    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        # create upper triangular matrix of 1's where we want to set these to -inf
        # don't include the diagonal since words can see themselves
        attn_size = attn_scores.shape[-1]
        mask = t.triu(t.ones((attn_size, attn_size)), diagonal=1).to(device)

        # setting them to -inf means the softmax will turn them to 0
        # this ensures that each word can only see the past and itself, not the future
        attn_scores.masked_fill_(mask.bool(), self.IGNORE)
        return attn_scores

def gelu_new(
    input: Float[Tensor, "batch pos d_mlp"]
) -> Float[Tensor, "batch pos d_mlp"]:
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    # Taken from TransformerLens
    return (
        0.5
        * input
        * (1.0 + t.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * t.pow(input, 3.0))))
    )

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # b = batch, s = sequence position, d = d_model, m = d_mlp
        resid = einops.einsum(normalized_resid_mid, self.W_in, 'b s d, d m -> b s m') + self.b_in

        # Pytorch's GELU didn't work for me, so I copied the TransformerLens implementation
        # GeLU smooths the ReLU function
        resid = gelu_new(resid)
        return einops.einsum(resid, self.W_out, 'b s m, m d -> b s d') + self.b_out
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        # b = batch, s = sequence position, d = d_model, v = d_vocab
        return einops.einsum(normalized_resid_final, self.W_U, 'b s d, d v -> b s v') + self.b_U

class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        resid = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            resid = block(resid)
        return self.unembed(self.ln_final(resid)) # returns logits

# %%
########################################################################################################
# Functions to get log probs and load actual model weights
########################################################################################################

def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )
    return log_probs_for_tokens

def load_gpt2_weights(local_model: DemoTransformer):
    '''
    Loads weights from the Hugging Face GPT-2 small model into a DemoTransformer instance.
    '''
    # Load the official HF model weights
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_state = hf_model.state_dict()
    
    local_state = local_model.state_dict()

    # 1. Embeddings
    local_state["embed.W_E"] = hf_state["transformer.wte.weight"]
    local_state["pos_embed.W_pos"] = hf_state["transformer.wpe.weight"]

    # 2. Blocks
    for i in range(local_model.cfg.n_layers):
        # LayerNorms
        local_state[f"blocks.{i}.ln1.w"] = hf_state[f"transformer.h.{i}.ln_1.weight"]
        local_state[f"blocks.{i}.ln1.b"] = hf_state[f"transformer.h.{i}.ln_1.bias"]
        local_state[f"blocks.{i}.ln2.w"] = hf_state[f"transformer.h.{i}.ln_2.weight"]
        local_state[f"blocks.{i}.ln2.b"] = hf_state[f"transformer.h.{i}.ln_2.bias"]

        # HF stores QKV combined as 'c_attn.weight' [768, 2304]
        # Split and transpose them to fit [n_heads, d_model, d_head]
        qkv = hf_state[f"transformer.h.{i}.attn.c_attn.weight"].T # Transpose to [2304, 768]
        q, k, v = qkv.chunk(3, dim=0)
        
        local_state[f"blocks.{i}.attn.W_Q"] = q.view(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        local_state[f"blocks.{i}.attn.W_K"] = k.view(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        local_state[f"blocks.{i}.attn.W_V"] = v.view(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)

        # Attention Biases
        qkv_b = hf_state[f"transformer.h.{i}.attn.c_attn.bias"]
        qb, kb, vb = qkv_b.chunk(3, dim=0)
        local_state[f"blocks.{i}.attn.b_Q"] = qb.view(cfg.n_heads, cfg.d_head)
        local_state[f"blocks.{i}.attn.b_K"] = kb.view(cfg.n_heads, cfg.d_head)
        local_state[f"blocks.{i}.attn.b_V"] = vb.view(cfg.n_heads, cfg.d_head)

        # Projection (Output) Layer
        # HF stores this as [d_model, d_model]. 
        # Reshape to [d_model, n_heads, d_head] then transpose to [n_heads, d_head, d_model]
        W_O_temp = hf_state[f"transformer.h.{i}.attn.c_proj.weight"].T
        local_state[f"blocks.{i}.attn.W_O"] = einops.rearrange(
            W_O_temp, "d_model (n_heads d_head) -> n_heads d_head d_model", 
            n_heads=cfg.n_heads
        )
        local_state[f"blocks.{i}.attn.b_O"] = hf_state[f"transformer.h.{i}.attn.c_proj.bias"]
        
        # MLP
        local_state[f"blocks.{i}.mlp.W_in"] = hf_state[f"transformer.h.{i}.mlp.c_fc.weight"]
        local_state[f"blocks.{i}.mlp.b_in"] = hf_state[f"transformer.h.{i}.mlp.c_fc.bias"]
        local_state[f"blocks.{i}.mlp.W_out"] = hf_state[f"transformer.h.{i}.mlp.c_proj.weight"]
        local_state[f"blocks.{i}.mlp.b_out"] = hf_state[f"transformer.h.{i}.mlp.c_proj.bias"]

    # 3. Final LayerNorm and Unembed
    local_state["ln_final.w"] = hf_state["transformer.ln_f.weight"]
    local_state["ln_final.b"] = hf_state["transformer.ln_f.bias"]
    local_state["unembed.W_U"] = hf_state["transformer.wte.weight"].T
    # Ensure bias is zeroed as HF GPT-2 doesn't use it
    local_state["unembed.b_U"] = t.zeros(local_model.cfg.d_vocab, device=device)

    local_model.load_state_dict(local_state)
    return local_model

def print_top_k_predictions(logits, tokenizer, k, model_name):
    """
    Print top-k predictions from logits.
    """
    last_token_logits = logits[0, -1]
    top_k_values, top_k_indices = t.topk(last_token_logits, k=k)
    probs = t.softmax(last_token_logits, dim=-1)
    
    print(f"\n🔹 {model_name} Top 5 Predictions:")
    for i in range(k):
        token_id = top_k_indices[i].item()
        token_text = tokenizer.decode(token_id)
        prob = probs[token_id].item()
        print(f"  {i+1}: ID {token_id:5} | Prob: {prob:6.2%} | Word: '{token_text}'")

# %%
########################################################################################################
# Testing this transformer implementation
########################################################################################################

# Create and Load Model to compare against HuggingFace's official GPT-2
demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
demo_gpt2 = load_gpt2_weights(demo_gpt2)
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