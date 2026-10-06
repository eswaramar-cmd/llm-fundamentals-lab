"""
FEATURE 5: PARAMETERS DEMO
A REAL educational miniature GPT-style language model implemented from scratch in PyTorch.

This is NOT a toy example like nn.Linear(1,3) -> ReLU -> nn.Linear(3,1).
This is a decoder-only Transformer that uses the SAME fundamental mechanism as GPT-2,
but with only a few thousand parameters so you can run it on a CPU and understand it.

Run: python 05_parameters_demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# =============================================================================
# 1. CHARACTER-LEVEL TOKENIZER
# =============================================================================
class CharTokenizer:
    """
    A character-level tokenizer converts raw text into token IDs.

    Unlike word-level or subword tokenizers (like tiktoken or BPE),
    a character-level tokenizer maps each individual character to an integer.
    This is the simplest possible tokenizer and is perfect for learning.

    Flow: text -> token IDs -> embeddings -> transformer -> logits
    """
    def __init__(self, text):
        # Build vocabulary from unique characters in the training text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        # Create mappings: character -> integer ID and back
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        """Convert text string to list of token IDs."""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, ids):
        """Convert list of token IDs back to text string."""
        return ''.join([self.idx_to_char[i] for i in ids])

# =============================================================================
# 2. CAUSAL SELF-ATTENTION
# =============================================================================
class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention is the heart of a Transformer.

    For each position in the sequence, it looks at all previous positions
    and computes a weighted sum of their values. The weights are learned
    by comparing each position's query to every key.

    "Causal" means each position can only attend to itself and earlier positions.
    We enforce this with a mask that hides future positions.

    This is where the model learns patterns like "the cat" -> "sat" or "t" -> "h" -> "e".
    """
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V projections: each transforms the input into query, key, value vectors
        # These are the LEARNED PARAMETERS of attention
        # Each is a weight matrix of shape (d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection: combines the attention heads back together
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

        # Causal mask: prevents attending to future tokens
        # This is a buffer (not a parameter) - it doesn't change during training
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  # Batch size, Sequence length, Model dimension

        # --- STEP: Q, K, V projections ---
        # These are where attention PARAMETERS live:
        #   q_proj.weight shape: [d_model, d_model]
        #   k_proj.weight shape: [d_model, d_model]
        #   v_proj.weight shape: [d_model, d_model]
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        # Reshape for multi-head attention: (B, T, n_heads, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)

        # --- STEP: Attention computation ---
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        # This is where the model learns which tokens to attend to
        # Scaling by sqrt(d_k) prevents the softmax from saturating
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, T, T)

        # Apply causal mask: future positions get -inf so softmax ignores them
        # This is what makes it "causal" - the model can't cheat by looking ahead
        attn_weights = attn_weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # Softmax converts raw scores into probabilities (sum to 1)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum of values
        out = attn_weights @ v  # (B, n_heads, T, head_dim)

        # Concatenate heads back together
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Output projection
        out = self.c_proj(out)

        return out

# =============================================================================
# 3. FEED-FORWARD NETWORK
# =============================================================================
class FeedForward(nn.Module):
    """
    After attention, each position goes through a small MLP (feed-forward network).
    This adds non-linearity and increases the model's representational power.

    Structure: Linear -> GELU -> Linear

    In real LLMs like GPT-3, this is often:
    Linear -> GELU -> Linear -> Dropout (in training)

    We simplify it slightly for clarity.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        # --- LEARNED PARAMETERS ---
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),  # Expand to higher dimension (d_model -> d_ff)
            nn.GELU(),                  # Non-linear activation function
            nn.Linear(d_ff, d_model),   # Project back to model dimension (d_ff -> d_model)
        )

    def forward(self, x):
        return self.net(x)

# =============================================================================
# 4. TRANSFORMER BLOCK
# =============================================================================
class TransformerBlock(nn.Module):
    """
    A single Transformer block combines attention, feed-forward, and layer norms.

    This is the building block of the GPT architecture.
    Real LLMs stack many of these blocks (GPT-2: 12, GPT-3: 96).
    """
    def __init__(self, d_model, n_heads, d_ff, block_size, dropout=0.1):
        super().__init__()
        # Multi-head self-attention
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff)
        # Layer normalization (pre-norm style, like GPT-2)
        # Parameters: weight and bias for each feature dimension
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # --- Pre-norm architecture (GPT-2 style) ---
        # LayerNorm BEFORE attention
        x = x + self.dropout(self.attn(self.ln1(x)))
        # LayerNorm BEFORE feed-forward
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

# =============================================================================
# 5. MINI GPT LANGUAGE MODEL
# =============================================================================
class MiniGPT(nn.Module):
    """
    A tiny decoder-only Transformer language model.

    It uses the SAME fundamental mechanism as GPT-2/Llama:
    - Token embeddings
    - Positional embeddings
    - Causal self-attention blocks
    - Layer normalization
    - Feed-forward networks
    - Language model head

    The only differences are the SCALE (tiny) and the fact that
    this is a learning implementation, not a production model.
    """
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=2,
                 d_ff=128, block_size=32, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

        # --- EMBEDDINGS ---
        # These are LEARNED PARAMETERS that map token IDs to dense vectors
        # token_embedding.weight shape: [vocab_size, d_model]
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # position_embedding.weight shape: [block_size, d_model]
        self.position_embedding = nn.Embedding(block_size, d_model)

        # --- TRANSFORMER BLOCKS ---
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_ff, block_size, dropout)
            for _ in range(n_layers)
        ])

        # --- FINAL LAYER NORM ---
        self.ln_f = nn.LayerNorm(d_model)

        # --- LANGUAGE MODEL HEAD ---
        # Maps from d_model back to vocab_size to get logits for each token
        # lm_head.weight shape: [d_model, vocab_size]
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights with small random values (standard practice)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.

        Args:
            idx: Token IDs, shape (B, T) where B=batch, T=sequence length
            targets: Target token IDs for computing loss, shape (B, T)

        Returns:
            logits: Unnormalized scores for each token in vocabulary
            loss: Cross-entropy loss (if targets provided)
        """
        B, T = idx.shape

        # --- STEP: EMBEDDINGS ---
        # Look up token embeddings: (B, T) -> (B, T, d_model)
        # These are the LEARNED PARAMETERS that store semantic meaning
        # For example, 't' and 'h' might learn similar embeddings because they often appear together
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)

        # Add positional information so the model knows WHERE each token is
        # Without this, the model wouldn't know if 'the' comes before or after 'cat'
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)  # (1, T, d_model)

        # Combine token and position embeddings
        x = tok_emb + pos_emb  # (B, T, d_model)

        # --- STEP: TRANSFORMER BLOCKS ---
        # Pass through N layers of attention and feed-forward
        # Inside each block:
        #   - Attention computes which tokens to focus on
        #   - Feed-forward processes each position independently
        #   - LayerNorm stabilizes training
        x = self.blocks(x)  # (B, T, d_model)

        # Final layer norm
        x = self.ln_f(x)  # (B, T, d_model)

        # --- STEP: LOGITS ---
        # Project to vocabulary size to get scores for each possible next token
        # These are called "logits" - raw unnormalized scores
        # Shape: (B, T, vocab_size)
        logits = self.lm_head(x)

        # --- STEP: LOSS ---
        loss = None
        if targets is not None:
            # CrossEntropyLoss expects (N, C) and (N,) where C=vocab_size
            # We flatten the batch and sequence dimensions
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # This is where the model computes how wrong its predictions are
            # Lower loss = better predictions
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# =============================================================================
# 6. PARAMETER INSPECTION UTILITIES
# =============================================================================
def count_parameters(model):
    """Return parameter count breakdown by component."""
    breakdown = {}
    total = 0

    # Token embeddings: vocab_size * d_model
    tok_params = sum(p.numel() for p in model.token_embedding.parameters())
    breakdown['Token embeddings'] = tok_params
    total += tok_params

    # Position embeddings: block_size * d_model
    pos_params = sum(p.numel() for p in model.position_embedding.parameters())
    breakdown['Position embeddings'] = pos_params
    total += pos_params

    # Attention and feed-forward parameters (all blocks)
    attn_params = 0
    ffn_params = 0
    ln_params = 0
    for block in model.blocks:
        attn_params += sum(p.numel() for p in block.attn.parameters())
        ffn_params += sum(p.numel() for p in block.ffn.parameters())
        ln_params += sum(p.numel() for p in block.ln1.parameters())
        ln_params += sum(p.numel() for p in block.ln2.parameters())
    breakdown['Attention'] = attn_params
    breakdown['Feed-forward'] = ffn_params
    breakdown['LayerNorm'] = ln_params
    total += attn_params + ffn_params + ln_params

    # Final layer norm
    ln_f_params = sum(p.numel() for p in model.ln_f.parameters())
    breakdown['LayerNorm (final)'] = ln_f_params
    total += ln_f_params

    # Output head: d_model * vocab_size
    head_params = sum(p.numel() for p in model.lm_head.parameters())
    breakdown['Output head'] = head_params
    total += head_params

    breakdown['Total'] = total
    return breakdown

def print_parameter_details(model):
    """Print parameter names, shapes, and a sample of values."""
    print("=" * 70)
    print("PARAMETER DETAILS")
    print("=" * 70)

    for name, param in model.named_parameters():
        print(f"\n{name}")
        print(f"  Shape: {list(param.shape)}")
        print(f"  Parameters: {param.numel():,}")
        print(f"  Sample values (first 5 elements):")
        print(f"    {param.data.flatten()[:5].tolist()}")

    print("\n" + "=" * 70)

# =============================================================================
# 7. TRAINING UTILITIES
# =============================================================================
def get_batch(data, block_size, batch_size=4):
    """
    Create a batch of training data using next-token prediction.

    Input:  tokens[:-1]  (all tokens except the last one)
    Target: tokens[1:]   (all tokens shifted right by one)

    Example:
        tokens = [1, 2, 3, 4, 5]
        input  = [1, 2, 3, 4]
        target = [2, 3, 4, 5]

    The model learns to predict the next character given the previous ones.
    """
    max_start = max(0, len(data) - block_size)
    if max_start == 0:
        # Dataset is smaller than block_size: use the whole sequence
        x = data[:-1]
        y = data[1:]
        # Repeat to create a batch
        x = x.unsqueeze(0).repeat(batch_size, 1)
        y = y.unsqueeze(0).repeat(batch_size, 1)
    else:
        ix = torch.randint(max_start, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model, data, block_size, eval_iters=10):
    """Estimate average loss on a dataset."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, block_size)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

# =============================================================================
# 8. TEXT GENERATION
# =============================================================================
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    """
    Generate text autoregressively.

    For each step:
    1. Tokenize the current prompt
    2. Run the model to get logits
    3. Sample the next token from the probability distribution
    4. Append the token to the prompt
    5. Repeat

    This is NEXT-TOKEN PREDICTION in action.
    """
    model.eval()

    # Tokenize the prompt
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Crop to block_size if needed (context window management)
        idx_cond = idx[:, -model.block_size:]

        # Forward pass: get logits for next token
        logits, _ = model(idx_cond)

        # Take only the last position's logits
        logits = logits[:, -1, :]  # (B, vocab_size)

        # Apply temperature for sampling diversity
        # Lower temperature = more deterministic, Higher = more random
        logits = logits / temperature

        # Softmax converts logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append the predicted token
        idx = torch.cat((idx, idx_next), dim=1)

    # Decode back to text
    generated = tokenizer.decode(idx[0].tolist())
    model.train()
    return generated

# =============================================================================
# 9. MAIN FUNCTION
# =============================================================================
def main():
    print("=" * 70)
    print("LLM Fundamentals Practice Lab")
    print("FEATURE 5: MINI GPT - REAL TRANSFORMER LANGUAGE MODEL")
    print("=" * 70)
    print()

    # Important disclaimer
    print("IMPORTANT CAVEAT:")
    print("This is a miniature Transformer with only a few thousand parameters.")
    print("It uses the SAME fundamental mechanism as GPT-2, GPT-3, and Llama,")
    print("but it is NOT equivalent to GPT-4, Llama 3, or any production model.")
    print("Real LLMs have BILLIONS of parameters trained on terabytes of data.")
    print("This model is for EDUCATION ONLY.")
    print()

    # Hyperparameters - kept small for CPU execution
    block_size = 16      # Maximum sequence length (small due to tiny dataset)
    d_model = 64         # Embedding dimension (size of each token vector)
    n_heads = 2          # Number of attention heads
    n_layers = 2         # Number of Transformer blocks
    d_ff = 128           # Hidden dimension in feed-forward networks
    dropout = 0.1
    batch_size = 4
    learning_rate = 3e-4
    max_epochs = 300
    eval_interval = 50
    max_new_tokens = 100

    # Training data
    text = "the cat sat on the mat. the cat ate food. the dog sat on the mat."
    print(f"Training text: '{text}'")
    print()

    # Initialize tokenizer
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary: {tokenizer.chars}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()

    # Encode the entire dataset
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Encoded data: {data.tolist()}")
    print(f"Data length: {len(data)} tokens")
    print()

    # Split into train/val (90/10 split on this tiny dataset)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    print()

    # Initialize model
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        block_size=block_size,
        dropout=dropout
    )

    # Count parameters
    breakdown = count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())

    print("=" * 70)
    print("PARAMETER COUNT BREAKDOWN")
    print("=" * 70)
    for component, count in breakdown.items():
        print(f"{component:30s}: {count:>10,}")
    print("=" * 70)
    print(f"\nTotal trainable parameters: {total_params:,}")
    print()

    # Show parameter details BEFORE training
    print_parameter_details(model)

    # Select a tensor to track before/after training
    track_tensor_name = "token_embedding.weight"
    track_tensor = None
    for name, param in model.named_parameters():
        if name == track_tensor_name:
            track_tensor = param.data.clone()
            break

    print("=" * 70)
    print("SELECTED TENSOR BEFORE TRAINING")
    print("=" * 70)
    print(f"Tensor: {track_tensor_name}")
    print(f"Shape: {list(track_tensor.shape)}")
    print(f"Values (first 10 elements):")
    print(f"  {track_tensor.flatten()[:10].tolist()}")
    print()

    # Set up optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # Used implicitly in the model

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"Optimizer: AdamW")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {max_epochs}")
    print()

    # Training loop
    for epoch in range(max_epochs):
        # Get a batch of training data
        # INPUT  = tokens[:-1]
        # TARGET = tokens[1:]  (shifted by one position)
        xb, yb = get_batch(train_data, block_size, batch_size)

        # Forward pass: compute logits and loss
        # LOGITS are raw scores for each token in the vocabulary
        # LOSS is how wrong the predictions are
        logits, loss = model(xb, yb)

        # Backward pass: compute gradients
        # This is BACKPROPAGATION - the chain rule computes how each parameter
        # contributed to the loss
        optimizer.zero_grad()  # Reset gradients from previous step
        loss.backward()        # Compute gradients via backpropagation
        optimizer.step()       # Update PARAMETERS using gradients

        # Print progress
        if epoch % eval_interval == 0 or epoch == max_epochs - 1:
            train_loss = loss.item()
            val_loss = estimate_loss(model, val_data, block_size)
            print(f"epoch {epoch:3d} -> train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    print()

    # Show parameter details AFTER training
    print_parameter_details(model)

    # Show the tracked tensor AFTER training
    print("=" * 70)
    print("SELECTED TENSOR AFTER TRAINING")
    print("=" * 70)
    for name, param in model.named_parameters():
        if name == track_tensor_name:
            new_tensor = param.data.clone()
            print(f"Tensor: {name}")
            print(f"Shape: {list(new_tensor.shape)}")
            print(f"Values (first 10 elements):")
            print(f"  {new_tensor.flatten()[:10].tolist()}")
            print()
            print("CHANGE:")
            diff = (new_tensor - track_tensor).flatten()[:10].tolist()
            print(f"  Delta (first 10 elements): {diff}")
            print(f"  Max absolute change: {(new_tensor - track_tensor).abs().max().item():.6f}")
            break

    print()
    print("=" * 70)
    print("TEXT GENERATION")
    print("=" * 70)

    # Generate from prompts
    test_prompts = ["the cat", "the dog", "t"]
    for prompt in test_prompts:
        generated = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated}'")

    print()
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("- Token embeddings convert character IDs into dense vectors.")
    print("  These are LEARNED PARAMETERS that capture relationships between characters.")
    print("- Position embeddings tell the model WHERE each character is in the sequence.")
    print("  Without them, the model would treat 'the' and 'eht' the same way.")
    print("- Causal self-attention computes weighted sums of PREVIOUS tokens.")
    print("  The weights are learned by the Q, K, V projections.")
    print("- Feed-forward networks add NON-LINEARITY to each position independently.")
    print("- LayerNorm STABILIZES training by normalizing activations.")
    print("- The LM head projects to VOCABULARY SIZE to predict the next token.")
    print("- CrossEntropyLoss measures PREDICTION ERROR.")
    print("- Backpropagation computes GRADIENTS via the chain rule.")
    print("- AdamW updates PARAMETERS to minimize loss.")
    print("- This is the SAME mechanism used in GPT-2, just much smaller.")
    print("- Real LLMs are GPT models scaled up by 1000x-1,000,000x.")
    print("- The fundamental principles are IDENTICAL.")

if __name__ == "__main__":
    main()
