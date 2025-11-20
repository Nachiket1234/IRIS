# IRIS core.py - Complete Deep Dive

## 📋 Table of Contents
1. [Overview](#overview)
2. [Class Structure](#class-structure)
3. [Initialization (`__init__`)](#initialization)
4. [Method 1: `encode_support()`](#method-1-encode_support)
5. [Method 2: `forward()`](#method-2-forward)
6. [Method 3: Memory Bank Operations](#method-3-memory-bank-operations)
7. [Method 4: In-Context Tuner](#method-4-in-context-tuner)
8. [Data Flow Examples](#data-flow-examples)
9. [Architecture Diagram](#architecture-diagram)

---

## Overview

**File**: `src/iris/model/core.py` (131 lines)

**Purpose**: The `IrisModel` class is the **main orchestrator** of the IRIS architecture. It assembles and coordinates:
- **3D UNet Encoder** (extracts features)
- **Task Encoding Module** (creates task embeddings from support set)
- **Mask Decoder** (generates segmentation predictions)
- **Memory Bank** (stores learned prototypes)
- **In-Context Tuner** (optional fine-tuning)

Think of `core.py` as the **conductor** that makes all the musicians (encoder, decoder, etc.) play together in harmony.

---

## Class Structure

```python
class IrisModel(nn.Module):
    """
    Implements the IRIS architecture for episodic medical image segmentation.
    """
```

### Inheritance
- Inherits from `torch.nn.Module` (PyTorch's base neural network class)
- This makes it trainable and compatible with PyTorch ecosystem

### Key Components (Attributes)
```python
self.volume_shape       # (D, H, W) - Expected input dimensions
self.encoder            # Medical3DUNetEncoder - Feature extraction
self.task_encoder       # TaskEncodingModule - Support set encoding
self.mask_decoder       # MaskDecoder - Segmentation generation
self.memory_bank        # ClassMemoryBank - Prototype storage (optional)
```

---

## Initialization (`__init__`)

### **Lines 24-59**: Constructor

```python
def __init__(
    self,
    *,                                    # Force keyword-only arguments
    in_channels: int = 1,                 # Input channels (1 for grayscale)
    base_channels: int = 32,              # Starting channel count
    num_query_tokens: int = 8,            # Contextual tokens (m)
    num_attention_heads: int = 8,         # Multi-head attention heads
    volume_shape: Tuple[int, int, int] = (128, 128, 128),  # (D, H, W)
    use_memory_bank: bool = True,         # Enable memory bank?
    memory_momentum: float = 0.999,       # EMA momentum for memory
) -> None:
```

### **Step-by-Step Initialization**

#### **Step 1: Store Volume Shape** (Line 37)
```python
self.volume_shape = volume_shape
```
- Stores expected input dimensions: `(Depth, Height, Width)`
- Example: `(16, 256, 256)` for Chest X-Ray
- Used later for decoder upsampling target

---

#### **Step 2: Create Encoder** (Lines 38-42)
```python
self.encoder = Medical3DUNetEncoder(
    in_channels=in_channels,      # 1 for grayscale medical images
    base_channels=base_channels,  # 32 → grows to [32, 64, 128, 256, 512]
    stages=4,                     # 4 downsampling stages (fixed)
)
```

**What the Encoder Does:**
- Takes input volume: `(B, 1, D, H, W)` → e.g., `(2, 1, 16, 256, 256)`
- Applies 4 downsampling stages via residual blocks
- Each stage: **halves spatial dimensions, doubles channels**

**Channel Progression:**
```
Stage 0 (Stem):  1 → 32   (D, H, W)         No downsample
Stage 1:        32 → 64   (D/2, H/2, W/2)   Stride 2
Stage 2:        64 → 128  (D/4, H/4, W/4)   Stride 2
Stage 3:       128 → 256  (D/8, H/8, W/8)   Stride 2
Stage 4:       256 → 512  (D/16, H/16, W/16) Stride 2
```

**Output:**
- **Features**: `(B, 512, D/16, H/16, W/16)` - Deepest features
- **Skip Connections**: 4 tensors at different resolutions for UNet decoder

**Example with (16, 256, 256) input:**
```python
Input:  (B, 1, 16, 256, 256)
Skip 0: (B, 32, 16, 256, 256)   # After stem
Skip 1: (B, 64, 8, 128, 128)    # After stage 1
Skip 2: (B, 128, 4, 64, 64)     # After stage 2
Skip 3: (B, 256, 2, 32, 32)     # After stage 3
Output: (B, 512, 1, 16, 16)     # After stage 4
```

---

#### **Step 3: Calculate Channel List** (Lines 43-45)
```python
encoder_channels = [
    base_channels * (2 ** i) for i in range(5)
]
# With base_channels=32: [32, 64, 128, 256, 512]
```

This creates the channel progression list used by decoder to match encoder's skip connections.

---

#### **Step 4: Create Task Encoding Module** (Lines 46-51)
```python
self.task_encoder = TaskEncodingModule(
    feature_channels=encoder_channels[-1],  # 512 (deepest features)
    num_query_tokens=num_query_tokens,      # 8 learnable tokens
    num_attention_heads=num_attention_heads,# 8 heads for attention
    downsample_ratio=self.encoder.downsample_ratio,  # 16 (2^4)
)
```

**What the Task Encoder Does:**
- Takes **support features** + **support masks** as input
- Produces **task embeddings** that encode "what to segment"
- Two parallel pathways:

**Pathway 1: Foreground Encoding** (Masked Average Pooling)
```python
# Upsample features to match mask resolution
upsampled_features = upsample(support_features)  # (B, C, D, H, W)

# Apply mask to focus on foreground
masked_features = upsampled_features * support_masks  # Element-wise multiply

# Pool over spatial dimensions
foreground_embedding = masked_features.sum(dims=[2,3,4]) / mask_area
# Shape: (B, K, C) where K = num_classes
```

**Pathway 2: Contextual Encoding** (Learnable Query Tokens)
```python
# Use Pixel Shuffle for resolution enhancement
context_features = pixel_shuffle_3d(support_features)  # Increases resolution

# Concatenate with mask
combined = concat([context_features, mask], dim=1)

# Apply convolution
context_features = conv(combined)  # (B, context_channels, D, H, W)

# Flatten to sequence
context_seq = context_features.flatten(2).transpose(1, 2)  # (B, D*H*W, C)

# Learnable query tokens attend to context
query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, 8, C)
context_embeddings = cross_attention(query_tokens, context_seq)  # (B, 8, C)
```

**Final Output:**
```python
# Combine foreground + contextual for each class
task_embeddings = concat([foreground_embedding, context_embeddings], dim=2)
# Shape: (B, K, m+1, C) = (B, K, 9, 512)
#        where m+1 = 1 foreground + 8 contextual tokens
```

---

#### **Step 5: Create Mask Decoder** (Lines 52-57)
```python
self.mask_decoder = MaskDecoder(
    encoder_channels=encoder_channels,      # [32, 64, 128, 256, 512]
    num_query_tokens=num_query_tokens,      # 8 (matches TEM)
    num_attention_heads=num_attention_heads,# 8
    final_upsample_target=volume_shape,     # (D, H, W)
)
```

**What the Decoder Does:**
- Takes **query features** + **task embeddings** + **skip connections**
- Progressively upsamples through 4 stages
- Each stage uses:
  - **Bidirectional Cross-Attention (BCA)**: Query ↔ Task
  - **FiLM Modulation**: Task-conditioned feature scaling
  - **Upsampling + Skip Connection Fusion**

**Stage-by-Stage Upsampling:**
```python
# Starting from deepest features: (B, 512, 1, 16, 16)

# Stage 1: Upsample to (2, 32, 32)
features = bca(features, task_embeddings)        # Bidirectional attention
features = film_modulate(features, task_embeddings)  # Task conditioning
features = upsample(features)                    # (B, 256, 2, 32, 32)
features = concat([features, skip_3], dim=1)     # Merge with skip connection
features = conv(features)                        # Reduce channels

# Stage 2: Upsample to (4, 64, 64)
features = bca(features, task_embeddings)
features = film_modulate(features, task_embeddings)
features = upsample(features)                    # (B, 128, 4, 64, 64)
features = concat([features, skip_2], dim=1)
features = conv(features)

# Stage 3: Upsample to (8, 128, 128)
features = bca(features, task_embeddings)
features = film_modulate(features, task_embeddings)
features = upsample(features)                    # (B, 64, 8, 128, 128)
features = concat([features, skip_1], dim=1)
features = conv(features)

# Stage 4: Upsample to (16, 256, 256)
features = bca(features, task_embeddings)
features = film_modulate(features, task_embeddings)
features = upsample(features)                    # (B, 32, 16, 256, 256)
features = concat([features, skip_0], dim=1)
features = conv(features)

# Final prediction head
logits = final_conv(features)                    # (B, 1, 16, 256, 256)
```

**Output:**
- **Logits**: `(B, 1, D, H, W)` - Raw predictions (apply sigmoid for probabilities)
- **Updated Tokens**: `(B, K, m+1, C)` - Task tokens after bidirectional attention

---

#### **Step 6: Create Memory Bank (Optional)** (Lines 58-60)
```python
self.memory_bank: Optional[ClassMemoryBank] = (
    ClassMemoryBank(momentum=memory_momentum) if use_memory_bank else None
)
```

**What the Memory Bank Does:**
- Stores **learned prototypes** (task embeddings) for each class
- Uses **Exponential Moving Average (EMA)** for stable updates
- Enables **zero-shot retrieval** for previously seen classes

**Update Formula:**
```python
# After each training episode:
new_memory = α * old_memory + (1 - α) * new_embedding
# Where α = 0.999 (momentum)
```

**Usage:**
```python
# During inference for known class:
task_embedding = memory_bank.retrieve(class_id=5)
prediction = model(query_img, task_embedding)  # No support set needed!
```

---

## Method 1: `encode_support()`

### **Lines 61-67**: Support Set Encoding

```python
def encode_support(
    self,
    support_images: torch.Tensor,     # (B, 1, D, H, W)
    support_masks: torch.Tensor,      # (B, K, D, H, W)
) -> Dict[str, torch.Tensor]:
```

**Purpose:** Convert support set (annotated examples) into **task embeddings** that encode "what to segment".

### **Execution Flow:**

#### **Step 1: Extract Support Features** (Line 64)
```python
encoder_out = self.encoder(support_images)
```

**Input:**
```python
support_images: (2, 1, 16, 256, 256)  # Batch of 2 support examples
```

**Process:**
- Passes through encoder's 4 downsampling stages
- Extracts multi-scale features

**Output:**
```python
encoder_out = EncoderOutput(
    features=torch.Tensor(2, 512, 1, 16, 16),      # Deepest features
    skip_connections=[
        torch.Tensor(2, 32, 16, 256, 256),  # Skip 0
        torch.Tensor(2, 64, 8, 128, 128),   # Skip 1
        torch.Tensor(2, 128, 4, 64, 64),    # Skip 2
        torch.Tensor(2, 256, 2, 32, 32),    # Skip 3
    ]
)
```

---

#### **Step 2: Encode Task** (Line 65)
```python
return self.task_encoder(encoder_out.features, support_masks)
```

**Input:**
```python
features: (2, 512, 1, 16, 16)     # Support features from encoder
masks:    (2, 1, 16, 256, 256)    # Support masks (K=1 class)
```

**Process:**
1. **Foreground encoding**: Masked average pooling
2. **Contextual encoding**: Learnable query tokens + attention
3. **Combine**: Concatenate both pathways

**Output:**
```python
{
    "task_embeddings": torch.Tensor(2, 1, 9, 512),
    # Shape breakdown:
    #   2   = Batch size
    #   1   = Number of classes (K)
    #   9   = m+1 tokens (1 foreground + 8 contextual)
    #   512 = Feature channels
    
    "foreground_embeddings": torch.Tensor(2, 1, 1, 512),
    "context_tokens": torch.Tensor(2, 1, 8, 512),
}
```

### **Real-World Example:**

```python
# Training iteration: Sample a polyp image
sample = train_dataset[42]  # Random polyp endoscopy image

support_img = sample["image"].unsqueeze(0)   # (1, 1, 16, 256, 256)
support_mask = sample["mask"].unsqueeze(0)   # (1, 1, 16, 256, 256)

# Encode what a "polyp" looks like
task_info = model.encode_support(support_img, support_mask)
task_embeddings = task_info["task_embeddings"]  # (1, 1, 9, 512)

# Now we can segment OTHER polyp images using these embeddings!
```

---

## Method 2: `forward()`

### **Lines 111-127**: Query Image Segmentation

```python
def forward(
    self,
    query_images: torch.Tensor,              # (B, 1, D, H, W)
    task_embeddings: torch.Tensor,           # (B, K, m+1, C)
    *,
    skip_connections: Sequence[torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
```

**Purpose:** Generate segmentation predictions for **query images** using **task embeddings**.

### **Execution Flow:**

#### **Step 1: Encode Query** (Line 120)
```python
encoder_out = self.encoder(query_images)
```

**Input:**
```python
query_images: (1, 1, 16, 256, 256)  # New unseen image to segment
```

**Process:**
- Same encoder as support (shared weights!)
- Extracts features at multiple scales

**Output:**
```python
encoder_out = EncoderOutput(
    features=torch.Tensor(1, 512, 1, 16, 16),      # Query features
    skip_connections=[...]                         # 4 skip tensors
)
```

---

#### **Step 2: Select Skip Connections** (Line 121)
```python
skips = skip_connections or encoder_out.skip_connections
```

**Why Optional Skip Connections?**
- Usually: Use query's own skip connections
- Advanced: Can provide support's skip connections for cross-attention experiments

---

#### **Step 3: Decode Prediction** (Lines 122-126)
```python
decoder_out = self.mask_decoder(
    encoder_out.features,      # Query features: (1, 512, 1, 16, 16)
    skips,                     # 4 skip connections
    task_embeddings,           # Task info: (1, 1, 9, 512)
)
```

**Process Inside Decoder:**

```python
# Initial: features = (1, 512, 1, 16, 16)
#          tokens = task_embeddings (1, 1, 9, 512)

# === DECODER STAGE 1 === (Upsample to 2×32×32)
# Bidirectional Cross-Attention
query_attn, _ = cross_attn(query=features, key=tokens, value=tokens)
token_attn, _ = cross_attn(query=tokens, key=features, value=features)
features = features + query_attn
tokens = tokens + token_attn

# FiLM Modulation
gamma, beta = film_predictor(tokens.mean(dim=2))  # (1, 1, 512), (1, 1, 512)
features = gamma * features + beta  # Task-conditioned scaling

# Upsample + Skip Fusion
features = upsample_conv(features)            # (1, 256, 2, 32, 32)
features = concat([features, skips[3]], dim=1)  # Merge skip_3
features = conv(features)                     # Reduce channels

# === DECODER STAGE 2 === (Upsample to 4×64×64)
# ... repeat BCA + FiLM + Upsample ...

# === DECODER STAGE 3 === (Upsample to 8×128×128)
# ... repeat BCA + FiLM + Upsample ...

# === DECODER STAGE 4 === (Upsample to 16×256×256)
# ... repeat BCA + FiLM + Upsample ...

# === FINAL PREDICTION HEAD ===
logits = final_conv(features)  # (1, 1, 16, 256, 256)
```

**Output:**
```python
decoder_out = DecoderOutput(
    logits=torch.Tensor(1, 1, 16, 256, 256),         # Raw predictions
    updated_tokens=torch.Tensor(1, 1, 9, 512),       # Refined task tokens
)
```

---

#### **Step 4: Return Results** (Lines 127-131)
```python
return {
    "logits": decoder_out.logits,                   # (1, 1, 16, 256, 256)
    "tokens": decoder_out.updated_tokens,           # (1, 1, 9, 512)
    "skip_connections": encoder_out.skip_connections,
}
```

### **Real-World Example:**

```python
# Step 1: Get task embeddings from support set
task_emb = model.encode_support(support_img, support_mask)["task_embeddings"]

# Step 2: Segment a NEW query image
query_img = test_dataset[99]["image"].unsqueeze(0)  # (1, 1, 16, 256, 256)
outputs = model(query_img, task_emb)

# Step 3: Get prediction
logits = outputs["logits"]              # (1, 1, 16, 256, 256)
prediction = torch.sigmoid(logits) > 0.5  # Binary mask

# Step 4: Visualize middle slice
pred_slice = prediction[0, 0, 8].cpu().numpy()  # Extract slice 8
plt.imshow(pred_slice, cmap='gray')
plt.title("Segmented Polyp")
```

---

## Method 3: Memory Bank Operations

### **Method 3a: `update_memory_bank()`** (Lines 69-76)

```python
def update_memory_bank(
    self,
    task_embeddings: torch.Tensor,              # (B, K, m+1, C)
    class_ids: Sequence[Sequence[int]] | Sequence[int],
) -> None:
```

**Purpose:** Store learned task embeddings in memory bank for future zero-shot retrieval.

**When Called:** After each training episode

**Process:**
```python
# During training loop:
# 1. Encode support → get task embeddings
task_emb = model.encode_support(support_img, support_mask)["task_embeddings"]

# 2. Train on query
outputs = model(query_img, task_emb)
loss = criterion(outputs["logits"], query_mask)
loss.backward()
optimizer.step()

# 3. Update memory bank with learned embeddings
model.update_memory_bank(
    task_embeddings=task_emb.detach(),  # Detach from computation graph
    class_ids=[[0]]                     # Class ID for this episode
)
```

**Inside Memory Bank:**
```python
# Exponential Moving Average (EMA) update
if class_id in memory:
    # Update existing prototype
    memory[class_id] = 0.999 * memory[class_id] + 0.001 * new_embedding
else:
    # Initialize new class
    memory[class_id] = new_embedding
```

**Benefits:**
- Accumulates knowledge across all training episodes
- Smooth, stable prototypes (EMA prevents oscillation)
- Enables zero-shot inference later

---

### **Method 3b: `retrieve_memory_embeddings()`** (Lines 78-96)

```python
def retrieve_memory_embeddings(
    self,
    class_ids: Sequence[int],                    # [0, 1, 2]
    *,
    fallback: Optional[torch.Tensor] = None,     # Default if not found
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
```

**Purpose:** Retrieve stored prototypes for **zero-shot inference** (no support set needed!).

**When Used:** Inference on known classes

**Process:**
```python
# Inference without support set!
class_id = 0  # "Polyp" class seen during training

# Retrieve learned prototype from memory
task_emb = model.retrieve_memory_embeddings(
    class_ids=[class_id],
    device=device
)  # (1, 1, 9, 512)

# Directly predict on new query
query_img = new_polyp_image.unsqueeze(0)
outputs = model(query_img, task_emb)
prediction = torch.sigmoid(outputs["logits"]) > 0.5

# Inference time: ~35ms (fastest strategy!)
```

**Error Handling:**
```python
if self.memory_bank is None:
    raise RuntimeError("Memory bank is disabled for this model instance.")
```

---

## Method 4: In-Context Tuner

### **Lines 98-109**: Create Tuner Instance

```python
def create_in_context_tuner(
    self,
    *,
    lr: float = 1e-3,      # Learning rate for tuning
    steps: int = 20,       # Number of gradient steps
) -> "InContextTuner":
```

**Purpose:** Create an **In-Context Tuner** for rapid task adaptation.

**How It Works:**
1. Start with initial task embeddings (from support set or memory)
2. **Optimize only the embeddings** (model weights frozen!)
3. Minimize loss on small validation set
4. Return tuned embeddings for better predictions

**Usage Example:**
```python
# Step 1: Get initial embeddings from support
initial_emb = model.encode_support(support_img, support_mask)["task_embeddings"]

# Step 2: Create tuner
tuner = model.create_in_context_tuner(lr=1e-3, steps=20)

# Step 3: Tune on query image with initial prediction
tuned_emb = tuner.tune(
    query_images=query_img,
    query_masks=initial_prediction,  # Use initial prediction as target
    initial_embeddings=initial_emb,
    steps=20,
    update_memory=False
)

# Step 4: Final prediction with tuned embeddings
outputs = model(query_img, tuned_emb)
prediction = torch.sigmoid(outputs["logits"]) > 0.5

# Accuracy boost: +7.1% Dice over one-shot!
# Time: ~2 seconds (20 gradient steps)
```

**Why It Works:**
- Embeddings adapt to query's specific characteristics
- No overfitting (model weights frozen)
- Fast (only updating small embedding tensors)

---

## Data Flow Examples

### **Example 1: One-Shot Training**

```python
# === SETUP ===
model = IrisModel(
    in_channels=1,
    base_channels=32,
    num_query_tokens=8,
    volume_shape=(16, 256, 256),
    use_memory_bank=True
).to(device)

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = DiceCrossEntropyLoss()

# === TRAINING ITERATION ===
# Sample: chest x-ray with lung mask
sample = train_dataset[42]
image = sample["image"].unsqueeze(0).to(device)  # (1, 1, 16, 256, 256)
mask = sample["mask"].unsqueeze(0).to(device)    # (1, 1, 16, 256, 256)

# Self-supervised: use same image as support and query
support_img, query_img = image, image
support_mask, query_mask = mask, mask

# 1. Encode support set
support_out = model.encode_support(support_img, support_mask)
task_embeddings = support_out["task_embeddings"]  # (1, 1, 9, 512)

# 2. Forward pass on query
outputs = model(query_img, task_embeddings)
logits = outputs["logits"]  # (1, 1, 16, 256, 256)

# 3. Compute loss
loss = criterion(logits, query_mask)

# 4. Backpropagation
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 5. Update memory bank
model.update_memory_bank(
    task_embeddings=task_embeddings.detach(),
    class_ids=[[0]]  # Lung class
)

print(f"Loss: {loss.item():.4f}")
```

---

### **Example 2: Ensemble Inference (K=3)**

```python
# === SETUP ===
model.eval()  # Evaluation mode

# === LOAD 3 SUPPORT EXAMPLES ===
support_imgs = []
support_masks = []

for idx in [10, 25, 47]:  # Random support indices
    sample = train_dataset[idx]
    support_imgs.append(sample["image"])
    support_masks.append(sample["mask"])

support_imgs = torch.stack(support_imgs).to(device)   # (3, 1, 16, 256, 256)
support_masks = torch.stack(support_masks).to(device) # (3, 1, 16, 256, 256)

# === ENCODE EACH SUPPORT SEPARATELY ===
all_embeddings = []

with torch.no_grad():
    for i in range(3):
        supp_img = support_imgs[i:i+1]   # (1, 1, 16, 256, 256)
        supp_mask = support_masks[i:i+1] # (1, 1, 16, 256, 256)
        
        support_out = model.encode_support(supp_img, supp_mask)
        all_embeddings.append(support_out["task_embeddings"])

# === AVERAGE EMBEDDINGS ===
task_embeddings = torch.stack(all_embeddings).mean(dim=0)  # (1, 1, 9, 512)

# === PREDICT ON QUERY ===
query_img = test_dataset[99]["image"].unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(query_img, task_embeddings)
    logits = outputs["logits"]
    prediction = (torch.sigmoid(logits) > 0.5).float()

# Dice score: ~85.1% (vs 81.5% one-shot)
```

---

### **Example 3: Zero-Shot Memory Retrieval**

```python
# === AFTER TRAINING ===
# Memory bank contains learned prototypes for classes 0, 1, 2

model.eval()

# === INFERENCE ON NEW IMAGE ===
query_img = new_chest_xray.unsqueeze(0).to(device)  # (1, 1, 16, 256, 256)

# Retrieve "lung" prototype (class 0)
with torch.no_grad():
    task_embeddings = model.retrieve_memory_embeddings(
        class_ids=[0],
        device=device
    )  # (1, 1, 9, 512)
    
    # Direct prediction (no support set needed!)
    outputs = model(query_img, task_embeddings)
    prediction = torch.sigmoid(outputs["logits"]) > 0.5

# Inference time: 35ms (fastest!)
# No support set required
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         IRIS MODEL (core.py)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐                                           │
│  │  SUPPORT SET     │                                           │
│  │  Images: (B,1,D,H,W)                                         │
│  │  Masks:  (B,K,D,H,W)                                         │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  ENCODER         │                                           │
│  │  (3D UNet)       │                                           │
│  │  • 4 stages      │                                           │
│  │  • ResBlocks     │                                           │
│  │  • InstanceNorm  │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           │ Features: (B, 512, d, h, w)                         │
│           │ Skips: 4 tensors                                    │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  TASK ENCODER    │                                           │
│  │  (TEM)           │                                           │
│  │  ┌──────────┐ ┌─────────────┐                               │
│  │  │Foreground│ │ Contextual  │                               │
│  │  │Encoding  │ │ Encoding    │                               │
│  │  │(Masked   │ │ (8 Query    │                               │
│  │  │ Pool)    │ │  Tokens)    │                               │
│  │  └────┬─────┘ └──────┬──────┘                               │
│  │       │              │                                        │
│  │       └──────┬───────┘                                       │
│  │              │                                                │
│  └──────────────┼────────┘                                      │
│                 │                                                │
│                 │ Task Embeddings: (B, K, 9, 512)               │
│                 │                                                │
│                 ├──────────────────────────────┐                │
│                 │                               │                │
│                 ▼                               │                │
│  ┌──────────────────────┐                      │                │
│  │  MEMORY BANK         │                      │                │
│  │  (Optional)          │                      │                │
│  │  • Store prototypes  │◄─────update─────────┤                │
│  │  • EMA momentum 0.999│                      │                │
│  │  • Zero-shot retrieval                      │                │
│  └──────────────────────┘                      │                │
│                                                 │                │
│  ┌──────────────────┐                          │                │
│  │  QUERY IMAGE     │                          │                │
│  │  (B, 1, D, H, W) │                          │                │
│  └────────┬─────────┘                          │                │
│           │                                     │                │
│           ▼                                     │                │
│  ┌──────────────────┐                          │                │
│  │  ENCODER         │                          │                │
│  │  (Same as above) │                          │                │
│  └────────┬─────────┘                          │                │
│           │                                     │                │
│           │ Query Features + Skips             │                │
│           ▼                                     │                │
│  ┌──────────────────┐                          │                │
│  │  DECODER         │◄─────Task Embeddings────┘                │
│  │  (Bidirectional) │                                           │
│  │  ┌─────────────┐ │                                           │
│  │  │  Stage 1    │ │                                           │
│  │  │  BCA + FiLM │ │                                           │
│  │  │  Upsample   │ │                                           │
│  │  └─────┬───────┘ │                                           │
│  │  ┌─────▼───────┐ │                                           │
│  │  │  Stage 2    │ │                                           │
│  │  │  BCA + FiLM │ │                                           │
│  │  │  Upsample   │ │                                           │
│  │  └─────┬───────┘ │                                           │
│  │  ┌─────▼───────┐ │                                           │
│  │  │  Stage 3    │ │                                           │
│  │  │  BCA + FiLM │ │                                           │
│  │  │  Upsample   │ │                                           │
│  │  └─────┬───────┘ │                                           │
│  │  ┌─────▼───────┐ │                                           │
│  │  │  Stage 4    │ │                                           │
│  │  │  BCA + FiLM │ │                                           │
│  │  │  Upsample   │ │                                           │
│  │  └─────┬───────┘ │                                           │
│  └────────┼─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  PREDICTION      │                                           │
│  │  Logits: (B,1,D,H,W)                                         │
│  │  Sigmoid → Binary Mask                                       │
│  └──────────────────┘                                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

### **1. Component Responsibilities**

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **Encoder** | Images (B,1,D,H,W) | Features + Skips | Extract multi-scale features |
| **Task Encoder** | Support Features + Masks | Task Embeddings (B,K,9,512) | Encode "what to segment" |
| **Decoder** | Query Features + Task Emb | Logits (B,1,D,H,W) | Generate segmentation |
| **Memory Bank** | Task Embeddings + Class IDs | Stored Prototypes | Enable zero-shot retrieval |
| **In-Context Tuner** | Initial Embeddings | Tuned Embeddings | Rapid task adaptation |

### **2. Data Shapes Throughout Pipeline**

```python
Input Image:           (B, 1, 16, 256, 256)    # Raw volume
  ↓ Encoder
Features:              (B, 512, 1, 16, 16)     # Compressed features
Skip Connections:      4 tensors at [2,4,8,16] scales
  ↓ Task Encoder
Task Embeddings:       (B, K, 9, 512)          # K classes, 9 tokens each
  ↓ Decoder
Logits:                (B, 1, 16, 256, 256)    # Restored to input size
  ↓ Sigmoid
Binary Mask:           (B, 1, 16, 256, 256)    # Final prediction
```

### **3. Four Inference Strategies**

1. **Memory Retrieval** (35ms): Retrieve stored prototype, zero-shot
2. **One-Shot** (52ms): Single support example encoding
3. **Ensemble K=3** (78ms): Average 3 support embeddings ⭐ **Recommended**
4. **In-Context Tuning** (2s): 20 gradient steps on embeddings

### **4. Design Philosophy**

- **Modularity**: Each component (encoder, TEM, decoder) is independent
- **Flexibility**: Support multiple inference strategies
- **Efficiency**: Shared encoder for support and query
- **Adaptability**: In-context tuning without retraining full model

---

**The `core.py` file is the **heart** of IRIS—it orchestrates all components to achieve few-shot medical image segmentation with minimal annotated data!**
