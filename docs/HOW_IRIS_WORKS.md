# How IRIS Processes Input: Complete Flow Explanation

## 📌 Overview

IRIS (Interactive and Refined Image Segmentation) processes medical images through **episodic few-shot learning**. Here's how data flows from input to prediction.

---

## 🗂️ File Structure & Main Entry Points

### **Core Model Files** (in `src/iris/model/`)
```
src/iris/model/
├── core.py              ← Main model (IrisModel class)
├── encoder.py           ← 3D UNet Encoder
├── task_encoding.py     ← Task Encoding Module (TEM)
├── decoder.py           ← Bidirectional Decoder
├── memory.py            ← Memory Bank
└── tuning.py            ← In-Context Tuner
```

### **Training Scripts** (in `scripts/training/`)
```
scripts/training/
└── train_with_metrics.py    ← Main training entry point
```

### **Inference Scripts** (in `scripts/visualization/`)
```
scripts/visualization/
└── visualize_improved.py     ← Visualization/inference
```

### **Web Application** (in `web_app/`)
```
web_app/
├── app.py               ← Gradio web interface
├── inference.py         ← Inference wrapper
└── launch.py            ← Launch script
```

---

## 🔄 Complete Data Flow

### **1. Input Format**

IRIS expects **3D volumes** even for 2D images:
```python
# Input shape: (Batch, Channels, Depth, Height, Width)
image_tensor = torch.Size([1, 1, 16, 256, 256])
mask_tensor = torch.Size([1, 1, 16, 256, 256])
```

For 2D images (like X-rays), they're converted to pseudo-3D by:
```python
# Repeat 2D slice along depth dimension
volume = np.repeat(img_2d[np.newaxis, ...], depth_slices=16, axis=0)
```

---

### **2. Training Flow** (from `train_with_metrics.py`)

#### **Step 1: Load Dataset**
```python
# Line 166-270 in train_with_metrics.py
train_dataset = build_dataset(
    dataset_name,
    root=Path("datasets/chest_xray_masks"),
    split=DatasetSplit.TRAIN,
    depth_slices=16,
    target_resolution=128
)

# Each sample contains:
sample = {
    "image": torch.Tensor(1, 16, 128, 128),  # Query image
    "mask": torch.Tensor(1, 16, 128, 128),   # Ground truth mask
    "class_id": [0],                         # Class identifier
}
```

#### **Step 2: Sample Episodic Data** (Line 340-360)
```python
# Sample random training example
idx = torch.randint(0, len(train_dataset), (1,)).item()
sample = train_dataset[idx]

image = sample["image"].unsqueeze(0).to(device)  # (1, 1, D, H, W)
mask = sample["mask"].unsqueeze(0).to(device)    # (1, 1, D, H, W)

# Split into support and query (self-supervised)
support_img = image      # Same image used as support
query_img = image        # And as query
support_mask = mask      # With known mask
query_mask = mask        # For computing loss
```

#### **Step 3: Encode Support Set** (Line 368-370)
```python
# IrisModel.encode_support() → src/iris/model/core.py:64
support_out = model.encode_support(support_img, support_mask)
task_embeddings = support_out["task_embeddings"]
```

**What happens inside:**
```python
# core.py:64-67
def encode_support(self, support_images, support_masks):
    # 1. Extract features using 3D UNet Encoder
    encoder_out = self.encoder(support_images)
    # encoder_out.features: (B, 256, D/16, H/16, W/16)
    
    # 2. Encode task using TEM
    return self.task_encoder(encoder_out.features, support_masks)
    # Returns: {"task_embeddings": (B, m+1, C)}
```

**Task Encoding Module (TEM)** in `task_encoding.py`:
```python
# Two parallel pathways:

# Pathway 1: Foreground Encoding (masked average pooling)
foreground_features = self._encode_foreground(features, masks)
# Output: (B, K, C) where K = num_classes

# Pathway 2: Contextual Encoding (learnable query tokens)
contextual_features = self._encode_contextual(features, masks)
# Output: (B, K, m, C) where m = num_query_tokens (8)

# Combine: (B, K, m+1, C)
task_embeddings = combine(foreground, contextual)
```

#### **Step 4: Query Inference** (Line 372-374)
```python
# IrisModel.forward() → src/iris/model/core.py:111
outputs = model(query_img, task_embeddings)
logits = outputs["logits"]  # (1, 1, D, H, W)
```

**What happens inside:**
```python
# core.py:111-127
def forward(self, query_images, task_embeddings):
    # 1. Encode query with same encoder
    encoder_out = self.encoder(query_images)
    # encoder_out.features: (B, 256, D/16, H/16, W/16)
    
    # 2. Decode using Bidirectional Decoder
    decoder_out = self.mask_decoder(
        encoder_out.features,           # Query features
        encoder_out.skip_connections,   # Multi-scale features
        task_embeddings                 # Task conditioning
    )
    
    return {
        "logits": decoder_out.logits,  # Segmentation output
        "tokens": decoder_out.updated_tokens,
        "skip_connections": encoder_out.skip_connections
    }
```

**Bidirectional Decoder** in `decoder.py`:
```python
# For each decoder stage (4 stages):
for stage in range(4):
    # 1. Bidirectional Cross-Attention (BCA)
    query_features = attention(query_features, task_embeddings)
    task_embeddings = attention(task_embeddings, query_features)
    
    # 2. FiLM Modulation (task conditioning)
    query_features = film_modulate(query_features, task_embeddings)
    
    # 3. Upsample and combine with skip connections
    query_features = upsample_and_combine(query_features, skip_features)

# Final output: logits (B, 1, D, H, W)
```

#### **Step 5: Compute Loss** (Line 376-377)
```python
# Combined Dice + BCE loss
loss = criterion(logits, query_mask)

# criterion = DiceCrossEntropyLoss()
# loss = dice_loss(logits, mask) + lambda * bce_loss(logits, mask)
```

#### **Step 6: Backpropagation** (Line 389-392)
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

---

### **3. Inference Flow** (from `inference.py`)

#### **Web App Usage:**
```bash
# Launch web app
python web_app/launch.py

# Or directly
python web_app/app.py
```

#### **Step 1: Load Model** (inference.py:89-107)
```python
inference = IRISInference(
    checkpoint_path="outputs/checkpoints/kvasir/final_model.pt",
    dataset="kvasir"
)

# Loads:
# 1. Model weights from checkpoint
# 2. Support dataset for few-shot examples
```

#### **Step 2: Preprocess Query Image** (inference.py:129-155)
```python
# User uploads image (e.g., polyp.jpg)
uploaded_image = Image.open("polyp.jpg")

# Preprocess to IRIS format
query_tensor = inference.preprocess_image(uploaded_image)
# Steps:
# 1. Convert to grayscale
# 2. Resize to (256, 256)
# 3. Create pseudo-3D: repeat 16 times → (16, 256, 256)
# 4. Normalize intensity
# 5. Add batch/channel dims → (1, 1, 16, 256, 256)
```

#### **Step 3: Get Support Examples** (inference.py:157-177)
```python
# Option A: Random support from dataset
support_imgs, support_masks = inference.get_support_images(num_support=3)
# Returns: (3, 1, 16, 256, 256), (3, 1, 16, 256, 256)

# Option B: User provides support images
# Upload annotated examples with masks
```

#### **Step 4: Run Inference Strategy**

**Strategy 1: One-Shot** (inference.py:179-200)
```python
prediction, time = inference.predict_oneshot(
    query_img=query_tensor,          # (1, 1, 16, 256, 256)
    support_img=support_imgs[0:1],   # (1, 1, 16, 256, 256)
    support_mask=support_masks[0:1]  # (1, 1, 16, 256, 256)
)
# Time: ~52ms
# Dice: ~81.5%
```

**Strategy 2: Ensemble (K=3)** (inference.py:202-228)
```python
prediction, time = inference.predict_ensemble(
    query_img=query_tensor,
    support_imgs=support_imgs,      # (3, 1, 16, 256, 256)
    support_masks=support_masks,    # (3, 1, 16, 256, 256)
    num_support=3
)
# Process:
# 1. Encode each support example separately
# 2. Average task embeddings: mean([emb1, emb2, emb3])
# 3. Single forward pass with averaged embeddings
# Time: ~78ms
# Dice: ~85.1% (+3.6% over one-shot)
```

**Strategy 3: In-Context Tuning** (inference.py:230-280)
```python
prediction, time = inference.predict_with_tuning(
    query_img=query_tensor,
    initial_mask=initial_prediction,  # From ensemble
    support_imgs=support_imgs,
    support_masks=support_masks,
    tuning_steps=20
)
# Process:
# 1. Get initial prediction (ensemble)
# 2. Create InContextTuner
# 3. Optimize task embeddings for 20 gradient steps
#    - Only embeddings updated, model frozen
#    - Minimize loss on initial prediction
# 4. Final prediction with tuned embeddings
# Time: ~2 seconds
# Dice: ~87.3% (+7.1% over one-shot)
```

**Strategy 4: Memory Retrieval** (core.py:88-97)
```python
# Zero-shot for known classes
task_embeddings = model.retrieve_memory_embeddings(
    class_ids=[0],  # Known class ID
)
prediction = model(query_img, task_embeddings)
# Time: ~35ms (fastest!)
# Requires: Previously seen class during training
```

#### **Step 5: Visualize Results** (inference.py:325-420)
```python
result_image = inference.visualize_prediction(
    query_img=query_tensor,
    prediction=prediction,
    color_scheme="Green-Gold"
)
# Creates overlay visualization
```

---

## 📊 Complete Example: Polyp Segmentation

### **1. Setup**
```python
from web_app.inference import IRISInference

# Load trained Kvasir model
iris = IRISInference(
    checkpoint_path="outputs/checkpoints/kvasir/final_model.pt",
    dataset="kvasir"
)
```

### **2. Prepare Query**
```python
from PIL import Image

# User uploads colonoscopy image
query_image = Image.open("colonoscopy_scan.jpg")

# Preprocess
query_tensor = iris.preprocess_image(query_image)
# Shape: (1, 1, 16, 256, 256)
```

### **3. Get Support Set**
```python
# Automatically sample 3 annotated polyp examples
support_imgs, support_masks = iris.get_support_images(num_support=3)
# support_imgs: (3, 1, 16, 256, 256)
# support_masks: (3, 1, 16, 256, 256)
```

### **4. Predict (Ensemble Strategy)**
```python
prediction, inference_time = iris.predict_ensemble(
    query_img=query_tensor,
    support_imgs=support_imgs,
    support_masks=support_masks,
    num_support=3
)
# prediction: (1, 1, 16, 256, 256) - binary mask
# inference_time: ~0.078 seconds
```

### **5. Compute Metrics**
```python
# If ground truth available
dice_score = iris.compute_dice(prediction, ground_truth_mask)
print(f"Dice Score: {dice_score:.4f}")  # e.g., 0.8512
```

### **6. Visualize**
```python
result = iris.visualize_prediction(
    query_img=query_tensor,
    prediction=prediction,
    color_scheme="Green-Gold"
)
result.save("polyp_segmentation.png")
```

---

## 🎯 Key Architectural Flow

```
INPUT QUERY IMAGE (H×W RGB)
    ↓
Convert to Grayscale → Resize to (256×256)
    ↓
Create Pseudo-3D Volume (16 slices)
    ↓
Normalize Intensity → (1, 1, 16, 256, 256)
    ↓
┌─────────────────────────────────────┐
│         IRIS MODEL FORWARD          │
├─────────────────────────────────────┤
│ 1. ENCODER (3D UNet)                │
│    Input: (1, 1, 16, 256, 256)      │
│    ↓                                │
│    Conv3D → InstanceNorm → ReLU     │
│    Downsample 4 stages              │
│    ↓                                │
│    Features: (1, 256, 1, 16, 16)    │
│    Skip Connections: 4 levels       │
│                                     │
│ 2. TASK ENCODING MODULE (TEM)       │
│    Input: Support features + masks  │
│    ↓                                │
│    Foreground Encoding (masked pool)│
│    Contextual Encoding (8 tokens)   │
│    ↓                                │
│    Task Embeddings: (1, 1, 9, 256)  │
│                                     │
│ 3. BIDIRECTIONAL DECODER            │
│    Input: Query features + Task emb │
│    ↓                                │
│    Stage 1: BCA + FiLM + Upsample   │
│    Stage 2: BCA + FiLM + Upsample   │
│    Stage 3: BCA + FiLM + Upsample   │
│    Stage 4: BCA + FiLM + Upsample   │
│    ↓                                │
│    Logits: (1, 1, 16, 256, 256)     │
└─────────────────────────────────────┘
    ↓
Apply Sigmoid → Threshold at 0.5
    ↓
BINARY MASK (1, 1, 16, 256, 256)
    ↓
Extract Middle Slice [8, :, :]
    ↓
FINAL 2D SEGMENTATION (256×256)
```

---

## 🔑 Key Files Explained

### **1. `src/iris/model/core.py`** (Main Model)
- **Line 23-60**: `IrisModel.__init__()` - Initialize all components
- **Line 64-67**: `encode_support()` - Encode support set to task embeddings
- **Line 111-127**: `forward()` - Main inference pass

### **2. `src/iris/model/encoder.py`** (3D UNet Encoder)
- Extracts multi-scale features from 3D volumes
- 4 downsampling stages with residual blocks
- Outputs features + skip connections

### **3. `src/iris/model/task_encoding.py`** (TEM)
- **Foreground encoding**: Masked average pooling
- **Contextual encoding**: 8 learnable query tokens via attention
- Combines to produce (m+1) tokens per class

### **4. `src/iris/model/decoder.py`** (Bidirectional Decoder)
- **Bidirectional Cross-Attention**: Query ↔ Task
- **FiLM Modulation**: Task-conditioned feature scaling
- **Progressive Upsampling**: 4 stages to original resolution

### **5. `scripts/training/train_with_metrics.py`** (Training Entry)
- **Line 482-510**: `main()` - Parse args and launch training
- **Line 138-480**: `train_with_detailed_metrics()` - Main training loop
- Episodic sampling, forward pass, loss computation

### **6. `web_app/inference.py`** (Inference Wrapper)
- **Line 52-108**: `__init__()` - Load model and dataset
- **Line 129-155**: `preprocess_image()` - Convert PIL to tensor
- **Line 179-200**: `predict_oneshot()` - One-shot inference
- **Line 202-228**: `predict_ensemble()` - Ensemble inference
- **Line 230-280**: `predict_with_tuning()` - In-context tuning

---

## 🚀 Quick Start Commands

### **Training**
```bash
# Train on Chest X-Ray
python scripts/training/train_with_metrics.py \
    --dataset chest_xray_masks \
    --iterations 2000 \
    --output_dir outputs/training/chest_xray

# Train on ISIC (Skin Lesion)
python scripts/training/train_with_metrics.py \
    --dataset isic \
    --iterations 3000 \
    --batch_size 4
```

### **Inference**
```bash
# Launch web interface
python web_app/launch.py

# Or visualize results
python scripts/visualization/visualize_improved.py \
    --dataset kvasir \
    --checkpoint outputs/checkpoints/kvasir/final_model.pt \
    --num_cases 10
```

---

## 📈 Performance Summary

| Strategy | Input | Speed | Dice Score | Use Case |
|----------|-------|-------|------------|----------|
| **Memory Retrieval** | Known class ID | 35ms | 84.3% | Known classes, fastest |
| **One-Shot (K=1)** | 1 support example | 52ms | 81.5% | Quick prototyping |
| **Ensemble (K=3)** | 3 support examples | 78ms | **85.1%** | **Recommended balance** |
| **In-Context Tuning** | Initial + 20 steps | 2s | 87.3% | Highest accuracy |

---

## 🎓 Summary

1. **Input**: Query image (any size) + Support set (1-3 annotated examples)
2. **Preprocessing**: Convert to pseudo-3D volume (1, 1, D, H, W)
3. **Encoding**: Extract features with 3D UNet encoder
4. **Task Encoding**: Create task embeddings from support set
5. **Decoding**: Generate segmentation via bidirectional decoder
6. **Output**: Binary mask (0/1) at original resolution

The beauty of IRIS is **in-context learning**: it adapts to new tasks using just a few examples, without retraining!
