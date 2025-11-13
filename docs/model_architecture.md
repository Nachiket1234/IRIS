# IRIS Model Architecture – Detailed Documentation

This document provides an in-depth walkthrough of the IRIS model (Section 3 of the paper), mapping every architectural component to the corresponding implementation files and functions in this repository. It covers tensor shapes, intermediate representations, design rationales, and extensibility notes.

---

## 1. High-Level Composition

The IRIS model is instantiated via `iris.model.core.IrisModel`. Internally it assembles three subsystems:

1. **Medical3DUNetEncoder** (`encoder.py`): feature extractor with four residual downsampling stages optimised for 128³ medical volumes.
2. **TaskEncodingModule** (`task_encoding.py`): converts support embeddings and segmentation masks into task tokens as described in Section 3.2.1 (foreground summary + contextual tokens).
3. **MaskDecoder** (`decoder.py`): performs bidirectional cross-attention between query features and task tokens and reconstructs per-class segmentation masks in a single forward pass.

Additional infrastructure:

- **ClassMemoryBank** (`memory.py`): persistent EMA store for per-class task embeddings (context ensemble, object retrieval).
- **InContextTuner** (`tuning.py`): gradient-based adaptation that updates only task embeddings while freezing all network weights.

`IrisModel` exposes:

- `encode_support(images, masks) → dict`: returns task embeddings, foreground tokens, context tokens.
- `forward(query_images, task_embeddings, skip_connections=None) → dict`: decodes logits and returns updated tokens + skip connections.
- `update_memory_bank(task_embeddings, class_ids)`: EMA update for memory.
- `retrieve_memory_embeddings(class_ids)`: fetch stored embeddings.
- `create_in_context_tuner(...)`: helper to initialise an `InContextTuner`.

---

## 2. Encoder – `Medical3DUNetEncoder`

**File**: `src/iris/model/encoder.py`  
**Class**: `Medical3DUNetEncoder`

### Key Traits

- **Stem**: `ResidualBlock(in_channels → base_channels)` with stride 1.
- **Downsampling stages**: 4 blocks, each `ResidualBlock(ch[i] → ch[i+1], stride=2)` halving spatial dimensions.
- **Channels**: `[32, 64, 128, 256, 512]` by default (configurable via `base_channels`).
- **Normalisation**: `nn.InstanceNorm3d` with affine parameters (robust across modalities).
- **Activation**: `nn.LeakyReLU(0.01)` (handles CT intensity distributions).

### Output

- `EncoderOutput.features`: deepest tensor of shape `(B, 512, 8, 8, 8)` when input is 128³.
- `EncoderOutput.skip_connections`: tuple of feature maps from each preceding stage (`len == 4` order: stem → stage1 → stage2 → stage3).
- `downsample_ratio = 2**stages = 16`: used by the task encoder to compute `PixelShuffle` factors.

### Extensibility

- Increase `stages` or channels by modifying `base_channels`.
- Add attention or squeeze-excite modules by customising `ResidualBlock`.

---

## 3. Task Encoding – `TaskEncodingModule`

**File**: `src/iris/model/task_encoding.py`  
**Class**: `TaskEncodingModule`

### Inputs

- `support_features F_s`: `(B, C, d, h, w)` (output of encoder; default `C=512`, `d=h=w=8`).
- `support_masks y_s`: `(B, K, D, H, W)` binary masks per class (`K` classes; volumes at full resolution 128³).

### Foreground Encoding (Eq. 2)

1. Upsample features to mask resolution using `upsample_to_reference` (trilinear).
2. Multiply with binary mask, average pool over spatial dimensions.
3. Result `T_f` has shape `(B, K, 1, C)`; appended as the first token per class.

### Contextual Encoding (Eq. 3–4)

1. `self.pre_shuffle`: 1×1×1 conv transforms `C → (C/r^3)*r^3`.
2. `pixel_shuffle_3d`: upscales to original spatial resolution.
3. Concatenate mask along channel axis, apply `self.context_conv (1×1×1)`.
4. `pixel_unshuffle_3d`: revert to downsampled resolution.
5. `self.post_unshuffle`: 1×1×1 conv to restore channel dimension `C`.
6. Flatten to token list and apply `nn.MultiheadAttention` twice:
   - Cross-attention between learnable query tokens (`self.query_tokens`) and spatial tokens.
   - Self-attention among query tokens.

Result `T_c`: `(B, K, m, C)` where `m = num_query_tokens`.

### Final Task Embedding

`task_embeddings = torch.cat([T_f, T_c], dim=2)` → `(B, K, m+1, C)`  
Used as the conditioning signal for all decoders and memory bank entries.

### Key Implementation Notes

- `feature_channels % num_attention_heads`: automatically adjusts head count via `math.gcd` to avoid runtime errors if divisibility requirements change.
- `m` (query tokens) defaults to 8 but can be tuned.
- `self.query_tokens`: registered parameter; same across all classes, expanded per batch/class.
- Masks are handled robustly: dtype converted to float, shape broadcast, random class dropping supported.

---

## 4. Decoder – `MaskDecoder` & `BidirectionalCrossAttention`

**File**: `src/iris/model/decoder.py`

### Components

1. **BidirectionalCrossAttention**:
   - `features_to_tokens`: updates task tokens with query feature context.
   - `tokens_to_features`: updates query features using enriched tokens.
   - `token_self_attn`: optional self-attention after cross interaction (aligns with Eq. 5).
   - Accepts `features_seq` `(B*K, N, C)` and `tokens` `(B*K, m+1, C)`.

2. **UNet-style decoder**:
   - `DecoderStage`: transpose conv upsampling + `ResidualBlock` fusion with skip connections.
   - `FiLMLayer`: modulates features using averaged token embeddings (`summary = tokens.mean(dim=1)`).
   - Iterates through skip connections in reverse order (from deepest to shallowest).

3. **Final Reconstruction**:
   - `final_conv`: 1×1×1 conv producing single-channel logits per class (applied to features).
   - Upsample via trilinear interpolation to `final_target = (128, 128, 128)`.
   - Reshape to `(B, K, D, H, W)`.

### Output

- `DecoderOutput.logits`: segmentation logits for all classes (pre-sigmoid).
- `DecoderOutput.updated_tokens`: tokens after bidirectional attention (potentially stored/analysed).

### Notes

- `num_attention_heads` automatically adjusted to divide `deepest_channels`.
- FiLM conditioning ensures task embeddings influence each decoder stage.
- Supports inference time measurement by isolating cross-attention from unet operations.

---

## 5. Memory Bank – `ClassMemoryBank`

**File**: `src/iris/model/memory.py`

### Purpose

- Store `T_k ∈ ℝ^{(m+1) × C}` for each class `k`.
- Update rule: `T_k ← α T_k + (1 − α) T̂_k` with `α=0.999` (momentum).  
  Implemented in `update()` and `update_episode()`.

### Capabilities

- `update_episode(task_embeddings, class_ids)`: handles batch updates; class IDs align with per-class tokens (skips background).
- `retrieve(class_ids, default=None)`: fetch embeddings for specific classes; used in object-level retrieval.
- `ensemble(embeddings)` static method: averages embeddings from multiple references (context ensemble).
- `summary()`: quick overview of stored classes and tensor shapes (useful for debugging training coverage).

### Integration Points

- `IrisModel.update_memory_bank(...)`: called after each training episode.
- `IrisModel.retrieve_memory_embeddings(...)`: used by evaluation suite strategies.
- `MedicalDemoRunner`: optionally summarises memory contents in demo reports.

---

## 6. In-Context Tuning – `InContextTuner`

**File**: `src/iris/model/tuning.py`

### Concept

- Optimise task embeddings `T` while freezing all network parameters.
- Follow Section 3.3: a lightweight adaptation loop operating on query images only.

### Implementation Steps

1. `task_embeddings = nn.Parameter(initial_embeddings.detach().clone())`.
2. Freeze `model.parameters()` inside context manager `_frozen_parameters`.
3. Use `Adam` optimiser (lr defaults to 1e-3; configurable).
4. Iterate for `steps`:
   - Forward: `model(query_images, task_embeddings)` (reuses the same task embeddings for support & query).
   - Loss: `DiceCrossEntropyLoss` comparing logits vs. target masks (class weights handled upstream if desired).
   - Backprop: update only task embeddings.
5. Optionally update memory bank with tuned embeddings (`update_memory=True` + `class_ids` provided).
6. Return tuned embeddings (detached) for downstream inference.

### Usage

- `IrisModel.create_in_context_tuner(lr, steps)` returns a tuner bound to current model & memory bank.
- `MedicalEvaluationSuite` strategy `in_context_tuning` leverages the tuner to refine embeddings on the fly.

---

## 7. Loss Function – `DiceCrossEntropyLoss`

**File**: `src/iris/model/tuning.py` (top of file)

### Behaviour

- Computes standard Dice loss (1 − Dice) with smooth term `1e-6`.
- Adds binary cross-entropy with optional per-class weights.
- Supports broadcast weights `(B, K)` → reshape to align with logits dimensions.
- Works for both training and in-context tuning to ensure consistent optimisation objective.

---

## 8. Model Instantiation – `IrisModel`

**File**: `src/iris/model/core.py`

### Constructor Parameters

- `in_channels`: input channels (default 1 for grayscale CT/MRI).
- `base_channels`: base feature width (default 32).
- `num_query_tokens`: `m` in Section 3.2.1 (default 8).
- `num_attention_heads`: for both task encoder and decoder (auto-adjusted if not divisible).
- `volume_shape`: expected input resolution (default `(128,128,128)`) used for final upsampling.
- `use_memory_bank / memory_momentum`: toggle and configure EMA storage.

### Methods Recap

- `encode_support(support_images, support_masks)`:
  - Encodes support images.
  - Returns dictionary with `task_embeddings`, `foreground_embeddings`, `context_tokens`.
- `forward(query_images, task_embeddings, skip_connections=None)`:
  - Encodes query images.
  - Runs decoder with provided embeddings.
- `update_memory_bank(task_embeddings, class_ids)`: skip if memory disabled.
- `retrieve_memory_embeddings(class_ids, fallback=None)`: fetch existing embeddings.
- `create_in_context_tuner(lr, steps)`: convenience wrapper around `InContextTuner`.

### Typical Usage Flow

1. `support_out = model.encode_support(x_s, y_s)` → obtain `task_embeddings`.
2. `logits = model(x_q, support_out["task_embeddings"])["logits"]`.
3. Compute loss vs. `y_q`, backprop.
4. Optionally update memory bank (`model.update_memory_bank(...)`) for classes present in support masks.

---

## 9. File Dependency Diagram (simplified)

```
core.py
 ├── encoder.Medical3DUNetEncoder
 ├── task_encoding.TaskEncodingModule
 ├── decoder.MaskDecoder
 ├── memory.ClassMemoryBank (optional)
 └── tuning.InContextTuner factory

task_encoding.py
 ├── utils.pixel_shuffle_3d / pixel_unshuffle_3d / upsample_to_reference
 └── torch.nn.MultiheadAttention (cross & self attention)

decoder.py
 ├── ResidualBlock from encoder.py (for UNet stages)
 ├── BidirectionalCrossAttention (local definition)
 └── FiLMLayer (task-conditioned feature modulation)

memory.py
 └── torch.nn.Parameter-like storage per class with EMA updates

tuning.py
 ├── DiceCrossEntropyLoss
 └── InContextTuner (uses IrisModel forward pass)
```

---

## 10. Extending / Experimenting

- **Changing attention depth**: adjust `num_query_tokens`, add more self-attention layers in `TaskEncodingModule`.
- **Alternative conditioning**: replace FiLM with conditional batch norm or cross-attention at each decoder stage.
- **New inference strategies**: extend `MedicalEvaluationSuite.strategies` to include new keys referencing custom inference functions (e.g., multi-shot fine-tuning).
- **Memory variants**: modify `ClassMemoryBank` to store distributions, confidence scores, or multi-modal embeddings.
- **Multi-modal inputs**: set `in_channels` > 1 and adapt preprocessing pipelines to stack modalities.

---

## 11. References and Validation

The implementation aligns with:

- Section 3.2.1 (task encoding) → pixel shuffle/unshuffle, cross + self attention, `T_f` + `T_c`.
- Section 3.2.2 (mask decoder) → bidirectional cross-attention, FiLM-modulated UNet decoder.
- Section 3.3 (memory bank & in-context tuning) → EMA updates, token-only optimisation.
- Section 4 (evaluation strategies) → direct mapping in `evaluation.py` and `demo.py`.

This documentation should equip you with the necessary insight to modify, extend, or audit the IRIS model architecture and its auxiliary mechanisms confidently.


