# IRIS V2 Project Overview & Workflow

This document explains how the repository is organised and how data flows through the full IRIS medical segmentation pipeline: dataset loading, model components, episodic training, evaluation, and clinical demonstrations.

---

## 1. Repository Structure

- `src/iris/data/`
  - `base.py`: shared abstractions (`MedicalDataset`, `VolumeRecord`, split utilities).
  - `io.py`: robust 3D medical I/O (NIfTI, MHD, etc.) via nibabel / SimpleITK.
  - `preprocessing.py`: resampling, resizing to 128³, modality-specific normalisation, mask handling.
  - `augmentations.py`: 3D medical augmentations (crop, flip, affine, intensity, random class drop).
  - `datasets/`: AMOS, ACDC, MSD Pancreas, SegTHOR loaders (easily extendable to all 12+5 datasets).
  - `samplers.py`: episodic sampling helpers for support/query episodes.

- `src/iris/model/`
  - `encoder.py`: four-stage residual 3D UNet encoder (128³ volumes → deep feature map + skips).
  - `task_encoding.py`: Section 3.2.1 task encoder (foreground pooling, pixel shuffle/unshuffle context, cross/self attention over query tokens).
  - `decoder.py`: Section 3.2.2 mask decoder with bidirectional cross-attention + FiLM-modulated UNet upsampling.
  - `memory.py`: class-specific EMA memory bank for context ensemble / retrieval.
  - `tuning.py`: in-context tuning loop (optimises only task embeddings).
  - `core.py`: ties encoder, task encoder, decoder, memory bank, tuner factory into `IrisModel`.

- `src/iris/training/`
  - `pipeline.py`: episodic training loop (Section 3.2.3) with Lamb optimiser, warm-up + decay schedule, augmentation, noise injection, memory updates.
  - `evaluation.py`: medical evaluation suite (Section 4) for in-distribution, OOD, novel-class splits; computes Dice, Hausdorff, latency, memory, strategy comparisons.
  - `demo.py`: clinical demonstration runner that orchestrates case studies, overlays, dashboards, and qualitative notes.
  - `visualization.py`: axial/coronal/sagittal rendering, dashboards, training curves.
  - `lamb.py`: Adam-style Lamb optimiser implementation.
  - `utils.py`: reproducibility helpers (set seeds, class weighting, dataset descriptions).

- `tests/`: synthetic unit tests covering dataset loaders, augmentations, encoder/decoder, memory bank, episodic sampler, training/evaluation plumbing.

- `demo_outputs/`: generated artefacts (console log, overlays, dashboards) from demo runs.

---

## 2. Data Flow & Pipeline

1. **Dataset Discovery (Section 3.1)**
   - `MedicalDataset` subclasses search dataset roots, produce `VolumeRecord`s with modality/anatomy metadata.
   - NIfTI volumes are loaded via `io.py`, resampled to target spacing, resized to 128×128×128, normalised (HU clipping for CT, percentile scaling for MRI).

2. **Episode Sampling**
   - `EpisodicTrainer` selects a dataset → draws support (`x_s`, `y_s`) and query (`x_q`, `y_q`) volumes → applies medical augmentations + query noise.
   - `random_class_drop_prob` simulates missing structures; per-class masks are stacked for multi-organ segmentation.

3. **Task Encoding (Section 3.2.1)**
   - Support image is encoded with the 3D UNet.
   - Foreground embedding: upsampled support features ⊙ support mask → global pool → `T_f`.
   - Context embedding: pixel shuffle, concat mask, 1×1×1 conv, pixel unshuffle → flattened tokens → cross/self attention with learnable queries → `T_c`.
   - Final task embedding: `T = [T_f ; T_c]` per target class.

4. **Query Decoding (Section 3.2.2)**
   - Query image passes through the encoder.
  - Bidirectional cross-attention exchanges information between query tokens and task embeddings.
  - FiLM-modulated UNet decoder reconstructs segmentation logits at 128³ resolution in a single pass for all classes.

5. **Loss & Optimisation**
   - `DiceCrossEntropyLoss` (Dice + BCE) on query masks; class weights mitigate imbalance.
   - Training uses Lamb optimiser with configurable warm-up, exponential decay, gradient clipping (80k iteration schedule in full runs).
   - Memory bank updates per class with EMA momentum (α = 0.999) for context ensemble / retrieval.

6. **Evaluation & Demo (Section 4)**
   - `MedicalEvaluationSuite` runs all four inference strategies (one-shot, context ensemble, object retrieval, in-context tuning) across in-distribution, OOD, and novel-class datasets.
   - Metrics: Dice mean/std, Hausdorff percentiles, inference time, GPU/CPU memory.
   - Results can include baseline comparisons (nnUNet, SAM variants) by providing reference scores.
   - `MedicalDemoRunner` drives case studies, overlays (when matplotlib available), dashboards, and qualitative deployment notes.

---

## 3. Typical Workflow

1. **Prepare datasets** (download, organise, point loaders at root directories).
2. **Instantiate `IrisModel`** (set `use_memory_bank=True` for full behaviour).
3. **Configure `EpisodicTrainingConfig`** (batch size, iterations, augmentation hyper-parameters, learning rate schedule).
4. **Create `EpisodicTrainer`** with a list of training datasets → call `train()` (optionally attach evaluation hook for periodic validation).
5. **Run `MedicalEvaluationSuite`** with real validation/test splits across in-distribution, domain-shift, and novel-class datasets; include baseline metrics for comparison.
6. **Generate clinical demonstrations** via `MedicalDemoRunner`, enabling visual outputs and JSON summaries for clinical review.
7. **Inspect artefacts** in `demo_outputs/` (images, dashboards, reports) and logs/checkpoints in the configured directories.

---

## 4. Extending the System

- **Adding a new dataset**: implement a `MedicalDataset` subclass, register it in `src/iris/data/datasets/__init__.py`, ensure metadata (modality, anatomy) are provided.
- **Custom augmentations**: extend `MedicalAugmentation` or plug new transforms into the episodic trainer.
- **Alternative evaluation strategies**: add new keys to `MedicalEvaluationSuite.strategies` with custom inference logic (e.g., multi-reference fine-tuning).
- **Clinical integrations**: hook demo outputs into dashboards or PACS by consuming the generated JSON and image overlays.

This document should help you navigate the codebase and understand how each module contributes to IRIS’s end-to-end medical segmentation workflow.


