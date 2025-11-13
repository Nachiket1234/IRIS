# Running IRIS on Real Medical Datasets

This guide walks through environment setup, dataset preparation, training, evaluation, and clinical demonstrations using the real-world medical datasets referenced in the IRIS paper (AMOS, ACDC, MSD Pancreas, SegTHOR, etc.).

---

## 1. Environment Setup

```powershell
# From the repository root
python -m venv .venv
.venv\Scripts\Activate.ps1         # On macOS/Linux use: source .venv/bin/activate

pip install -r requirements.txt    # Core dependencies
pip install nibabel SimpleITK matplotlib  # Medical I/O + visualisation

# Make the src/ directory discoverable
$env:PYTHONPATH = "$PWD\src"
```

Optional (compile ops faster on GPU):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. Dataset Organisation

1. Download the 12 training datasets and 5 held-out evaluation datasets cited by the paper (e.g. AMOS, BCV, WORD, MSD, SegTHOR, ACDC, IVDM3Seg, etc.).
2. Arrange each dataset in a folder with the standard splits:
   ```
   DATASET_ROOT/
     imagesTr/  (or training/)
     labelsTr/
     imagesTs/  (optional for held-out)
     labelsTs/
   ```
   Check each `MedicalDataset` loader in `src/iris/data/datasets/` for exact expectations.
3. Note the root paths; you will pass them to the loaders.

Example mapping:
```python
DATA_ROOTS = {
    "amos": "D:/datasets/AMOS22",
    "msd_pancreas": "D:/datasets/MSD/Pancreas",
    "acdc": "D:/datasets/ACDC",
    "segthor": "D:/datasets/SegTHOR",
    # add remaining datasets...
}
```

---

## 3. Training with Episodic Pipeline

Create a driver script (e.g. `train_real.py`) similar to the following:

```python
from pathlib import Path

from iris.data import build_dataset, DatasetSplit
from iris.model import IrisModel
from iris.training import EpisodicTrainer, EpisodicTrainingConfig, set_global_seed

DATA_ROOTS = {
    "amos": "D:/datasets/AMOS22",
    "msd_pancreas": "D:/datasets/MSD/Pancreas",
    "segthor": "D:/datasets/SegTHOR",
    "acdc": "D:/datasets/ACDC",
    # extend with remaining datasets...
}

set_global_seed(42)

train_datasets = [
    build_dataset("amos", root=DATA_ROOTS["amos"], split=DatasetSplit.TRAIN),
    build_dataset("msd_pancreas", root=DATA_ROOTS["msd_pancreas"], split=DatasetSplit.TRAIN),
    build_dataset("segthor", root=DATA_ROOTS["segthor"], split=DatasetSplit.TRAIN),
    # add others as needed...
]

model = IrisModel(
    in_channels=1,
    base_channels=32,
    num_query_tokens=8,
    num_attention_heads=8,
    volume_shape=(128, 128, 128),
    use_memory_bank=True,
    memory_momentum=0.999,
)

config = EpisodicTrainingConfig(
    base_learning_rate=2e-3,
    weight_decay=1e-5,
    total_iterations=80_000,
    warmup_iterations=2_000,
    batch_size=32,
    decay_interval=5_000,
    lr_decay_gamma=0.98,
    gradient_clip_norm=1.0,
    checkpoint_dir="checkpoints/real_run",
    log_every=50,
    eval_every=2_000,
    checkpoint_every=5_000,
    volume_size=(128, 128, 128),
    random_class_drop_prob=0.15,
)

trainer = EpisodicTrainer(model, train_datasets, config, device="cuda")
trainer.train()
```

**Tips**
- Ensure GPU memory is sufficient for batch size 32 (adjust if necessary).
- Attach an evaluation hook in `trainer.train()` to call the evaluation suite periodically.
- Checkpoints and optimizer state are saved under `checkpoints/real_run/`.

---

## 4. Evaluation & Baseline Comparison

Once training finishes (or during checkpoints), run the evaluation suite:

```python
from iris.training import EvaluationConfig, MedicalEvaluationSuite

eval_config = EvaluationConfig(
    in_distribution=[
        build_dataset("amos", root=DATA_ROOTS["amos"], split=DatasetSplit.VALID),
        build_dataset("msd_pancreas", root=DATA_ROOTS["msd_pancreas"], split=DatasetSplit.VALID),
    ],
    out_of_distribution=[
        build_dataset("acdc", root=DATA_ROOTS["acdc"], split=DatasetSplit.TEST),
        build_dataset("segthor", root=DATA_ROOTS["segthor"], split=DatasetSplit.TEST),
        # add IVDM3Seg, Pelvic, etc.
    ],
    novel_classes=[
        build_dataset("msd_pancreas", root=DATA_ROOTS["msd_pancreas"], split=DatasetSplit.TEST),
        # add other unseen anatomy datasets
    ],
    num_episodes=32,
    repetitions=5,
    ensemble_size=4,
    tuner_steps=25,
    tuner_lr=5e-4,
    random_seed=123,
    baseline_scores={
        "AMOS": {"nnUNet": 0.880, "SAM-adapted": 0.842},
        "ACDC": {"nnUNet": 0.905},
        # populate with literature numbers for reference
    },
)

evaluator = MedicalEvaluationSuite(model, eval_config)
results = evaluator.evaluate()
print(results)
```

`results` includes for each dataset and strategy:
- Mean/stdev Dice
- Hausdorff mean/stdev (95th percentile)
- Inference time / memory usage
- Per-class metrics

Persist the dictionary as JSON for further analysis.

---

## 5. Clinical Demo & Visualisation

Generate comparative overlays, dashboards, and qualitative notes:

```python
from iris.training import ClinicalDemoConfig, MedicalDemoRunner

demo_config = ClinicalDemoConfig(
    num_examples=5,
    strategies=("one_shot", "context_ensemble", "object_retrieval", "in_context_tuning"),
    output_dir="demo_outputs/real_demo",
    save_visualizations=True,
    save_reports=True,
    include_memory_bank_summary=True,
)

demo_runner = MedicalDemoRunner(model, evaluator, demo_config)
report = demo_runner.run_demo(eval_config.in_distribution + eval_config.out_of_distribution)

print("Demo report:", report["cases"][:3])  # preview first few entries
print("Dashboard:", report["dashboard"])
```

Outputs (`demo_outputs/real_demo/`):
- Multi-planar overlays (PNG) combining query image, prediction, ground truth, reference mask.
- `demo_report.json` summarising Dice, Hausdorff, latency per strategy and per case.
- Optional bar-chart dashboard of mean Dice per dataset & strategy.

---

## 6. Best Practices & Considerations

- **Hardware**: training assumes at least one high-memory GPU; adjust `batch_size` and `volume_size` if memory constrained.
- **Data privacy**: ensure de-identification and compliance before copying datasets to the machine.
- **Checkpointing**: keep multiple checkpoints (`checkpoint_every`) to analyse convergence or resume training.
- **Hausdorff computation**: requires `scipy` and can be slow—consider subsampling during prototyping.
- **Memory bank**: persisted automatically in checkpoints; verify the number of stored classes via `model.memory_bank.summary()`.
- **Reproducibility**: stick to fixed seeds, document dataset versions, and log baseline comparisons.

---

## 7. Quick CLI Summary

```powershell
# Activate environment & expose modules
cd "C:\Users\nachi\Downloads\IRIS\IRIS V2"
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"

# Train
python train_real.py

# Evaluate
python evaluate_real.py

# Demo (produces visual artefacts + JSON)
python demo_real.py
```

Replace the script names with the driver files you create based on the templates above.

This runbook should get you from raw datasets to trained checkpoints, quantitative evaluation, and qualitative clinical demonstrations using the full IRIS pipeline.


