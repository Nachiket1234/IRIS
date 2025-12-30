# IRIS V2 – Real-World Medical Segmentation Framework

IRIS V2 is a full-stack implementation of the **Imaging Retrieval via In-context Segmentation (IRIS)** architecture, adapted for real clinical datasets. The project reproduces the paper’s Section 3.2–3.3 pipeline – including the medical-optimised encoder, task encoding module, bidirectional decoder, memory bank, and in-context tuning – and pairs it with a complete episodic training, evaluation, and demo stack for 3D medical segmentation.

---

## Key Capabilities

- 🚑 **Medical 3D UNet Encoder** tuned for 128³ CT/MRI/PET volumes with residual downsampling.
- 🧠 **Task Encoding Module** (foreground pooling + contextual tokens via pixel shuffle/unshuffle + cross/self attention).
- 🎯 **Mask Decoder** with bidirectional cross-attention + FiLM-modulated UNet stages for simultaneous multi-class predictions.
- 🧾 **Class-Specific Memory Bank** using EMA updates for context ensemble and object-level retrieval.
- 🔄 **In-Context Tuning** that updates only task embeddings while freezing the model for rapid adaptation.
- 🧪 **Episodic Training Loop** with Lamb optimiser, medical augmentations, noise injection, and class dropping.
- 📊 **Evaluation Suite & Clinical Demo** covering in-distribution, OOD, and novel-class scenarios across four inference strategies (one-shot, context ensemble, object retrieval, in-context tuning).
- 🖼️ **Visualisation Toolkit** for multi-planar overlays, dashboards, and training curves.

---
## Visuals
<!-- Two-column row with equal-width images -->
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/14a5c470-5d8c-48d2-b3fe-16521a69931b" width="620" alt="IRIS Home" />
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/095480d2-6df7-41e8-b8c7-693594bfcf9f" width="420" alt="IRIS Live Demo" />
    </td>
  </tr>
  <tr>
    <td align="center"><strong>IRIS Home</strong> — overview, performance metrics, architecture map, and links to demo and docs.</td>
    <td align="center"><strong>IRIS Live Demo</strong> — upload support/query images, choose evidence strategy, run segmentation, and view metrics.</td>
  </tr>
</table>

<!-- Full-width third image below -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/d0e20e70-f001-4553-8bb8-d2747b40b768" width="880" alt="IRIS In-Context Learning Brain Tumor" />
</p>

<p align="center"><strong>IRIS In-Context Learning</strong> — support / query / prediction / ground-truth overlays and overlap visualization for brain MRI cases.</p>





## Repository Structure

```
docs/
  model_architecture.md      # Deep dive into encoder/task encoder/decoder/memory/tuner
  workflow_overview.md       # Repository organisation and end-to-end pipeline
  run_real_datasets.md       # Runbook for training/evaluation/demo on real datasets

src/iris/
  data/                      # Dataset loaders, I/O, preprocessing, augmentations, episodic samplers
  model/                     # Core IRIS components (encoder, task encoding, decoder, memory, tuning)
  training/                  # Episodic trainer, evaluation suite, demo runner, visualisation

demo_outputs/                # Generated artefacts (logs, overlays, dashboards)
tests/                       # Synthetic unit tests covering critical modules
```

For a detailed explanation of each file and the pipeline, read the documentation in `docs/`.

---

## Getting Started

### 1. Environment Setup

```powershell
cd "C:\Users\nachi\Downloads\IRIS\IRIS V2"
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install nibabel SimpleITK matplotlib
$env:PYTHONPATH = "$PWD\src"
```

### 2. Quick Synthetic Demo (sanity check)

```powershell
python docs/examples/run_demo.py   # or use the provided demo script (see docs)
type demo_outputs\synthetic_demo\results.txt
```

### 3. Train on Real Datasets

Use `docs/run_real_datasets.md` as a runbook. In summary:

```python
from iris.data import build_dataset, DatasetSplit
from iris.model import IrisModel
from iris.training import EpisodicTrainer, EpisodicTrainingConfig

DATA_ROOTS = {"amos": "D:/datasets/AMOS22", "msd_pancreas": "...", ...}

train_sets = [
    build_dataset("amos", root=DATA_ROOTS["amos"], split=DatasetSplit.TRAIN),
    build_dataset("msd_pancreas", root=DATA_ROOTS["msd_pancreas"], split=DatasetSplit.TRAIN),
    # add remaining datasets
]

model = IrisModel(use_memory_bank=True)
config = EpisodicTrainingConfig(total_iterations=80_000, batch_size=32, ...)

trainer = EpisodicTrainer(model, train_sets, config, device="cuda")
trainer.train()
```

### 4. Evaluate & Produce Clinical Demos

```python
from iris.training import EvaluationConfig, MedicalEvaluationSuite, ClinicalDemoConfig, MedicalDemoRunner

eval_cfg = EvaluationConfig(
    in_distribution=[...],
    out_of_distribution=[...],
    novel_classes=[...],
    strategies=("one_shot", "context_ensemble", "object_retrieval", "in_context_tuning"),
    baseline_scores={"AMOS": {"nnUNet": 0.880}},
)
evaluator = MedicalEvaluationSuite(model, eval_cfg)
results = evaluator.evaluate()

demo_cfg = ClinicalDemoConfig(output_dir="demo_outputs/real_demo", save_visualizations=True)
demo_runner = MedicalDemoRunner(model, evaluator, demo_cfg)
demo_report = demo_runner.run_demo(eval_cfg.in_distribution)
```

Outputs include JSON summaries, overlays, and dashboards for clinical review.

---

## Documentation & References

- **Architecture Guide**: `docs/model_architecture.md`
- **Workflow Overview**: `docs/workflow_overview.md`
- **Real Dataset Runbook**: `docs/run_real_datasets.md`
- **Paper**: IRIS research paper (see `/IRIS- Research paper.pdf`)

---

## Testing

Run the synthetic unit test suite:

```powershell
pytest
```

This exercises dataset loaders, model components, training/evaluation utilities, and ensures regressions are caught early.

---

## Contributing

Pull requests are welcome! If you extend the architecture (new datasets, inference strategies, or clinical integrations), please add or update unit tests and documentation accordingly.

---

## License

Specify your licence here (e.g., MIT, Apache 2.0) once finalised.


