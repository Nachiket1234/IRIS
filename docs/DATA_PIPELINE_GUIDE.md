# IRIS Data Pipeline Guide

**Complete guide to data handling and processing in IRIS medical segmentation**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Registry System](#2-dataset-registry-system)
3. [Supported Datasets](#3-supported-datasets)
4. [Data Loading Architecture](#4-data-loading-architecture)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Data Augmentation](#6-data-augmentation)
7. [Adding Custom Datasets](#7-adding-custom-datasets)
8. [Data Statistics & Analysis](#8-data-statistics--analysis)
9. [File Formats & I/O](#9-file-formats--io)
10. [Performance Optimization](#10-performance-optimization)
11. [Troubleshooting](#11-troubleshooting)
12. [Data Directory Structure](#12-data-directory-structure)
13. [API Reference](#13-api-reference)
14. [Code Examples](#14-code-examples)
15. [Visual Assets](#15-visual-assets)
16. [Conclusion](#16-conclusion)

---

## 1. Introduction

### Why Medical Image Data is Unique

Medical imaging data differs fundamentally from natural images in several critical ways:

| Aspect | Natural Images | Medical Images |
|--------|---------------|----------------|
| **Dimensionality** | 2D (RGB) | 2D or 3D (grayscale/multi-channel) |
| **File Formats** | JPEG, PNG | NIfTI, DICOM, TIFF |
| **Value Range** | [0, 255] RGB | Variable (Hounsfield units, intensity) |
| **Modalities** | Single (camera) | CT, MRI, X-ray, Ultrasound, etc. |
| **Resolution** | Fixed | Highly variable (512×512 to 1024³) |
| **Annotations** | Bounding boxes | Pixel/voxel-wise segmentation masks |
| **Class Imbalance** | Moderate | Extreme (target organs <5% of volume) |
| **Clinical Context** | None | Critical for diagnosis |

### Few-Shot Learning Requirements

IRIS implements **few-shot segmentation**, which requires a fundamentally different data loading approach than traditional supervised learning:

**Traditional Segmentation**:
```
Training: 1,000 images → Train model on all images → Model
Inference: New image → Model → Prediction
```

**Few-Shot Segmentation (IRIS)**:
```
Training: Multiple datasets → Learn to adapt from examples
Inference: New image + 3 support examples → Model → Prediction
```

This paradigm requires **episodic training** where each training iteration consists of:
- **Support Set**: K examples showing "what to segment" (K=1-5 typically)
- **Query Set**: Images to segment using knowledge from support set

### Episodic Training Paradigm

```mermaid
graph TB
    subgraph Episode["Training Episode"]
        DATASET["Select Dataset<br/>(e.g., Chest X-Ray)"]
        
        SUPPORT["Support Set<br/>K=3 images with masks<br/>'Here's what lungs look like'"]
        QUERY["Query Set<br/>B=4 images with masks<br/>'Segment these lungs'"]
        
        ENCODE["Encode Support<br/>Extract task embedding<br/>'Learn what to segment'"]
        SEGMENT["Segment Query<br/>Using task knowledge<br/>'Apply to new images'"]
        
        LOSS["Compute Loss<br/>Dice + BCE<br/>Compare prediction vs. ground truth"]
        UPDATE["Update Model<br/>Backpropagation<br/>Learn to segment better"]
    end
    
    DATASET --> SUPPORT
    DATASET --> QUERY
    
    SUPPORT --> ENCODE
    QUERY --> SEGMENT
    ENCODE --> SEGMENT
    
    SEGMENT --> LOSS
    LOSS --> UPDATE
    
    UPDATE -.Next Episode.-> DATASET
    
    style Episode fill:#fff3e0,stroke:#f57c00
    style SUPPORT fill:#e3f2fd,stroke:#1976d2
    style QUERY fill:#e8f5e9,stroke:#388e3c
```

### Support Set vs. Query Set

**Support Set** (K examples):
- **Purpose**: Define the segmentation task
- **Content**: Images WITH ground truth masks
- **Role**: "Here are examples of what you should segment"
- **Typical Size**: K=1 (one-shot), K=3 (few-shot), K=5 (many-shot)
- **Usage**: Encoded into task embedding that guides segmentation

**Query Set** (B examples):
- **Purpose**: Test images to segment
- **Content**: Images WITH ground truth masks (during training)
- **Role**: "Now segment these images using task knowledge"
- **Typical Size**: B=4-8 images per episode
- **Usage**: Segmented by model, used to compute loss

**Key Insight**: During inference, only support set is needed (no ground truth required for query images).

### Data Flow Overview

```mermaid
graph LR
    subgraph Storage["1. Raw Data Storage"]
        RAW["Raw Medical Images<br/>NIfTI, DICOM, PNG, JPG"]
        ANNO["Annotations<br/>Segmentation Masks"]
    end
    
    subgraph Loading["2. Data Loading"]
        DISC["Dataset Discovery<br/>Find image-mask pairs"]
        SPLIT["Train/Val/Test Split<br/>75% / 5% / 20%"]
    end
    
    subgraph Preproc["3. Preprocessing"]
        LOAD["Load Volume<br/>nibabel, PIL, SimpleITK"]
        NORM["Normalize Intensity<br/>Clip, standardize"]
        RESIZE["Resize to 128³<br/>Interpolation"]
        CONVERT["Convert to Tensor<br/>PyTorch format"]
    end
    
    subgraph Episodic["4. Episodic Sampling"]
        SAMPLE["Sample Episode<br/>Select dataset"]
        SUPPORT_S["Sample K Support<br/>With masks"]
        QUERY_S["Sample B Query<br/>With masks"]
        BATCH["Create Batch<br/>Dict[str, Tensor]"]
    end
    
    subgraph Model["5. Model Input"]
        FORWARD["Forward Pass<br/>IRIS Model"]
    end
    
    RAW --> DISC
    ANNO --> DISC
    DISC --> SPLIT
    
    SPLIT --> LOAD
    LOAD --> NORM
    NORM --> RESIZE
    RESIZE --> CONVERT
    
    CONVERT --> SAMPLE
    SAMPLE --> SUPPORT_S
    SAMPLE --> QUERY_S
    SUPPORT_S --> BATCH
    QUERY_S --> BATCH
    
    BATCH --> FORWARD
    
    style Storage fill:#ffebee,stroke:#c62828
    style Loading fill:#fff3e0,stroke:#f57c00
    style Preproc fill:#e8f5e9,stroke:#388e3c
    style Episodic fill:#e3f2fd,stroke:#1976d2
    style Model fill:#f3e5f5,stroke:#7b1fa2
```

### Episode Structure

Each episode returns a dictionary with the following structure:

```python
episode = {
    # Support set (K examples, typically K=3)
    'support_images': torch.Tensor,  # Shape: (K, 1, D, H, W)
    'support_masks': torch.Tensor,   # Shape: (K, 1, D, H, W)
    
    # Query set (B examples, typically B=4)
    'query_images': torch.Tensor,    # Shape: (B, 1, D, H, W)
    'query_masks': torch.Tensor,     # Shape: (B, 1, D, H, W)
    
    # Metadata
    'class_ids': List[int],          # [1, 1, 1] for single-class tasks
    'dataset_name': str,             # e.g., "chest_xray_masks"
    'modality': str,                 # e.g., "X-ray", "CT", "MRI"
    'anatomy': str,                  # e.g., "chest", "brain", "skin"
}
```

**Dimension Explanation**:
- **K**: Number of support examples (1-5)
- **B**: Batch size / number of query examples (4-8)
- **1**: Single channel (grayscale medical images)
- **D, H, W**: Depth, Height, Width (typically 128×128×128 for 3D, or 16×256×256 for 2D as pseudo-3D)

---

## 2. Dataset Registry System

### Overview

IRIS uses a **centralized dataset registry** pattern that provides:

1. **Uniform Interface**: All datasets accessible through single API
2. **Automatic Discovery**: Datasets auto-register on import
3. **Extensibility**: Easy to add new datasets
4. **Type Safety**: Consistent return types across datasets
5. **Factory Pattern**: Build datasets by name string

### Architecture

```mermaid
graph TB
    subgraph Registry["Dataset Registry (Singleton)"]
        DICT["DATASET_REGISTRY<br/>Dict[str → DatasetClass]"]
    end
    
    subgraph Datasets["Dataset Implementations"]
        CHEST["@register_dataset('chest_xray_masks')<br/>ChestXrayMasksDataset"]
        BRAIN["@register_dataset('brain_tumor')<br/>BrainTumorDataset"]
        ISIC["@register_dataset('isic')<br/>ISICDataset"]
        DRIVE["@register_dataset('drive')<br/>DRIVEDataset"]
        KVASIR["@register_dataset('kvasir')<br/>KvasirDataset"]
        MORE["... + 5 more datasets"]
    end
    
    subgraph Factory["Factory Functions"]
        BUILD["build_dataset(name, root, split)<br/>Returns: MedicalDataset instance"]
        LIST["list_datasets()<br/>Returns: List[str] of available datasets"]
    end
    
    subgraph Usage["User Code"]
        USER["dataset = build_dataset(<br/>    'chest_xray_masks',<br/>    root='data/',<br/>    split='train'<br/>)"]
    end
    
    CHEST -.Decorator registers.-> DICT
    BRAIN -.Decorator registers.-> DICT
    ISIC -.Decorator registers.-> DICT
    DRIVE -.Decorator registers.-> DICT
    KVASIR -.Decorator registers.-> DICT
    MORE -.Decorator registers.-> DICT
    
    BUILD --> DICT
    DICT --> BUILD
    
    USER --> BUILD
    BUILD -.Returns dataset instance.-> USER
    
    style Registry fill:#fff3e0,stroke:#f57c00
    style Datasets fill:#e3f2fd,stroke:#1976d2
    style Factory fill:#e8f5e9,stroke:#388e3c
```

### Core Components

#### 2.1 DatasetRegistry Class

**File**: `src/iris/data/factory.py`

```python
class DatasetRegistry(dict):
    """
    Simple registry mapping dataset identifiers to dataset classes.
    """

    def register(self, name: str, dataset_cls: Type[MedicalDataset]) -> None:
        """Register a dataset class with a name."""
        key = name.lower()
        if key in self:
            raise KeyError(f"Dataset '{name}' is already registered.")
        self[key] = dataset_cls

    def build(
        self,
        name: str,
        root: str,
        split: DatasetSplit,
        **kwargs: Any,
    ) -> MedicalDataset:
        """Build a dataset instance by name."""
        key = name.lower()
        if key not in self:
            available = ", ".join(self.keys()) or "<empty>"
            raise KeyError(f"Dataset '{name}' not registered. Available: {available}")
        
        dataset_cls = self[key]
        return dataset_cls(root=root, split=split, **kwargs)


# Global singleton instance
DATASET_REGISTRY = DatasetRegistry()
```

#### 2.2 Registration Decorator

**Pattern**: Decorator-based auto-registration

```python
def register_dataset(name: str):
    """
    Decorator to register dataset classes in the global registry.
    
    Usage:
        @register_dataset('my_dataset')
        class MyDataset(MedicalDataset):
            ...
    """
    def decorator(cls: Type[MedicalDataset]) -> Type[MedicalDataset]:
        DATASET_REGISTRY.register(name, cls)
        return cls
    
    return decorator
```

**Why This Pattern?**
- **Automatic**: No manual registration needed
- **Declarative**: Clear which name maps to which class
- **Side-effect on import**: Datasets register when module imported
- **Clean**: No central list to maintain

#### 2.3 Factory Function

**Convenience builder** with flexible split argument:

```python
def build_dataset(
    name: str,
    root: str,
    split: str | DatasetSplit,
    **kwargs: Any,
) -> MedicalDataset:
    """
    Build dataset instance by name.
    
    Args:
        name: Dataset identifier (e.g., 'chest_xray_masks')
        root: Path to dataset root directory
        split: 'train', 'val', or 'test' (string or enum)
        **kwargs: Additional dataset-specific arguments
    
    Returns:
        Dataset instance ready for use with DataLoader
    
    Example:
        >>> dataset = build_dataset(
        ...     'chest_xray_masks',
        ...     root='data/chest_xray',
        ...     split='train',
        ...     depth_slices=16
        ... )
    """
    split_enum = (
        split if isinstance(split, DatasetSplit) 
        else DatasetSplit(split.lower())
    )
    return DATASET_REGISTRY.build(name=name, root=root, split=split_enum, **kwargs)
```

### Usage Examples

#### Example 1: List Available Datasets

```python
from iris.data import DATASET_REGISTRY

# List all registered datasets
available = list(DATASET_REGISTRY.keys())
print(f"Available datasets: {available}")

# Output:
# Available datasets: ['chest_xray_masks', 'brain_tumor', 'drive', 
#                      'isic', 'kvasir', 'amos', 'covid_ct', 
#                      'segthor', 'msd_pancreas', 'acdc']
```

#### Example 2: Build a Dataset

```python
from iris.data import build_dataset

# Build chest X-ray training set
train_dataset = build_dataset(
    name='chest_xray_masks',
    root='data/chest_xray',
    split='train',
    depth_slices=16,
    target_resolution=256
)

print(f"Dataset: {train_dataset.dataset_name}")
print(f"Modality: {train_dataset.modality}")
print(f"Samples: {len(train_dataset)}")

# Output:
# Dataset: chest_xray_masks
# Modality: X-ray
# Samples: 662
```

#### Example 3: Register Custom Dataset

```python
from iris.data import MedicalDataset, register_dataset, DatasetSplit
from pathlib import Path

@register_dataset('my_custom_dataset')
class MyCustomDataset(MedicalDataset):
    """My custom medical imaging dataset."""
    
    dataset_name = "my_custom_dataset"
    modality = "CT"
    anatomy = "abdomen"
    
    def discover_records(self):
        """Find all image-mask pairs in dataset."""
        records = []
        # Implementation here...
        return records

# Now usable via registry
dataset = build_dataset('my_custom_dataset', root='data/my_data', split='train')
```

### Benefits

| Benefit | Description | Example |
|---------|-------------|---------|
| **Decoupling** | User code doesn't import dataset classes directly | `build_dataset('chest_xray')` instead of `from ... import ChestXrayDataset` |
| **Discovery** | Easy to find available datasets | `list(DATASET_REGISTRY.keys())` |
| **Consistency** | All datasets follow same interface | Same API for all: `dataset[0]` returns episode dict |
| **Configuration** | Dataset selection via config files | `config.yaml: dataset: chest_xray_masks` |
| **Testing** | Mock datasets for unit tests | Register test dataset without modifying imports |
| **Plugins** | Third-party datasets can register themselves | Users can add datasets without modifying IRIS code |

---

## 3. Supported Datasets

### Overview

IRIS supports **10 medical imaging datasets** across 6 modalities and 8 anatomical regions. All datasets are unified under a common interface while preserving modality-specific characteristics.

### Master Comparison Table

| Dataset | Modality | Anatomy | Classes | Images | Split | Resolution | Status | File |
|---------|----------|---------|---------|--------|-------|------------|--------|------|
| **Chest X-Ray** | X-ray | Chest/Lung | 2 | 662 | Auto | Variable → 256² | ✅ **95.81% Dice** | `chest_xray_masks.py` |
| **ISIC** | Dermoscopy | Skin | 10 | 2,853 | Train/Test | Variable → 256² | ✅ **87.42% Dice** | `isic.py` |
| **Kvasir-SEG** | Endoscopy | Colon | 2 | 1,000 | Auto | Variable → 256² | ✅ **66.76% Dice** | `kvasir.py` |
| **Brain Tumor** | MRI | Brain | 2 | Variable | Auto | 512² → 256² | ✅ **42.53% Dice** | `brain_tumor.py` |
| **DRIVE** | Fundus | Retina | 2 | 40 | Train/Test | 565×584 → 256² | ✅ **21.82% Dice** | `drive.py` |
| **AMOS** | CT | Abdomen | 16 | 500 | Train/Val/Test | Variable | 🔧 Available | `amos.py` |
| **COVID-19 CT** | CT | Chest | 2 | Variable | Auto | Variable | 🔧 Available | `covid_ct.py` |
| **SegTHOR** | CT | Thorax | 5 | 40 | Train/Test | Variable | 🔧 Available | `segthor.py` |
| **MSD Pancreas** | CT | Pancreas | 3 | 281 | Auto | Variable | 🔧 Available | `msd_pancreas.py` |
| **ACDC** | MRI | Heart | 4 | 150 | Train/Test | Variable | 🔧 Available | `acdc.py` |

**Legend**:
- ✅ **Trained & Validated**: Model trained with published results
- 🔧 **Available**: Dataset loader implemented, not yet trained
- **Classes**: Foreground classes (background not counted)
- **Auto**: Automatic 75%/5%/20% train/val/test split
- **Resolution**: Original → Target (after preprocessing)

---

### 3.1 Chest X-Ray Masks ⭐ BEST PERFORMANCE

**Dataset**: Montgomery County & Shenzhen Hospital Chest X-rays  
**Performance**: 🏆 **95.81% Dice Score** (Best performing dataset)

#### Overview
- **Source**: NIH & Tuberculosis Research
- **Modality**: 2D Chest X-ray (grayscale)
- **Anatomy**: Lungs
- **Classes**: Background (0), Lung mask (1)
- **Medical Use Case**: Lung segmentation for TB screening, pneumonia detection

#### Dataset Characteristics
```python
Dataset Statistics:
- Total Images: 662
- Split: Automatic (75% train / 5% val / 20% test)
- Original Resolution: Variable (mostly 4020×4892 or 3000×3000)
- Target Resolution: 16×256×256 (2D → pseudo-3D)
- File Format: PNG images and masks
- Image Depth: 8-bit grayscale
- Mask Type: Binary (0=background, 255=lung)
```

#### Clinical Challenges
- **Low Contrast**: Subtle lung boundaries
- **Artifacts**: Medical devices, patient positioning
- **Variability**: Different scanners, exposure settings
- **Anatomical Variation**: Lung shape varies significantly

#### File Structure
```
data/chest_xray_masks/
├── images/
│   ├── MCUCXR_0001_0.png
│   ├── MCUCXR_0002_0.png
│   └── ...
└── masks/
    ├── MCUCXR_0001_0.png
    ├── MCUCXR_0002_0.png
    └── ...
```

#### Training Results
| Metric | Value |
|--------|-------|
| **Dice Score** | 95.81% |
| **IoU** | 92.0% |
| **Iterations** | 2,000 |
| **Training Time** | 211 minutes |
| **Convergence** | ~1,500 iterations |

#### Code Example
```python
from iris.data import build_dataset

dataset = build_dataset(
    name='chest_xray_masks',
    root='data/chest_xray_masks',
    split='train',
    depth_slices=16,          # Convert 2D → pseudo-3D
    target_resolution=256,
    cache_data=False
)

print(f"Samples: {len(dataset)}")
print(f"Modality: {dataset.modality}")  # "X-ray"
print(f"Anatomy: {dataset.anatomy}")    # "chest"

# Get a sample
sample = dataset[0]
print(sample['image'].shape)  # torch.Size([1, 16, 256, 256])
print(sample['mask'].shape)   # torch.Size([1, 16, 256, 256])
```

#### Citation
```
@misc{jaeger2014two,
  title={Two public chest X-ray datasets for computer-aided screening of pulmonary diseases},
  author={Jaeger, Stefan and Candemir, Sema and Antani, Sameer and others},
  year={2014}
}
```

---

### 3.2 ISIC (Skin Lesion Segmentation)

**Dataset**: International Skin Imaging Collaboration  
**Performance**: ✅ **87.42% Dice Score**

#### Overview
- **Source**: ISIC Archive (melanoma detection initiative)
- **Modality**: Dermoscopic images (RGB → grayscale)
- **Anatomy**: Skin lesions
- **Classes**: 9 lesion types (converted to binary segmentation)
- **Medical Use Case**: Melanoma and skin cancer screening

#### Dataset Characteristics
```python
Dataset Statistics:
- Total Images: 2,853 (2,594 train + 259 test)
- Split: Pre-defined train/test
- Original Resolution: Variable (mostly 1024×768 to 6000×4000)
- Target Resolution: 16×256×256
- File Format: JPG (images), PNG (masks)
- Classes: 10 total (actinic keratosis, basal cell carcinoma, 
           dermatofibroma, melanoma, nevus, pigmented benign keratosis,
           seborrheic keratosis, squamous cell carcinoma, vascular lesion)
```

#### Clinical Challenges
- **Hair Artifacts**: Dark hairs obscure lesion boundaries
- **Reflections**: Shiny skin surfaces create highlights
- **Varied Backgrounds**: Rulers, ink markings, varied lighting
- **Irregular Shapes**: Lesions have complex, asymmetric boundaries
- **Color Variation**: From pink to dark brown/black

#### File Structure
```
data/isic/
├── Skin cancer ISIC The International Skin Imaging Collaboration/
│   ├── Train/
│   │   ├── actinic keratosis/
│   │   │   ├── ISIC_0000001.jpg
│   │   │   └── ...
│   │   ├── melanoma/
│   │   └── ...
│   └── Test/
│       └── ...
└── masks/
    ├── Train/
    │   ├── actinic keratosis/
    │   │   ├── ISIC_0000001.png
    │   │   └── ...
    │   └── ...
    └── Test/
        └── ...
```

#### Training Results
| Metric | Value |
|--------|-------|
| **Dice Score** | 87.42% |
| **Iterations** | 500 |
| **Training Time** | 32 minutes |

#### Code Example
```python
dataset = build_dataset(
    name='isic',
    root='data/isic',
    split='train',
    depth_slices=16,
    target_resolution=256
)

sample = dataset[0]
print(sample['meta']['modality'])    # "Dermatoscopic"
print(sample['meta']['anatomy'])     # "skin"
print(sample['meta']['classes'])     # [1, 2, 3, ...] (lesion type)
```

#### Citation
```
@article{codella2018skin,
  title={Skin lesion analysis toward melanoma detection},
  author={Codella, Noel CF and Gutman, David and others},
  journal={arXiv preprint arXiv:1710.05006},
  year={2018}
}
```

---

### 3.3 Kvasir-SEG (Polyp Segmentation)

**Dataset**: Kvasir Colonoscopy Dataset  
**Performance**: ✅ **66.76% Dice Score**

#### Overview
- **Source**: Norwegian endoscopy database
- **Modality**: Colonoscopy (RGB → grayscale)
- **Anatomy**: Colon polyps
- **Classes**: Background (0), Polyp (1)
- **Medical Use Case**: Colorectal cancer screening

#### Dataset Characteristics
```python
Dataset Statistics:
- Total Images: 1,000
- Split: Automatic (75% train / 5% val / 20% test)
- Original Resolution: Variable (mostly 720×576 or 1920×1072)
- Target Resolution: 16×256×256
- File Format: JPG (images and masks)
- Challenge: Small polyps, variable lighting, mucus
```

#### Clinical Challenges
- **Variable Lighting**: Endoscope light creates uneven illumination
- **Mucus & Reflections**: Shiny surfaces confuse boundaries
- **Small Targets**: Polyps can be very small (few pixels)
- **Motion Blur**: Patient/scope movement
- **Color Variation**: From pale pink to dark red

#### File Structure
```
data/kvasir/
└── Kvasir-SEG/
    └── Kvasir-SEG/
        ├── images/
        │   ├── cju0qkwl35piu0993l0dewei2.jpg
        │   ├── cju0qoxqj9q6s0835b43399p4.jpg
        │   └── ...
        └── masks/
            ├── cju0qkwl35piu0993l0dewei2.jpg
            ├── cju0qoxqj9q6s0835b43399p4.jpg
            └── ...
```

#### Training Results
| Metric | Value |
|--------|-------|
| **Dice Score** | 66.76% |
| **Iterations** | 1,000 |
| **Training Time** | 32 minutes |

#### Code Example
```python
dataset = build_dataset(
    name='kvasir',
    root='data/kvasir',
    split='train'
)

print(f"Modality: {dataset.modality}")  # "Endoscopy"
print(f"Anatomy: {dataset.anatomy}")    # "colon"
```

#### Citation
```
@inproceedings{jha2020kvasir,
  title={Kvasir-seg: A segmented polyp dataset},
  author={Jha, Debesh and others},
  booktitle={MMM 2020},
  year={2020}
}
```

---

### 3.4 Brain Tumor Segmentation

**Dataset**: Brain Tumor MRI Dataset  
**Performance**: ✅ **42.53% Dice Score**

#### Overview
- **Source**: BraTS Challenge derivatives
- **Modality**: MRI (T1, T2, FLAIR, T1ce → grayscale)
- **Anatomy**: Brain tumors
- **Classes**: Background (0), Tumor (1)
- **Medical Use Case**: Glioma delineation for treatment planning

#### Dataset Characteristics
```python
Dataset Statistics:
- Total Images: Variable
- Split: Automatic
- Original Resolution: 512×512 (2D slices from 3D volumes)
- Target Resolution: 16×256×256
- File Format: PNG/JPG (preprocessed from NIfTI)
- Multi-modal: Often combines T1/T2/FLAIR sequences
```

#### Clinical Challenges
- **Heterogeneous Appearance**: Tumors vary greatly in intensity
- **Ill-Defined Boundaries**: Infiltrative tumors have fuzzy edges
- **Small Targets**: Early-stage tumors are very small
- **Multi-Modal Fusion**: Requires integration of multiple MRI sequences
- **Artifact Sensitivity**: Motion, metal artifacts

#### File Structure
```
data/brain_tumor/
├── images/
│   ├── brain_001.jpg
│   ├── brain_002.jpg
│   └── ...
└── masks/
    ├── brain_001.png
    ├── brain_002.png
    └── ...
```

#### Training Results
| Metric | Value |
|--------|-------|
| **Dice Score** | 42.53% |
| **Iterations** | 500 |
| **Training Time** | 18 minutes |
| **Note** | Lower performance due to extreme difficulty |

#### Code Example
```python
dataset = build_dataset(
    name='brain_tumor',
    root='data/brain_tumor',
    split='train',
    depth_slices=16,
    target_resolution=256
)
```

#### Citation
```
@article{brats2021,
  title={The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)},
  author={Menze, Bjoern H and others},
  journal={IEEE TMI},
  year={2015}
}
```

---

### 3.5 DRIVE (Retinal Vessel Segmentation)

**Dataset**: Digital Retinal Images for Vessel Extraction  
**Performance**: ✅ **21.82% Dice Score**

#### Overview
- **Source**: DRIVE Challenge (diabetic retinopathy screening)
- **Modality**: Fundus photography (RGB → grayscale)
- **Anatomy**: Retinal blood vessels
- **Classes**: Background (0), Vessel (1)
- **Medical Use Case**: Diabetic retinopathy detection, vascular analysis

#### Dataset Characteristics
```python
Dataset Statistics:
- Total Images: 40 (20 train + 20 test)
- Split: Pre-defined train/test
- Original Resolution: 565×584
- Target Resolution: 16×256×256
- File Format: TIF (images), GIF (masks)
- Extreme Class Imbalance: Vessels ~12% of image
```

#### Clinical Challenges
- **Extreme Class Imbalance**: Vessels occupy <12% of pixels
- **Thin Structures**: Vessels are 1-10 pixels wide
- **Branching Topology**: Complex tree-like structure
- **Variable Visibility**: Some vessels barely visible
- **Background Complexity**: Optic disc, lesions, hemorrhages

#### File Structure
```
data/drive/
└── DRIVE/
    ├── training/
    │   ├── images/
    │   │   ├── 21_training.tif
    │   │   └── ...
    │   └── 1st_manual/
    │       ├── 21_manual1.gif
    │       └── ...
    └── test/
        ├── images/
        │   ├── 01_test.tif
        │   └── ...
        └── mask/
            ├── 01_test_mask.gif
            └── ...
```

#### Training Results
| Metric | Value |
|--------|-------|
| **Dice Score** | 21.82% |
| **Iterations** | 1,000 |
| **Training Time** | 37 minutes |
| **Challenge** | Extreme class imbalance, very thin structures |

#### Code Example
```python
dataset = build_dataset(
    name='drive',
    root='data/drive',
    split='train'  # Uses 'training' folder
)

print(f"Modality: {dataset.modality}")  # "Fundus"
print(f"Anatomy: {dataset.anatomy}")    # "retina"
```

#### Citation
```
@article{staal2004ridge,
  title={Ridge-based vessel segmentation in color images of the retina},
  author={Staal, Joes and others},
  journal={IEEE TMI},
  year={2004}
}
```

---

### 3.6-3.10 Additional Datasets (Available, Not Yet Trained)

#### 3.6 AMOS (Abdominal Multi-Organ Segmentation)
- **Modality**: CT
- **Anatomy**: Abdomen (16 organs: spleen, kidney, gallbladder, esophagus, liver, stomach, aorta, postcava, pancreas, adrenal glands, duodenum, bladder, prostate/uterus)
- **Images**: 500 CT scans
- **Classes**: 16 foreground classes
- **Status**: 🔧 Loader implemented (`amos.py`)
- **Use Case**: Multi-organ segmentation in radiotherapy planning

#### 3.7 COVID-19 CT
- **Modality**: CT
- **Anatomy**: Chest/Lungs
- **Classes**: Background (0), COVID-19 lesion (1)
- **Status**: 🔧 Loader implemented (`covid_ct.py`)
- **Use Case**: COVID-19 pneumonia detection

#### 3.8 SegTHOR (Cardiothoracic Segmentation)
- **Modality**: CT
- **Anatomy**: Thorax (heart, aorta, trachea, esophagus)
- **Images**: 40 CT scans
- **Classes**: 5 (4 organs + background)
- **Status**: 🔧 Loader implemented (`segthor.py`)
- **Use Case**: Radiotherapy planning for thoracic cancers

#### 3.9 MSD Pancreas (Medical Segmentation Decathlon)
- **Modality**: CT
- **Anatomy**: Pancreas + tumors
- **Images**: 281 CT scans
- **Classes**: 3 (pancreas, tumor, background)
- **Status**: 🔧 Loader implemented (`msd_pancreas.py`)
- **Use Case**: Pancreatic cancer diagnosis

#### 3.10 ACDC (Automated Cardiac Diagnosis Challenge)
- **Modality**: MRI (cine MRI)
- **Anatomy**: Heart (left ventricle, right ventricle, myocardium)
- **Images**: 150 patients
- **Classes**: 4 (3 cardiac structures + background)
- **Status**: 🔧 Loader implemented (`acdc.py`)
- **Use Case**: Cardiac function assessment

---

### Dataset Diversity Analysis

#### Modality Coverage

```
Medical Imaging Modalities in IRIS:

X-ray         █████████████████████░░░  1 dataset  (10%)
MRI           ████████████████████████  2 datasets (20%)
CT            ████████████████████████  4 datasets (40%)
Fundus        █████████████████████░░░  1 dataset  (10%)
Dermoscopy    █████████████████████░░░  1 dataset  (10%)
Endoscopy     █████████████████████░░░  1 dataset  (10%)
```

#### Anatomical Coverage

```
Anatomical Regions:

Chest/Lungs   ████████████████████░░░░  2 datasets (20%)
Brain         ██████████░░░░░░░░░░░░░░  1 dataset  (10%)
Skin          ██████████░░░░░░░░░░░░░░  1 dataset  (10%)
Retina        ██████████░░░░░░░░░░░░░░  1 dataset  (10%)
Colon         ██████████░░░░░░░░░░░░░░  1 dataset  (10%)
Abdomen       ██████████░░░░░░░░░░░░░░  1 dataset  (10%)
Pancreas      ██████████░░░░░░░░░░░░░░  1 dataset  (10%)
Heart         ██████████░░░░░░░░░░░░░░  1 dataset  (10%)
Thorax        ██████████░░░░░░░░░░░░░░  1 dataset  (10%)
```

#### Performance vs. Difficulty

```mermaid
graph LR
    subgraph Easy["Easy (>80% Dice)"]
        CHEST["Chest X-Ray<br/>95.81%"]
        ISIC2["ISIC<br/>87.42%"]
    end
    
    subgraph Medium["Medium (50-80% Dice)"]
        KVASIR2["Kvasir<br/>66.76%"]
    end
    
    subgraph Hard["Hard (<50% Dice)"]
        BRAIN2["Brain Tumor<br/>42.53%"]
        DRIVE2["DRIVE<br/>21.82%"]
    end
    
    style Easy fill:#c8e6c9,stroke:#388e3c
    style Medium fill:#fff9c4,stroke:#f57c00
    style Hard fill:#ffcdd2,stroke:#c62828
```

**Performance Insights**:
- **Best**: Chest X-Ray (95.81%) - Large, clear boundaries, high contrast
- **Good**: ISIC (87.42%) - Clear lesion boundaries, moderate size
- **Moderate**: Kvasir (66.76%) - Small targets, variable lighting
- **Challenging**: Brain Tumor (42.53%) - Ill-defined boundaries, infiltrative
- **Hardest**: DRIVE (21.82%) - Extreme class imbalance, thin structures

---

## 4. Data Loading Architecture

### Overview

IRIS's data loading system is built on three core abstractions:

1. **VolumeRecord**: Metadata for a single image-mask pair
2. **MedicalDataset**: Base class for all datasets
3. **EpisodicBatchSampler**: Constructs support-query episodes

```mermaid
graph TB
    subgraph Abstract["Abstract Base Classes"]
        BASE["MedicalDataset<br/>(Abstract Base)"]
        RECORD["VolumeRecord<br/>(Dataclass)"]
    end
    
    subgraph Concrete["Concrete Implementations"]
        CHEST_D["ChestXrayMasksDataset"]
        BRAIN_D["BrainTumorDataset"]
        ISIC_D["ISICDataset"]
        OTHER["... 7 more datasets"]
    end
    
    subgraph Sampling["Episodic Sampling"]
        SAMPLER["EpisodicBatchSampler"]
        LOADER["PyTorch DataLoader"]
    end
    
    subgraph Output["Episode Structure"]
        EPISODE["Episode Dictionary<br/>{support_images, support_masks,<br/>query_images, query_masks,<br/>class_ids, metadata}"]
    end
    
    RECORD --> BASE
    BASE --> CHEST_D
    BASE --> BRAIN_D
    BASE --> ISIC_D
    BASE --> OTHER
    
    CHEST_D --> SAMPLER
    BRAIN_D --> SAMPLER
    ISIC_D --> SAMPLER
    OTHER --> SAMPLER
    
    SAMPLER --> LOADER
    LOADER --> EPISODE
    
    style Abstract fill:#fff3e0,stroke:#f57c00
    style Concrete fill:#e3f2fd,stroke:#1976d2
    style Sampling fill:#e8f5e9,stroke:#388e3c
    style Output fill:#f3e5f5,stroke:#7b1fa2
```

### 4.1 VolumeRecord (Metadata Container)

**File**: `src/iris/data/base.py`

```python
@dataclass
class VolumeRecord:
    """
    Metadata describing a single medical imaging volume and its segmentation mask.
    
    This dataclass stores all information needed to locate and load a medical
    image-mask pair, along with relevant metadata for preprocessing and logging.
    """
    
    # File paths
    image_path: Path              # Path to image (NIfTI, PNG, JPG, etc.)
    mask_path: Optional[Path]     # Path to mask (can be None for unlabeled data)
    
    # Identifiers
    subject_id: str               # Unique ID (e.g., "patient_001")
    dataset_name: str             # Dataset identifier (e.g., "chest_xray_masks")
    
    # Medical metadata
    modality: str                 # Imaging modality ("CT", "MRI", "X-ray", etc.)
    anatomy: str                  # Anatomical region ("chest", "brain", etc.)
    classes: Sequence[int]        # Label IDs in mask (e.g., [1] for binary)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def as_dict(self) -> Dict[str, Any]:
        """Serialize for logging or caching."""
        return {
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "subject_id": self.subject_id,
            "dataset_name": self.dataset_name,
            "modality": self.modality,
            "anatomy": self.anatomy,
            "classes": list(self.classes),
            "metadata": dict(self.metadata),
        }
```

**Usage Example**:
```python
record = VolumeRecord(
    image_path=Path("data/chest_xray/images/MCUCXR_0001_0.png"),
    mask_path=Path("data/chest_xray/masks/MCUCXR_0001_0.png"),
    subject_id="MCUCXR_0001_0",
    dataset_name="chest_xray_masks",
    modality="X-ray",
    anatomy="chest",
    classes=[1],  # Single foreground class
    metadata={
        "image_filename": "MCUCXR_0001_0.png",
        "original_resolution": (4020, 4892)
    }
)
```

### 4.2 MedicalDataset (Base Class)

**File**: `src/iris/data/base.py`

The `MedicalDataset` base class implements the **template method pattern**, where subclasses override specific hooks while inheriting common functionality.

#### Class Structure

```python
class MedicalDataset(Dataset):
    """
    Base dataset implementation handling medical volume loading and preprocessing.
    
    Subclasses must implement:
        - discover_records(): Enumerate image-mask pairs
    
    Subclasses can optionally override:
        - configure_preprocessing(): Dataset-specific preprocessing settings
        - _load_record(): Custom loading logic
    """
    
    # Class attributes (override in subclasses)
    dataset_name: str = "medical"
    modality: str = "CT"
    anatomy: str = "generic"
    target_classes: Optional[Sequence[int]] = None
    
    def __init__(
        self,
        root: Path | str,
        split: DatasetSplit = DatasetSplit.TRAIN,
        *,
        transforms: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        spacing: Optional[Tuple[float, float, float]] = None,
        cache_data: bool = False,
        random_seed: int = 42,
        allow_missing_masks: bool = False,
        **preprocess_overrides: Any,
    ) -> None:
        # Validate root exists
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        
        # Store configuration
        self.split = split
        self.transforms = transforms
        self.augmentation = augmentation
        self.target_size = target_size
        self.spacing = spacing
        self.cache_data = cache_data
        self.random_seed = random_seed
        self.allow_missing_masks = allow_missing_masks
        self._preprocess_overrides = preprocess_overrides
        
        # Discover all records
        records = list(self.discover_records())
        if not records:
            raise RuntimeError(
                f"No volume records discovered for {self.dataset_name} at {self.root}"
            )
        
        # Split into train/val/test
        self._records_by_split = default_split_strategy(records)
        self.records = self._records_by_split[split]
        
        # Optional data cache
        self._cache: Dict[str, Dict[str, Any]] = {}
```

#### Key Methods

**1. discover_records() - MUST IMPLEMENT**

Subclasses enumerate all image-mask pairs in dataset:

```python
def discover_records(self) -> Iterable[VolumeRecord]:
    """
    Find all image-mask pairs in dataset directory.
    
    Returns:
        Iterable of VolumeRecord instances
    
    Example Implementation:
        records = []
        for image_path in self.images_root.glob("*.png"):
            mask_path = self.masks_root / image_path.name
            records.append(VolumeRecord(
                image_path=image_path,
                mask_path=mask_path,
                subject_id=image_path.stem,
                dataset_name=self.dataset_name,
                modality=self.modality,
                anatomy=self.anatomy,
                classes=[1]
            ))
        return records
    """
    raise NotImplementedError
```

**2. __getitem__() - AUTO-IMPLEMENTED**

Loads and preprocesses a single sample:

```python
def __getitem__(self, index: int) -> Dict[str, Any]:
    """
    Load sample by index.
    
    Returns:
        Dictionary with keys:
            - 'image': torch.Tensor of shape (1, D, H, W)
            - 'mask': torch.Tensor of shape (1, D, H, W)
            - 'meta': Dict with metadata (subject_id, modality, etc.)
    """
    record = self.records[index]
    
    # Check cache
    if self.cache_data and record.subject_id in self._cache:
        sample = self._cache[record.subject_id].copy()
    else:
        # Load from disk
        sample = self._load_record(record)
        if self.cache_data:
            self._cache[record.subject_id] = sample.copy()
    
    # Apply augmentation (training only)
    if self.augmentation and self.split == DatasetSplit.TRAIN:
        sample = self.augmentation(sample)
    
    # Apply transforms
    if self.transforms:
        sample = self.transforms(sample)
    
    # Add index metadata
    sample["meta"]["index"] = index
    sample["meta"]["split"] = self.split.value
    sample["meta"]["dataset"] = self.dataset_name
    
    return sample
```

**3. _load_record() - CAN OVERRIDE**

Loads and preprocesses a single record:

```python
def _load_record(self, record: VolumeRecord) -> Dict[str, Any]:
    """
    Load image and mask from disk, apply preprocessing.
    
    Default implementation:
        1. Load image/mask using load_medical_volume()
        2. Apply preprocessing (normalize, resize, etc.)
        3. Return dict with image, mask, metadata
    
    Subclasses can override for custom loading logic.
    """
    # Load files
    image, image_meta = load_medical_volume(record.image_path)
    mask, mask_meta = load_medical_volume(record.mask_path)
    
    # Get preprocessing config
    config = self.configure_preprocessing()
    config.update(self._preprocess_overrides)
    
    # Preprocess
    processed = preprocess_image_and_mask(
        image=image,
        mask=mask,
        modality=self.modality,
        target_size=config["target_size"],
        target_spacing=config["target_spacing"],
        ...
    )
    
    # Add metadata
    processed["meta"].update({
        "subject_id": record.subject_id,
        "dataset_name": record.dataset_name,
        "anatomy": record.anatomy,
        "modality": record.modality,
    })
    
    return processed
```

#### Train/Val/Test Splitting

**Default Strategy** (75% / 5% / 20%):

```python
def default_split_strategy(
    records: Sequence[VolumeRecord],
    train_ratio: float = 0.75,
    val_ratio: float = 0.05,
    random_seed: int = 42,
) -> Dict[DatasetSplit, List[VolumeRecord]]:
    """
    Deterministically split records into train/val/test.
    
    Args:
        records: All dataset records
        train_ratio: Fraction for training (default: 0.75)
        val_ratio: Fraction for validation (default: 0.05)
        random_seed: Seed for reproducibility
    
    Returns:
        Dict mapping DatasetSplit to list of records
    """
    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)  # Shuffle deterministically
    
    total = len(records)
    train_count = int(train_ratio * total)
    val_count = int(val_ratio * total)
    
    splits = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.VALID: [],
        DatasetSplit.TEST: [],
    }
    
    for i, idx in enumerate(indices):
        if i < train_count:
            splits[DatasetSplit.TRAIN].append(records[idx])
        elif i < train_count + val_count:
            splits[DatasetSplit.VALID].append(records[idx])
        else:
            splits[DatasetSplit.TEST].append(records[idx])
    
    return splits
```

### 4.3 Episodic Batch Sampler

**File**: `src/iris/data/samplers.py`

The `EpisodicBatchSampler` constructs support-query episodes for few-shot learning.

```python
class EpisodicBatchSampler:
    """
    Sample support-query episodes from dataset.
    
    Each episode contains:
        - n_support examples (with masks)
        - n_query examples (with masks)
    
    All examples in an episode are from the same dataset.
    """
    
    def __init__(
        self,
        indices: Sequence[int],           # All dataset indices
        *,
        dataset_names: Sequence[str],     # Dataset name per index
        n_support: int,                    # Support set size (K)
        n_query: int,                      # Query set size (B)
        episodes_per_epoch: int,           # Episodes per epoch
        rng: np.random.Generator = None,   # Random generator
    ) -> None:
        self.indices = list(indices)
        self.dataset_names = list(dataset_names)
        self.n_support = n_support
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        self.rng = rng or np.random.default_rng()
        
        # Group indices by dataset
        self._indices_by_dataset: Dict[str, List[int]] = defaultdict(list)
        for idx, name in zip(self.indices, self.dataset_names):
            self._indices_by_dataset[name].append(idx)
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield episodes (lists of indices)."""
        dataset_keys = list(self._indices_by_dataset.keys())
        
        for _ in range(self.episodes_per_epoch):
            # 1. Sample a dataset
            dataset = str(self.rng.choice(dataset_keys))
            candidates = self._indices_by_dataset[dataset]
            
            # 2. Check if enough samples
            if len(candidates) < self.n_support + self.n_query:
                raise ValueError(
                    f"Dataset {dataset} needs at least "
                    f"{self.n_support + self.n_query} samples"
                )
            
            # 3. Sample support + query indices
            shuffled = self.rng.permutation(candidates)
            support_indices = shuffled[:self.n_support]
            query_indices = shuffled[self.n_support:self.n_support + self.n_query]
            
            # 4. Yield combined indices
            yield support_indices.tolist() + query_indices.tolist()
    
    def __len__(self) -> int:
        return self.episodes_per_epoch
```

#### Usage Example

```python
from iris.data import build_dataset, EpisodicBatchSampler
from torch.utils.data import DataLoader

# 1. Build dataset
dataset = build_dataset('chest_xray_masks', root='data/', split='train')

# 2. Create episodic sampler
sampler = EpisodicBatchSampler(
    indices=list(range(len(dataset))),
    dataset_names=[dataset.dataset_name] * len(dataset),
    n_support=3,          # K=3 support examples
    n_query=4,            # B=4 query examples
    episodes_per_epoch=100,  # 100 episodes = 1 epoch
)

# 3. Create DataLoader
loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=4,
    pin_memory=True
)

# 4. Training loop
for episode_batch in loader:
    # episode_batch is a list of K+B=7 samples
    support_samples = episode_batch[:3]   # First K samples
    query_samples = episode_batch[3:]     # Remaining B samples
    
    # Stack into tensors
    support_images = torch.stack([s['image'] for s in support_samples])
    support_masks = torch.stack([s['mask'] for s in support_samples])
    query_images = torch.stack([s['image'] for s in query_samples])
    query_masks = torch.stack([s['mask'] for s in query_samples])
    
    # Train model...
```

### 4.4 Complete Data Loading Pipeline

**End-to-End Example**:

```python
from iris.data import build_dataset
from torch.utils.data import DataLoader

# Step 1: Build dataset
train_dataset = build_dataset(
    name='chest_xray_masks',
    root='data/chest_xray',
    split='train',
    target_size=(16, 256, 256),
    cache_data=False
)

# Step 2: Create standard DataLoader (non-episodic)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Step 3: Iterate
for batch in train_loader:
    images = batch['image']       # (4, 1, 16, 256, 256)
    masks = batch['mask']         # (4, 1, 16, 256, 256)
    meta = batch['meta']
    
    # Training logic...
    print(f"Batch shape: {images.shape}")
    print(f"Dataset: {meta['dataset'][0]}")
    print(f"Modality: {meta['modality'][0]}")
```

---

## 5. Preprocessing Pipeline

### Overview

Medical image preprocessing is critical for model performance. IRIS implements a multi-stage preprocessing pipeline that handles variable resolutions, different modalities, and clinical value ranges.

### Preprocessing Architecture

```mermaid
flowchart TB
    A[Raw Medical Volume] --> B{Check Spacing}
    B -->|Has Physical Spacing| C[Resample to Target Spacing]
    B -->|No Spacing| D[Resize to Target Size]
    C --> E[Resize to 128³]
    D --> E
    E --> F{Determine Modality}
    F -->|CT| G[Hounsfield Normalization]
    F -->|MRI| H[Percentile Normalization]
    F -->|PET| I[Z-Score Normalization]
    F -->|Other| J[Min-Max Normalization]
    G --> K[Convert to Tensor]
    H --> K
    I --> K
    J --> K
    K --> L[Add Channel Dimension]
    L --> M[Preprocessed Sample]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style M fill:#9f9,stroke:#333,stroke-width:2px
    style F fill:#ff9,stroke:#333,stroke-width:2px
```

### Key Preprocessing Steps

#### 1. **Resampling (Physical Spacing Normalization)**

When physical spacing metadata is available, volumes are resampled to consistent spacing:

```python
def _resample_volume(
    volume: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int = 3
) -> np.ndarray:
    """
    Resample volume to target physical spacing using scipy.
    
    Args:
        volume: Input volume (D, H, W)
        current_spacing: Current voxel spacing in mm (z, y, x)
        target_spacing: Target voxel spacing in mm (z, y, x)
        order: Interpolation order (3=cubic for images, 0=nearest for masks)
    
    Returns:
        Resampled volume
    """
    from scipy.ndimage import zoom
    
    # Calculate zoom factors from spacing ratios
    zoom_factors = np.array(current_spacing) / np.array(target_spacing)
    
    # Apply resampling
    resampled = zoom(volume, zoom_factors, order=order)
    return resampled
```

**Example**:
```python
# CT scan with 2mm slice thickness, 0.5mm in-plane resolution
current_spacing = (2.0, 0.5, 0.5)  # (z, y, x) in mm
target_spacing = (1.0, 1.0, 1.0)   # Isotropic 1mm³

# Original: (100, 512, 512) → Resampled: (200, 256, 256)
resampled_ct = _resample_volume(ct_volume, current_spacing, target_spacing, order=3)
```

#### 2. **Resizing (Fixed Dimension Normalization)**

All volumes are resized to a fixed target size (default: 128×128×128) for batch processing:

```python
def _resize_volume(
    volume: np.ndarray,
    target_size: Tuple[int, int, int],
    order: int = 3
) -> np.ndarray:
    """
    Resize volume to fixed target dimensions.
    
    Args:
        volume: Input volume (D, H, W)
        target_size: Target dimensions (D', H', W')
        order: Interpolation order
    
    Returns:
        Resized volume of shape target_size
    """
    from scipy.ndimage import zoom
    
    # Calculate zoom factors from size ratios
    zoom_factors = np.array(target_size) / np.array(volume.shape)
    
    # Apply resizing
    resized = zoom(volume, zoom_factors, order=order)
    return resized
```

**Key Points**:
- **Order 3 (cubic)** for images: Smooth interpolation
- **Order 0 (nearest)** for masks: Preserve discrete class labels
- Default target: `(128, 128, 128)` balances memory and detail

#### 3. **Intensity Normalization (Modality-Specific)**

Different medical imaging modalities require different normalization strategies:

##### **CT Scans (Hounsfield Units)**

```python
def _normalize_ct(
    volume: np.ndarray,
    clip_values: Tuple[float, float] = (-1024, 1024)
) -> np.ndarray:
    """
    Normalize CT scan using Hounsfield unit windowing.
    
    Hounsfield Units (HU):
        -1024: Air
        -500 to -100: Lung tissue
        0: Water
        +30 to +70: Soft tissue
        +400 to +1000: Bone
    
    Args:
        volume: Raw CT volume in Hounsfield units
        clip_values: Min/max values for windowing (default: soft tissue window)
    
    Returns:
        Normalized volume in [0, 1]
    """
    # Clip to clinical range
    clipped = np.clip(volume, clip_values[0], clip_values[1])
    
    # Normalize to [0, 1]
    normalized = (clipped - clip_values[0]) / (clip_values[1] - clip_values[0])
    
    return normalized.astype(np.float32)
```

**Common CT Windows**:
```python
# Soft tissue window (default)
soft_tissue = (-160, 240)

# Lung window
lung = (-1024, -400)

# Bone window
bone = (-200, 1000)

# Abdomen window
abdomen = (-150, 250)
```

##### **MRI Scans (Percentile Clipping)**

```python
def _normalize_mri(
    volume: np.ndarray,
    percentiles: Tuple[float, float] = (1.0, 99.0)
) -> np.ndarray:
    """
    Normalize MRI using percentile-based clipping.
    
    MRI intensities are arbitrary and scanner-dependent.
    Percentile clipping removes outliers and normalizes distribution.
    
    Args:
        volume: Raw MRI volume
        percentiles: Lower and upper percentiles for clipping
    
    Returns:
        Normalized volume in [0, 1]
    """
    # Calculate percentile values
    p_low = np.percentile(volume, percentiles[0])
    p_high = np.percentile(volume, percentiles[1])
    
    # Clip to percentile range
    clipped = np.clip(volume, p_low, p_high)
    
    # Normalize to [0, 1]
    if p_high > p_low:
        normalized = (clipped - p_low) / (p_high - p_low)
    else:
        normalized = np.zeros_like(clipped)
    
    return normalized.astype(np.float32)
```

**Why Percentiles for MRI?**
- MRI intensities lack standardized units (unlike CT's Hounsfield units)
- Different scanners/sequences produce different intensity ranges
- Percentile clipping ensures consistent normalization across datasets

##### **PET Scans (Z-Score Normalization)**

```python
def _normalize_pet(volume: np.ndarray) -> np.ndarray:
    """
    Normalize PET scans using z-score normalization.
    
    PET measures metabolic activity (SUV - Standardized Uptake Value).
    Z-score ensures mean=0, std=1 distribution.
    """
    mean = volume.mean()
    std = volume.std()
    
    if std > 0:
        normalized = (volume - mean) / std
    else:
        normalized = volume - mean
    
    return normalized.astype(np.float32)
```

##### **Unified Normalization Interface**

```python
def normalize_intensity(
    volume: np.ndarray,
    modality: str,
    clip_values: Optional[Tuple[float, float]] = None,
    mri_percentiles: Tuple[float, float] = (1.0, 99.0)
) -> np.ndarray:
    """
    Normalize intensity based on modality.
    
    Args:
        volume: Input volume
        modality: "CT", "MRI", "PET", "X-ray", etc.
        clip_values: For CT windowing
        mri_percentiles: For MRI clipping
    
    Returns:
        Normalized volume in [0, 1]
    """
    if modality.upper() == "CT":
        clip_values = clip_values or (-1024, 1024)
        return _normalize_ct(volume, clip_values)
    
    elif modality.upper() == "MRI":
        return _normalize_mri(volume, mri_percentiles)
    
    elif modality.upper() == "PET":
        normalized = _normalize_pet(volume)
        # Map to [0, 1] range
        return (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
    
    else:
        # Default: min-max normalization
        vol_min, vol_max = volume.min(), volume.max()
        if vol_max > vol_min:
            return (volume - vol_min) / (vol_max - vol_min)
        return volume.astype(np.float32)
```

#### 4. **Complete Preprocessing Pipeline**

```python
def preprocess_image_and_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    image_meta: Dict[str, Any],
    modality: str,
    target_size: Tuple[int, int, int] = (128, 128, 128),
    target_spacing: Optional[Tuple[float, float, float]] = None,
    clip_values: Optional[Tuple[float, float]] = None,
    mri_percentiles: Tuple[float, float] = (1.0, 99.0),
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for medical images.
    
    Pipeline:
        1. Resample to target spacing (if provided)
        2. Resize to target size
        3. Normalize intensity (modality-specific)
        4. Convert to PyTorch tensors
        5. Add channel dimension
    
    Args:
        image: Raw medical image volume
        mask: Segmentation mask (or None)
        image_meta: Metadata dict with spacing, affine, etc.
        modality: Image modality ("CT", "MRI", etc.)
        target_size: Output size (default: 128³)
        target_spacing: Target physical spacing in mm
        clip_values: CT windowing values
        mri_percentiles: MRI percentile clipping
        random_state: RNG seed for reproducibility
    
    Returns:
        Preprocessed sample dict with:
            - image: Tensor of shape (1, D, H, W)
            - mask: Tensor of shape (D, H, W) or None
            - meta: Metadata dict
    """
    import torch
    
    # Extract current spacing
    current_image_spacing = image_meta.get("spacing", None)
    
    # Step 1: Resample if spacing available
    if target_spacing and current_image_spacing:
        image = _resample_volume(image, current_image_spacing, target_spacing, order=3)
        if mask is not None:
            mask = _resample_volume(mask, current_image_spacing, target_spacing, order=0)
    
    # Step 2: Resize to target size
    resized_image = _resize_volume(image, target_size, order=3)
    resized_mask = _resize_volume(mask, target_size, order=0) if mask is not None else None
    
    # Step 3: Normalize intensity (modality-specific)
    normalized_image = normalize_intensity(
        resized_image,
        modality,
        clip_values=clip_values,
        mri_percentiles=mri_percentiles
    )
    
    # Step 4: Prepare mask
    if resized_mask is not None:
        resized_mask = resized_mask.astype(np.int16)
        unique_classes = np.unique(resized_mask)
    else:
        unique_classes = np.array([], dtype=np.int16)
    
    # Step 5: Convert to tensors
    sample = {
        "image": torch.from_numpy(normalized_image).unsqueeze(0),  # (1, D, H, W)
        "mask": torch.from_numpy(resized_mask) if resized_mask is not None else None,  # (D, H, W)
        "meta": {
            "affine": image_meta.get("affine"),
            "spacing": target_spacing or current_image_spacing,
            "original_spacing": current_image_spacing,
            "original_shape": image_meta.get("original_shape"),
            "target_size": target_size,
            "unique_classes": unique_classes.tolist(),
            "modality": modality,
            "rng_seed": random_state,
            **(image_meta or {})
        }
    }
    
    return sample
```

### Preprocessing Configuration

```python
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Spatial parameters
    target_size: Tuple[int, int, int] = (128, 128, 128)
    target_spacing: Optional[Tuple[float, float, float]] = None
    
    # Modality-specific parameters
    modality: str = "CT"
    ct_clip_values: Tuple[float, float] = (-1024, 1024)
    mri_percentiles: Tuple[float, float] = (1.0, 99.0)
    
    # Interpolation
    image_interpolation_order: int = 3  # Cubic
    mask_interpolation_order: int = 0   # Nearest neighbor
    
    # Reproducibility
    random_seed: Optional[int] = 42

# Usage
config = PreprocessingConfig(
    target_size=(128, 128, 128),
    modality="CT",
    ct_clip_values=(-160, 240)  # Soft tissue window
)
```

### Preprocessing Best Practices

| Best Practice | Rationale |
|--------------|-----------|
| **Use cubic interpolation for images** | Preserves smooth transitions and detail |
| **Use nearest neighbor for masks** | Preserves discrete class labels (no interpolation artifacts) |
| **Apply same transforms to image and mask** | Maintains spatial alignment |
| **Normalize before augmentation** | Ensures augmentations work on [0,1] range |
| **Cache preprocessed data if possible** | Preprocessing is computationally expensive |
| **Validate preprocessed output** | Check for NaN, extreme values, shape mismatches |

### Example: Complete Preprocessing

```python
from iris.data.preprocessing import preprocess_image_and_mask
from iris.data.io import load_medical_volume

# Load raw medical volume
image, image_meta = load_medical_volume("datasets/chest_xray/case_001/image.nii.gz")
mask, mask_meta = load_medical_volume("datasets/chest_xray/case_001/mask.nii.gz")

# Preprocess
sample = preprocess_image_and_mask(
    image=image,
    mask=mask,
    image_meta=image_meta,
    modality="X-ray",
    target_size=(128, 128, 128),
    target_spacing=None  # X-rays are 2D, no spacing
)

# Result
print(sample["image"].shape)  # torch.Size([1, 128, 128, 128])
print(sample["mask"].shape)   # torch.Size([128, 128, 128])
print(sample["meta"]["unique_classes"])  # [0, 1] (background and lung)
```

---

## 6. Data Augmentation

### Overview

Data augmentation is essential for few-shot learning to:
1. **Increase sample diversity** with limited training data
2. **Improve model generalization** across anatomical variations
3. **Simulate real-world variability** in patient positioning, scanner settings

IRIS implements a composable augmentation system in `src/iris/data/augmentations.py` with medical-specific transformations.

### Augmentation Architecture

```mermaid
flowchart TB
    A[Preprocessed Sample] --> B[Random Crop]
    B --> C[Random Flip]
    C --> D[Intensity Shift/Scale]
    D --> E[Affine Transform]
    E --> F{Has Mask?}
    F -->|Yes| G[Random Class Drop]
    F -->|No| H[Augmented Sample]
    G --> H
    
    I[3D Rotation] --> E
    J[3D Scaling] --> E
    K[3D Translation] --> E
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#9f9,stroke:#333,stroke-width:2px
    style E fill:#ff9,stroke:#333,stroke-width:2px
```

### Core Augmentation Operations

#### 1. **Random Cropping**

Extracts random 3D patches for computational efficiency:

```python
def _random_crop(
    image: torch.Tensor,
    mask: Optional[torch.Tensor],
    crop_size: Tuple[int, int, int],
    rng: np.random.Generator
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Random spatial crop for 3D volumes.
    
    Args:
        image: (1, D, H, W) input tensor
        mask: (D, H, W) or (1, D, H, W) mask tensor
        crop_size: (crop_d, crop_h, crop_w)
        rng: NumPy random generator
    
    Returns:
        Cropped image and mask
    """
    _, depth, height, width = image.shape
    cz, cy, cx = crop_size
    
    # Validate crop size
    if cz > depth or cy > height or cx > width:
        return image, mask
    
    # Random starting indices
    z = rng.integers(0, depth - cz + 1)
    y = rng.integers(0, height - cy + 1)
    x = rng.integers(0, width - cx + 1)
    
    # Crop image
    cropped_image = image[:, z:z+cz, y:y+cy, x:x+cx]
    
    # Crop mask
    if mask is None:
        cropped_mask = None
    elif mask.ndim == 4:
        cropped_mask = mask[:, z:z+cz, y:y+cy, x:x+cx]
    else:
        cropped_mask = mask[z:z+cz, y:y+cy, x:x+cx]
    
    return cropped_image, cropped_mask
```

**Example**:
```python
# Original: (1, 128, 128, 128)
# Crop to: (1, 112, 112, 112) for faster training
cropped_img, cropped_mask = _random_crop(image, mask, (112, 112, 112), rng)
```

#### 2. **Random Flipping**

Spatial flipping along anatomical axes:

```python
def _random_flip(
    image: torch.Tensor,
    mask: Optional[torch.Tensor],
    flip_axes: Tuple[bool, bool, bool]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Random flipping along spatial axes.
    
    Args:
        image: (1, D, H, W) input tensor
        mask: (D, H, W) or (1, D, H, W) mask tensor
        flip_axes: (flip_z, flip_y, flip_x) boolean tuple
    
    Returns:
        Flipped image and mask
    """
    # Flip along depth (z-axis)
    if flip_axes[0]:
        image = torch.flip(image, dims=[1])
        if mask is not None:
            dim = 1 if mask.ndim == 4 else 0
            mask = torch.flip(mask, dims=[dim])
    
    # Flip along height (y-axis)
    if flip_axes[1]:
        image = torch.flip(image, dims=[2])
        if mask is not None:
            dim = 2 if mask.ndim == 4 else 1
            mask = torch.flip(mask, dims=[dim])
    
    # Flip along width (x-axis)
    if flip_axes[2]:
        image = torch.flip(image, dims=[3])
        if mask is not None:
            dim = 3 if mask.ndim == 4 else 2
            mask = torch.flip(mask, dims=[dim])
    
    return image, mask
```

**Medical Imaging Considerations**:
- **Left-right flip**: Often safe (e.g., lungs are symmetric)
- **Superior-inferior flip**: Generally NOT safe (anatomy is not vertically symmetric)
- **Anterior-posterior flip**: Depends on organ (safe for brain, not for abdomen)

**Example**:
```python
# Flip with 50% probability along each axis
flip_axes = (rng.random() < 0.5, rng.random() < 0.5, rng.random() < 0.5)
flipped_img, flipped_mask = _random_flip(image, mask, flip_axes)
```

#### 3. **Intensity Augmentation**

Simulates scanner variability and contrast differences:

```python
def _random_intensity_shift(
    image: torch.Tensor,
    max_shift: float,
    max_scale: float,
    rng: np.random.Generator
) -> torch.Tensor:
    """
    Random brightness/contrast augmentation.
    
    Formula:
        augmented = image * scale + shift
    
    Args:
        image: Normalized image in [0, 1]
        max_shift: Max brightness shift (e.g., 0.1 = ±10%)
        max_scale: Max contrast scale (e.g., 0.15 = ±15%)
        rng: Random generator
    
    Returns:
        Augmented image clipped to [0, 1]
    """
    # Random shift and scale
    shift = rng.uniform(-max_shift, max_shift)
    scale = rng.uniform(1.0 - max_scale, 1.0 + max_scale)
    
    # Apply transformation
    augmented = image * scale + shift
    
    # Clip to valid range
    return augmented.clamp(0.0, 1.0)
```

**Typical Values**:
```python
max_shift = 0.1    # ±10% brightness shift
max_scale = 0.15   # ±15% contrast scaling
```

#### 4. **Affine Transformations (3D Rotation, Scaling, Translation)**

Simulates patient positioning variability:

```python
def _affine_matrix(
    angles: Tuple[float, float, float],
    scales: Tuple[float, float, float],
    translations: Tuple[float, float, float]
) -> np.ndarray:
    """
    Construct 4×4 affine transformation matrix.
    
    Combines:
        - Rotation around x, y, z axes (radians)
        - Scaling along x, y, z axes
        - Translation along x, y, z axes
    
    Args:
        angles: (rx, ry, rz) rotation angles in radians
        scales: (sx, sy, sz) scale factors
        translations: (tx, ty, tz) translation in voxels
    
    Returns:
        4×4 affine transformation matrix
    """
    rx, ry, rz = angles
    sx, sy, sz = scales
    tx, ty, tz = translations
    
    # Rotation matrices
    cx, sx_sin = np.cos(rx), np.sin(rx)
    cy, sy_sin = np.cos(ry), np.sin(ry)
    cz, sz_sin = np.cos(rz), np.sin(rz)
    
    rot_x = np.array([[1, 0, 0], 
                      [0, cx, -sx_sin], 
                      [0, sx_sin, cx]], dtype=np.float32)
    
    rot_y = np.array([[cy, 0, sy_sin], 
                      [0, 1, 0], 
                      [-sy_sin, 0, cy]], dtype=np.float32)
    
    rot_z = np.array([[cz, -sz_sin, 0], 
                      [sz_sin, cz, 0], 
                      [0, 0, 1]], dtype=np.float32)
    
    # Combined rotation: R = Rz @ Ry @ Rx
    rotation = rot_z @ rot_y @ rot_x
    
    # Scale matrix
    scale_mat = np.diag([sx, sy, sz]).astype(np.float32)
    
    # Combined affine: A = R @ S
    affine = rotation @ scale_mat
    
    # 4×4 homogeneous matrix
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = affine
    matrix[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    
    return matrix


def _apply_affine(
    tensor: torch.Tensor,
    matrix: np.ndarray,
    order: int
) -> torch.Tensor:
    """
    Apply affine transformation to 3D tensor.
    
    Args:
        tensor: (1, D, H, W) or (D, H, W) tensor
        matrix: 4×4 affine matrix
        order: Interpolation order (3 for images, 0 for masks)
    
    Returns:
        Transformed tensor
    """
    from scipy.ndimage import affine_transform
    
    # Convert to numpy
    array = tensor.cpu().numpy()
    had_channel = False
    
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
        had_channel = True
    
    # Extract 3×3 rotation/scale and 3×1 translation
    matrix3x3 = matrix[:3, :3]
    offset = matrix[:3, 3]
    
    # Apply transformation
    transformed = affine_transform(
        array,
        matrix3x3,
        offset=offset,
        order=order,
        mode='nearest',
        cval=0.0
    )
    
    # Convert back to tensor
    if had_channel:
        transformed_tensor = torch.from_numpy(transformed).unsqueeze(0).type_as(tensor)
    else:
        transformed_tensor = torch.from_numpy(transformed).type_as(tensor)
    
    return transformed_tensor
```

**Typical Ranges**:
```python
rotation_range = (10.0, 10.0, 10.0)    # ±10° around each axis
scale_range = (0.9, 1.1)                # ±10% scaling
translation_range = (5.0, 5.0, 5.0)     # ±5 voxels translation
```

#### 5. **Random Class Drop (Medical-Specific)**

Simulates missing or ambiguous annotations:

```python
def _random_class_drop(
    mask: torch.Tensor,
    drop_prob: float,
    rng: np.random.Generator
) -> torch.Tensor:
    """
    Randomly remove a class from the segmentation mask.
    
    Useful for:
        - Simulating annotation uncertainty
        - Training robustness to missing labels
        - Multi-organ segmentation scenarios
    
    Args:
        mask: Segmentation mask with class labels
        drop_prob: Probability of dropping a class
        rng: Random generator
    
    Returns:
        Mask with one class potentially removed
    """
    unique_classes = torch.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]  # Exclude background
    
    if len(unique_classes) == 0:
        return mask
    
    if rng.random() < drop_prob:
        # Randomly select a class to drop
        drop_class = int(rng.choice(unique_classes.cpu().numpy()))
        
        # Set that class to background (0)
        mask = mask.clone()
        mask[mask == drop_class] = 0
    
    return mask
```

### Complete Augmentation Pipeline

```python
class MedicalAugmentation:
    """
    Composable augmentation callable for 3D medical volumes.
    
    Applies in sequence:
        1. Random crop
        2. Random flip
        3. Intensity shift/scale
        4. Affine transform (rotation + scale + translation)
        5. Random class drop (optional)
    """
    
    def __init__(
        self,
        *,
        crop_size: Optional[Tuple[int, int, int]] = (112, 112, 112),
        flip_prob: float = 0.5,
        intensity_shift: float = 0.1,
        intensity_scale: float = 0.15,
        rotation_range: Tuple[float, float, float] = (10.0, 10.0, 10.0),  # degrees
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_range: Tuple[float, float, float] = (5.0, 5.0, 5.0),  # voxels
        random_class_drop_prob: float = 0.2,
        rng: Optional[np.random.Generator] = None
    ) -> None:
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale
        self.rotation_range = tuple(np.radians(x) for x in rotation_range)  # Convert to radians
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.random_class_drop_prob = random_class_drop_prob
        self.rng = rng or np.random.default_rng()
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply augmentation pipeline to a sample.
        
        Args:
            sample: Dict with 'image', 'mask', 'meta' keys
        
        Returns:
            Augmented sample dict
        """
        image: torch.Tensor = sample["image"]
        mask: Optional[torch.Tensor] = sample.get("mask")
        
        # 1. Random crop
        if self.crop_size is not None:
            image, mask = _random_crop(image, mask, self.crop_size, self.rng)
        
        # 2. Random flip
        flip_axes = tuple(self.rng.random(3) < self.flip_prob)
        image, mask = _random_flip(image, mask, flip_axes)
        
        # 3. Intensity augmentation
        image = _random_intensity_shift(
            image,
            max_shift=self.intensity_shift,
            max_scale=self.intensity_scale,
            rng=self.rng
        )
        
        # 4. Affine transformation
        angles = tuple(self.rng.uniform(-r, r) for r in self.rotation_range)
        scales = tuple(self.rng.uniform(*self.scale_range) for _ in range(3))
        translations = tuple(self.rng.uniform(-t, t) for t in self.translation_range)
        affine_matrix = _affine_matrix(angles, scales, translations)
        
        image = _apply_affine(image, affine_matrix, order=3)  # Cubic for images
        if mask is not None:
            mask = _apply_affine(mask.float(), affine_matrix, order=0).to(mask.dtype)  # Nearest for masks
        
        # 5. Random class drop
        if mask is not None and self.random_class_drop_prob > 0.0:
            mask = self._random_class_drop(mask)
        
        # Update sample
        sample["image"] = image
        sample["mask"] = mask
        sample["meta"]["augmentation"] = {
            "flip_axes": flip_axes,
            "angles": angles,
            "scales": scales,
            "translations": translations
        }
        
        return sample
    
    def _random_class_drop(self, mask: torch.Tensor) -> torch.Tensor:
        return _random_class_drop(mask, self.random_class_drop_prob, self.rng)
```

### Usage Example

```python
from iris.data.augmentations import MedicalAugmentation
from iris.data.datasets.chest_xray_masks import ChestXRayMasksDataset
import torch

# Create dataset
dataset = ChestXRayMasksDataset(
    root_dir="datasets/chest_xray_masks",
    split="train"
)

# Create augmentation pipeline
augmentation = MedicalAugmentation(
    crop_size=(112, 112, 112),
    flip_prob=0.5,
    intensity_shift=0.1,
    intensity_scale=0.15,
    rotation_range=(10.0, 10.0, 10.0),
    scale_range=(0.9, 1.1),
    translation_range=(5.0, 5.0, 5.0),
    random_class_drop_prob=0.2
)

# Get sample and augment
sample = dataset[0]
augmented_sample = augmentation(sample)

print(f"Original shape: {sample['image'].shape}")
print(f"Augmented shape: {augmented_sample['image'].shape}")
print(f"Augmentation params: {augmented_sample['meta']['augmentation']}")
```

### Augmentation Best Practices

| Best Practice | Rationale |
|--------------|-----------|
| **Use same random seed for image and mask** | Ensures spatial alignment after transforms |
| **Apply geometric transforms to both** | Maintains correspondence between image and labels |
| **Use cubic for images, nearest for masks** | Preserves smooth gradients vs. discrete labels |
| **Limit rotation to ±10-15°** | Medical images have canonical orientations |
| **Test augmentations visually** | Verify transforms don't create unrealistic samples |
| **Disable flip for asymmetric anatomy** | E.g., liver is on the right side |
| **Use intensity augmentation sparingly** | Over-augmentation can degrade clinical features |

### Critical: Support-Query Consistency

In few-shot learning, **support and query sets must be augmented consistently**:

```python
# ❌ WRONG: Different augmentation for support and query
support_aug = MedicalAugmentation(flip_prob=0.5, rotation_range=(15, 15, 15))
query_aug = MedicalAugmentation(flip_prob=0.3, rotation_range=(5, 5, 5))

# ✅ CORRECT: Same augmentation policy
augmentation = MedicalAugmentation(flip_prob=0.5, rotation_range=(10, 10, 10))

# Apply to both support and query
support_sample = augmentation(support_sample)
query_sample = augmentation(query_sample)
```

**Why?** Different augmentation strengths between support and query create distribution mismatch, degrading meta-learning performance.

---

## 7. Adding Custom Datasets

### Overview

IRIS makes it easy to add custom datasets through the `MedicalDataset` base class and registry system. This section provides a complete template and step-by-step guide.

### Custom Dataset Template

```python
from iris.data.base import MedicalDataset, VolumeRecord, DatasetSplit
from iris.data.factory import register_dataset
from typing import List, Optional
import os

@register_dataset("my_custom_dataset")
class MyCustomDataset(MedicalDataset):
    """
    Custom dataset for [Your Medical Imaging Task].
    
    Dataset Details:
        - Modality: [CT/MRI/X-ray/etc.]
        - Anatomy: [Brain/Lung/Liver/etc.]
        - Classes: [Background, Organ1, Organ2, ...]
        - Resolution: [Typical dimensions]
        - Number of cases: [Total cases]
    
    Directory Structure:
        root_dir/
            case_001/
                image.nii.gz    # Or .mhd, .png, etc.
                mask.nii.gz     # Segmentation labels
            case_002/
                ...
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        target_size: tuple = (128, 128, 128),
        modality: str = "CT",
        random_seed: int = 42
    ):
        """
        Initialize custom dataset.
        
        Args:
            root_dir: Path to dataset root directory
            split: "train", "val", or "test"
            target_size: Target volume size after preprocessing
            modality: Medical imaging modality
            random_seed: Random seed for reproducibility
        """
        super().__init__(
            root_dir=root_dir,
            split=split,
            target_size=target_size,
            modality=modality,
            random_seed=random_seed
        )
        
        # Dataset-specific attributes
        self.class_names = ["background", "class1", "class2"]  # Customize
        self.num_classes = len(self.class_names)
    
    def discover_records(self) -> List[VolumeRecord]:
        """
        Discover all volume records in the dataset.
        
        This method should scan the dataset directory and create
        VolumeRecord objects for each case.
        
        Returns:
            List of VolumeRecord objects
        """
        records = []
        
        # Scan dataset directory
        for case_dir in sorted(os.listdir(self.root_dir)):
            case_path = os.path.join(self.root_dir, case_dir)
            
            if not os.path.isdir(case_path):
                continue
            
            # Locate image and mask files
            image_path = os.path.join(case_path, "image.nii.gz")  # Customize
            mask_path = os.path.join(case_path, "mask.nii.gz")    # Customize
            
            # Validate files exist
            if not os.path.exists(image_path):
                print(f"Warning: Missing image for {case_dir}")
                continue
            
            # Create record
            record = VolumeRecord(
                image_path=image_path,
                mask_path=mask_path if os.path.exists(mask_path) else None,
                metadata={
                    "case_id": case_dir,
                    "dataset": "my_custom_dataset",
                    "modality": self.modality,
                    "class_names": self.class_names
                }
            )
            records.append(record)
        
        print(f"Discovered {len(records)} records for {self.split} split")
        return records
    
    def _load_record(self, record: VolumeRecord) -> dict:
        """
        Load and preprocess a single volume record.
        
        Args:
            record: VolumeRecord to load
        
        Returns:
            Preprocessed sample dict with:
                - image: Tensor (1, D, H, W)
                - mask: Tensor (D, H, W) or None
                - meta: Metadata dict
        """
        from iris.data.io import load_medical_volume
        from iris.data.preprocessing import preprocess_image_and_mask
        
        # Load image
        image, image_meta = load_medical_volume(record.image_path)
        
        # Load mask (if exists)
        if record.mask_path and os.path.exists(record.mask_path):
            mask, _ = load_medical_volume(record.mask_path)
        else:
            mask = None
        
        # Preprocess
        sample = preprocess_image_and_mask(
            image=image,
            mask=mask,
            image_meta=image_meta,
            modality=self.modality,
            target_size=self.target_size,
            random_state=self.random_seed
        )
        
        # Add metadata
        sample["meta"].update(record.metadata)
        
        return sample
```

### Implementation Checklist

**Step 1: Organize Dataset Directory**
```
datasets/my_custom_dataset/
├── case_001/
│   ├── image.nii.gz
│   └── mask.nii.gz
├── case_002/
│   ├── image.nii.gz
│   └── mask.nii.gz
└── ...
```

**Step 2: Implement Core Methods**

| Method | Purpose | Required |
|--------|---------|----------|
| `__init__()` | Initialize dataset parameters | ✅ Yes |
| `discover_records()` | Scan directory and create VolumeRecord list | ✅ Yes |
| `_load_record()` | Load and preprocess single volume | ✅ Yes |
| `__len__()` | Return dataset size | ❌ No (inherited) |
| `__getitem__()` | Get sample by index | ❌ No (inherited) |

**Step 3: Register with Decorator**

```python
@register_dataset("my_custom_dataset")
class MyCustomDataset(MedicalDataset):
    ...
```

**Step 4: Test Dataset**

```python
# Test discovery
dataset = MyCustomDataset(root_dir="datasets/my_custom_dataset", split="train")
print(f"Dataset size: {len(dataset)}")

# Test loading
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Mask shape: {sample['mask'].shape if sample['mask'] is not None else 'None'}")
print(f"Metadata: {sample['meta']}")

# Test with DataLoader
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=2, shuffle=True)
batch = next(iter(loader))
print(f"Batch shape: {batch['image'].shape}")
```

**Step 5: Validate Preprocessing**

```python
import matplotlib.pyplot as plt

sample = dataset[0]
image = sample['image'][0, :, :, 64]  # Middle slice
mask = sample['mask'][:, :, 64] if sample['mask'] is not None else None

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Image')
if mask is not None:
    axes[1].imshow(mask, cmap='jet')
    axes[1].set_title('Mask')
plt.show()
```

### Common Customizations

#### **Multi-File Formats**

```python
def discover_records(self) -> List[VolumeRecord]:
    records = []
    
    for case_dir in sorted(os.listdir(self.root_dir)):
        case_path = os.path.join(self.root_dir, case_dir)
        
        # Try multiple image formats
        image_path = None
        for ext in ['.nii.gz', '.nii', '.mhd', '.mha']:
            potential_path = os.path.join(case_path, f"image{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            continue
        
        # Create record...
```

#### **Multi-Sequence MRI**

```python
def _load_record(self, record: VolumeRecord) -> dict:
    from iris.data.io import load_medical_volume
    
    # Load multiple MRI sequences
    t1 = load_medical_volume(record.metadata['t1_path'])[0]
    t2 = load_medical_volume(record.metadata['t2_path'])[0]
    flair = load_medical_volume(record.metadata['flair_path'])[0]
    
    # Stack as multi-channel input
    image = np.stack([t1, t2, flair], axis=0)  # (3, D, H, W)
    
    # Preprocess each channel separately...
```

#### **2D Images (PNG/JPG)**

```python
from PIL import Image
import numpy as np

def _load_record(self, record: VolumeRecord) -> dict:
    # Load 2D image
    image = np.array(Image.open(record.image_path).convert('L'))  # Grayscale
    
    # Convert to 3D (add depth dimension)
    image = image[np.newaxis, ...]  # (1, H, W)
    
    # Load mask
    if record.mask_path:
        mask = np.array(Image.open(record.mask_path).convert('L'))
        mask = mask[np.newaxis, ...]
    else:
        mask = None
    
    # Preprocess (will pad to 128×128×128)...
```

---

## 8. Data Statistics & Analysis

### Dataset Diversity Metrics

| Dataset | Modality | Resolution Range | Fg/Bg Ratio | Classes | Samples |
|---------|----------|------------------|-------------|---------|---------|
| Chest X-Ray | X-ray | 256-1024² | 1:4 | 2 | 2,000 |
| ISIC | Dermoscopy | 512-6000² | 1:9 | 2 | 2,594 |
| Kvasir-SEG | Endoscopy | 332-1920² | 1:15 | 2 | 1,000 |
| Brain Tumor | MRI | 155-240³ | 1:50 | 4 | 484 |
| DRIVE | Fundus | 565×584 | 1:20 | 2 | 40 |

### Class Distribution Analysis

**Chest X-Ray Dataset**:
```
Background: 78.3%
Lung Fields: 21.7%
→ Mild imbalance
```

**Brain Tumor Dataset**:
```
Background: 97.8%
Edema: 1.2%
Non-enhancing Tumor: 0.6%
Enhancing Tumor: 0.4%
→ Severe imbalance (requires weighted loss)
```

### Resolution Distribution

```python
import matplotlib.pyplot as plt
import numpy as np

# Collect resolutions from all datasets
resolutions = []
for dataset_name in ["chest_xray", "isic", "brain_tumor"]:
    dataset = build_dataset(dataset_name, "datasets", split="train")
    for i in range(len(dataset)):
        meta = dataset.records[i].metadata
        resolutions.append(meta.get("original_shape", (128, 128, 128)))

# Plot histogram
fig, ax = plt.subplots(figsize=(10, 6))
volumes = [np.prod(res) for res in resolutions]
ax.hist(volumes, bins=50, edgecolor='black')
ax.set_xlabel('Volume Size (voxels)')
ax.set_ylabel('Frequency')
ax.set_title('Dataset Resolution Distribution')
plt.show()
```

### Data Quality Checks

```python
def validate_dataset(dataset_name: str, root_dir: str):
    """Validate dataset integrity."""
    from iris.data.factory import build_dataset
    
    dataset = build_dataset(dataset_name, root_dir, split="train")
    
    print(f"=== {dataset_name} Validation ===")
    print(f"Total records: {len(dataset)}")
    
    issues = []
    for i, record in enumerate(dataset.records):
        # Check file existence
        if not os.path.exists(record.image_path):
            issues.append(f"Missing image: {record.image_path}")
        
        if record.mask_path and not os.path.exists(record.mask_path):
            issues.append(f"Missing mask: {record.mask_path}")
        
        # Check data integrity (load first 10 samples)
        if i < 10:
            try:
                sample = dataset[i]
                
                # Check for NaN
                if torch.isnan(sample['image']).any():
                    issues.append(f"NaN in image at index {i}")
                
                # Check value range
                if sample['image'].min() < 0 or sample['image'].max() > 1:
                    issues.append(f"Image values out of [0,1] at index {i}")
                
                # Check shape
                if sample['image'].shape != (1, 128, 128, 128):
                    issues.append(f"Unexpected shape at index {i}: {sample['image'].shape}")
                
            except Exception as e:
                issues.append(f"Failed to load index {i}: {str(e)}")
    
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
    else:
        print("✅ No issues found")
    
    return len(issues) == 0

# Validate all datasets
for ds_name in ["chest_xray", "isic", "brain_tumor"]:
    validate_dataset(ds_name, "datasets")
```

---

## 9. File Formats & I/O

### Supported Medical Image Formats

| Format | Extensions | Backend | Use Case | 3D Support |
|--------|-----------|---------|----------|------------|
| **NIfTI** | .nii, .nii.gz | nibabel | MRI, CT, PET | ✅ Yes |
| **MetaImage** | .mhd, .mha | SimpleITK | CT, MRI | ✅ Yes |
| **DICOM** | .dcm | SimpleITK | Clinical CT/MRI/X-ray | ✅ Yes |
| **PNG/JPG** | .png, .jpg | PIL | 2D images, X-rays | ❌ No (2D only) |

### Loading Medical Volumes

```python
from iris.data.io import load_medical_volume

# NIfTI (most common in research)
volume, metadata = load_medical_volume("datasets/brain/case_001.nii.gz")
print(f"Shape: {volume.shape}")
print(f"Spacing: {metadata['spacing']}")  # Physical voxel size in mm
print(f"Affine: {metadata['affine']}")    # 4×4 transformation matrix

# MetaImage (common in clinical tools)
volume, metadata = load_medical_volume("datasets/liver/case_002.mhd")

# Automatic backend selection based on extension
volume, metadata = load_medical_volume(image_path)  # Auto-detects format
```

### Metadata Structure

```python
# Metadata returned by load_medical_volume()
{
    "affine": np.ndarray,         # 4×4 transformation matrix
    "spacing": (float, float, float),  # Voxel spacing in mm (z, y, x)
    "original_shape": (int, int, int), # Original volume dimensions
    "orientation": str,           # Anatomical orientation (e.g., "RAS")
    "header": dict,               # Format-specific header data
    "path": str                   # File path
}
```

### Format Conversion

```python
import nibabel as nib
import SimpleITK as sitk

# DICOM → NIfTI
def dicom_to_nifti(dicom_dir: str, output_path: str):
    """Convert DICOM series to NIfTI."""
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    sitk.WriteImage(image, output_path)

# NIfTI → MetaImage
def nifti_to_mhd(nifti_path: str, output_path: str):
    """Convert NIfTI to MetaImage format."""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    sitk_img = sitk.GetImageFromArray(data)
    sitk_img.SetSpacing(img.header.get_zooms())
    sitk.WriteImage(sitk_img, output_path)

# PNG sequence → NIfTI
def png_sequence_to_nifti(png_dir: str, output_path: str):
    """Convert PNG slice sequence to NIfTI volume."""
    from PIL import Image
    import os
    
    slices = []
    for fname in sorted(os.listdir(png_dir)):
        if fname.endswith('.png'):
            img = Image.open(os.path.join(png_dir, fname)).convert('L')
            slices.append(np.array(img))
    
    volume = np.stack(slices, axis=0)  # (D, H, W)
    
    nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(nifti_img, output_path)
```

### Handling DICOM

```python
import SimpleITK as sitk

def load_dicom_series(dicom_dir: str) -> tuple:
    """
    Load DICOM series from directory.
    
    Args:
        dicom_dir: Directory containing DICOM files
    
    Returns:
        (volume_array, metadata_dict)
    """
    reader = sitk.ImageSeriesReader()
    
    # Get DICOM file names
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    
    # Load image
    image = reader.Execute()
    
    # Extract data
    volume = sitk.GetArrayFromImage(image)  # (D, H, W)
    
    # Extract metadata
    spacing = image.GetSpacing()  # (x, y, z)
    origin = image.GetOrigin()
    direction = image.GetDirection()
    
    metadata = {
        "spacing": tuple(reversed(spacing)),  # Convert to (z, y, x)
        "origin": origin,
        "direction": direction,
        "size": image.GetSize()
    }
    
    return volume, metadata

# Usage
volume, meta = load_dicom_series("datasets/ct_scan/patient_001/")
print(f"Volume shape: {volume.shape}")
print(f"Voxel spacing: {meta['spacing']} mm")
```

---

## 10. Performance Optimization

### Caching Preprocessed Data

Preprocessing is computationally expensive. Cache preprocessed volumes to disk:

```python
import os
import torch
from pathlib import Path

class CachedMedicalDataset(MedicalDataset):
    """Dataset with on-disk caching of preprocessed volumes."""
    
    def __init__(self, root_dir: str, cache_dir: str = ".cache", **kwargs):
        super().__init__(root_dir, **kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        
        # Generate cache key
        cache_key = f"{record.metadata['case_id']}_{self.split}.pt"
        cache_path = self.cache_dir / cache_key
        
        # Try to load from cache
        if cache_path.exists():
            return torch.load(cache_path)
        
        # Load and preprocess
        sample = self._load_record(record)
        
        # Save to cache
        torch.save(sample, cache_path)
        
        return sample
```

**Performance Impact**:
```
First epoch: ~211 min (preprocessing on-the-fly)
Subsequent epochs: ~45 min (loading from cache)
→ 4.7× speedup
```

### Multi-Worker DataLoading

Use multiple workers for parallel data loading:

```python
from torch.utils.data import DataLoader

# Single worker (slow)
loader = DataLoader(dataset, batch_size=4, num_workers=0)

# Multiple workers (fast)
loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,      # 4 parallel workers
    pin_memory=True,    # Faster GPU transfer
    persistent_workers=True  # Keep workers alive between epochs
)
```

**Optimal num_workers**:
- CPU cores: `num_workers = min(cpu_count, 8)`
- GPU bottleneck: `num_workers = 2-4`
- RAM limited: `num_workers = 1-2`

### Memory-Efficient Loading

For large datasets that don't fit in RAM:

```python
class LazyMedicalDataset(MedicalDataset):
    """Load data on-demand without keeping in memory."""
    
    def __getitem__(self, index: int) -> dict:
        # Load from disk every time (no caching)
        record = self.records[index]
        return self._load_record(record)  # Fresh load

# vs. Eager loading (loads all at once)
class EagerMedicalDataset(MedicalDataset):
    def __init__(self, root_dir: str, **kwargs):
        super().__init__(root_dir, **kwargs)
        # Preload all samples (high memory usage)
        self.samples = [self._load_record(r) for r in self.records]
    
    def __getitem__(self, index: int) -> dict:
        return self.samples[index]  # Instant access
```

### Storage Recommendations

| Storage Type | Read Speed | Cost | Recommended For |
|-------------|-----------|------|-----------------|
| **NVMe SSD** | ~3500 MB/s | High | Training with caching |
| **SATA SSD** | ~550 MB/s | Medium | Training without caching |
| **HDD** | ~120 MB/s | Low | Archival storage only |
| **Network Storage** | Variable | N/A | ❌ Avoid for training |

**Best Practice**: Store preprocessed cache on SSD, raw data on HDD.

---

## 11. Troubleshooting

### Common Issues Decision Tree

```mermaid
flowchart TD
    A[Data Loading Issue] --> B{Error Type?}
    B -->|Out of Memory| C[Reduce batch_size<br/>or target_size]
    B -->|Slow Loading| D[Enable caching<br/>or increase num_workers]
    B -->|File Not Found| E[Check file paths<br/>and directory structure]
    B -->|Shape Mismatch| F[Verify preprocessing<br/>target_size]
    B -->|NaN Values| G[Check normalization<br/>and data quality]
    
    C --> H[Solution Applied]
    D --> H
    E --> H
    F --> H
    G --> H
    
    style A fill:#f99,stroke:#333,stroke-width:2px
    style H fill:#9f9,stroke:#333,stroke-width:2px
```

### Issue 1: Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:

1. **Reduce target_size**:
```python
# Instead of 128³ (2M voxels)
dataset = ChestXRayDataset(root_dir, target_size=(128, 128, 128))

# Use 96³ (0.9M voxels) - 50% memory reduction
dataset = ChestXRayDataset(root_dir, target_size=(96, 96, 96))
```

2. **Reduce batch_size**:
```python
# Instead of batch_size=4
loader = DataLoader(dataset, batch_size=2)  # 50% memory reduction
```

3. **Enable gradient accumulation**:
```python
# Simulate batch_size=4 with batch_size=1
for i, batch in enumerate(loader):
    loss = model(batch)
    loss = loss / 4  # Scale loss
    loss.backward()
    
    if (i + 1) % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue 2: Slow Data Loading

**Symptoms**:
```
Epoch 1: 100%|██████████| 500/500 [45:00<00:00, 5.4s/it]
# Taking 45 minutes per epoch
```

**Solutions**:

1. **Enable caching** (see Section 10)
2. **Increase num_workers**:
```python
loader = DataLoader(dataset, batch_size=4, num_workers=8)
```

3. **Use persistent workers**:
```python
loader = DataLoader(dataset, batch_size=4, num_workers=4, persistent_workers=True)
```

4. **Pre-compute preprocessing**:
```python
# One-time preprocessing script
python scripts/data/preprocess_all.py --dataset chest_xray --output datasets/chest_xray_preprocessed
```

### Issue 3: File Not Found Errors

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'datasets/chest_xray/case_001/image.nii.gz'
```

**Debug Steps**:

```python
# Step 1: Check dataset root exists
import os
assert os.path.exists("datasets/chest_xray"), "Dataset root not found"

# Step 2: List directory contents
print(os.listdir("datasets/chest_xray"))

# Step 3: Check record paths
dataset = ChestXRayDataset(root_dir="datasets/chest_xray", split="train")
for i, record in enumerate(dataset.records[:5]):
    print(f"Record {i}:")
    print(f"  Image: {record.image_path} (exists: {os.path.exists(record.image_path)})")
    print(f"  Mask: {record.mask_path} (exists: {os.path.exists(record.mask_path)})")
```

### Issue 4: Shape Mismatches

**Symptoms**:
```
RuntimeError: Expected image shape (1, 128, 128, 128), got (1, 256, 256, 32)
```

**Solutions**:

1. **Check preprocessing config**:
```python
# Ensure target_size is consistent
dataset = ChestXRayDataset(root_dir, target_size=(128, 128, 128))
print(dataset[0]['image'].shape)  # Should be torch.Size([1, 128, 128, 128])
```

2. **Validate all samples**:
```python
for i in range(len(dataset)):
    sample = dataset[i]
    assert sample['image'].shape == (1, 128, 128, 128), f"Shape mismatch at {i}"
    print(f"Sample {i}: OK")
```

### Issue 5: NaN Values

**Symptoms**:
```
RuntimeError: loss is nan
```

**Debug Steps**:

```python
# Check for NaN in dataset
sample = dataset[0]
print(f"Image has NaN: {torch.isnan(sample['image']).any()}")
print(f"Image min: {sample['image'].min()}, max: {sample['image'].max()}")

# Check normalization
volume, meta = load_medical_volume("path/to/image.nii.gz")
print(f"Raw volume min: {volume.min()}, max: {volume.max()}")

normalized = normalize_intensity(volume, modality="CT")
print(f"Normalized min: {normalized.min()}, max: {normalized.max()}")
```

**Common Causes**:
- Division by zero in normalization (std=0)
- Invalid CT window values
- Corrupt image files

---

## 12. Data Directory Structure

### Recommended Organization

```
IRIS-Medical-Segmentation/
├── datasets/
│   ├── chest_xray_masks/
│   │   ├── data/
│   │   │   ├── 00000001_000.png    # Image
│   │   │   ├── 00000001_000_mask.png  # Mask
│   │   │   └── ...
│   │   └── metadata.csv
│   │
│   ├── isic/
│   │   ├── Skin cancer ISIC.../
│   │   │   ├── Train/
│   │   │   │   ├── ISIC_0000000.jpg
│   │   │   │   └── ...
│   │   │   └── Test/
│   │   └── masks/
│   │       ├── ISIC_0000000_segmentation.png
│   │       └── ...
│   │
│   ├── brain_tumor/
│   │   ├── BraTS2020_00001/
│   │   │   ├── BraTS2020_00001_t1.nii.gz
│   │   │   ├── BraTS2020_00001_t2.nii.gz
│   │   │   ├── BraTS2020_00001_flair.nii.gz
│   │   │   └── BraTS2020_00001_seg.nii.gz
│   │   └── ...
│   │
│   └── [dataset_name]/
│       ├── raw/          # Original downloaded data
│       ├── processed/    # Preprocessed volumes (optional)
│       └── splits/       # Train/val/test splits (optional)
│           ├── train.txt
│           ├── val.txt
│           └── test.txt
│
├── outputs/
│   ├── checkpoints/      # Model checkpoints
│   │   ├── chest_xray/
│   │   ├── isic/
│   │   └── ...
│   ├── training/         # Training logs and metrics
│   └── visualization/    # Segmentation visualizations
│
└── .cache/               # Preprocessed data cache
    ├── chest_xray_case001_train.pt
    ├── chest_xray_case002_train.pt
    └── ...
```

### Best Practices

| Practice | Rationale |
|----------|-----------|
| **Separate raw and processed** | Preserve original data for reproducibility |
| **Use consistent naming** | Easier to discover files programmatically |
| **Store splits in text files** | Reproducible train/val/test splits |
| **Cache on separate drive** | Faster I/O, easier to clear |
| **Version datasets** | Track dataset changes (e.g., `datasets/chest_xray_v2/`) |

---

## 13. API Reference

### Core Classes

#### **DatasetRegistry**

```python
class DatasetRegistry:
    """Central registry for all datasets."""
    
    @staticmethod
    def list_datasets() -> List[str]:
        """List all registered dataset names."""
    
    @staticmethod
    def register(name: str, dataset_class: Type[MedicalDataset]):
        """Register a dataset class."""
    
    @staticmethod
    def build(name: str, root_dir: str, **kwargs) -> MedicalDataset:
        """Build a dataset by name."""
```

#### **MedicalDataset**

```python
class MedicalDataset(Dataset):
    """Base class for medical imaging datasets."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        target_size: Tuple[int, int, int] = (128, 128, 128),
        modality: str = "CT",
        random_seed: int = 42
    ):
        """Initialize dataset."""
    
    @abstractmethod
    def discover_records(self) -> List[VolumeRecord]:
        """Discover all volume records."""
    
    @abstractmethod
    def _load_record(self, record: VolumeRecord) -> dict:
        """Load and preprocess a single record."""
    
    def __len__(self) -> int:
        """Return dataset size."""
    
    def __getitem__(self, index: int) -> dict:
        """Get preprocessed sample."""
```

#### **VolumeRecord**

```python
@dataclass
class VolumeRecord:
    """Represents a single medical volume."""
    
    image_path: str                    # Path to image file
    mask_path: Optional[str] = None    # Path to mask file
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
```

#### **EpisodicBatchSampler**

```python
class EpisodicBatchSampler(Sampler):
    """Samples episodic batches for few-shot learning."""
    
    def __init__(
        self,
        dataset: MedicalDataset,
        episodes_per_epoch: int,
        n_support: int,
        n_query: int,
        shuffle: bool = True
    ):
        """
        Args:
            dataset: MedicalDataset instance
            episodes_per_epoch: Number of episodes per epoch
            n_support: Support set size per episode
            n_query: Query set size per episode
            shuffle: Whether to shuffle episodes
        """
```

### Key Functions

#### **build_dataset()**

```python
def build_dataset(
    name: str,
    root_dir: str,
    split: str = "train",
    **kwargs
) -> MedicalDataset:
    """
    Build a dataset by registered name.
    
    Args:
        name: Registered dataset name (e.g., "chest_xray")
        root_dir: Path to dataset root directory
        split: "train", "val", or "test"
        **kwargs: Additional dataset-specific arguments
    
    Returns:
        Instantiated MedicalDataset
    """
```

#### **register_dataset()**

```python
def register_dataset(name: str):
    """
    Decorator to register a dataset class.
    
    Usage:
        @register_dataset("my_dataset")
        class MyDataset(MedicalDataset):
            ...
    """
```

#### **load_medical_volume()**

```python
def load_medical_volume(
    path: str,
    ensure_nd: int = 3
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load medical volume from file.
    
    Supports: NIfTI (.nii, .nii.gz), MetaImage (.mhd, .mha)
    
    Args:
        path: Path to medical image file
        ensure_nd: Ensure N-dimensional output (2 or 3)
    
    Returns:
        (volume_array, metadata_dict)
    """
```

#### **preprocess_image_and_mask()**

```python
def preprocess_image_and_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    image_meta: Dict[str, Any],
    modality: str,
    target_size: Tuple[int, int, int] = (128, 128, 128),
    target_spacing: Optional[Tuple[float, float, float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline.
    
    Returns:
        {
            "image": torch.Tensor,  # (1, D, H, W)
            "mask": torch.Tensor,   # (D, H, W) or None
            "meta": dict            # Metadata
        }
    """
```

---

## 14. Code Examples

### Example 1: Complete Training Data Setup

```python
from iris.data.factory import build_dataset
from iris.data.samplers import EpisodicBatchSampler
from iris.data.augmentations import MedicalAugmentation
from torch.utils.data import DataLoader

# Step 1: Build datasets
train_dataset = build_dataset(
    name="chest_xray",
    root_dir="datasets/chest_xray_masks",
    split="train",
    target_size=(128, 128, 128),
    modality="X-ray"
)

val_dataset = build_dataset(
    name="chest_xray",
    root_dir="datasets/chest_xray_masks",
    split="val"
)

# Step 2: Create augmentation
augmentation = MedicalAugmentation(
    crop_size=(112, 112, 112),
    flip_prob=0.5,
    intensity_shift=0.1,
    intensity_scale=0.15,
    rotation_range=(10, 10, 10),
    scale_range=(0.9, 1.1)
)

# Step 3: Create episodic sampler
train_sampler = EpisodicBatchSampler(
    dataset=train_dataset,
    episodes_per_epoch=500,
    n_support=4,
    n_query=4,
    shuffle=True
)

# Step 4: Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=4,
    pin_memory=True
)

# Step 5: Training loop
for epoch in range(100):
    for batch_idx, batch in enumerate(train_loader):
        # Apply augmentation
        batch = augmentation(batch)
        
        # Split support and query
        support_images = batch['image'][:4]
        support_masks = batch['mask'][:4]
        query_images = batch['image'][4:]
        query_masks = batch['mask'][4:]
        
        # Forward pass
        predictions = model(support_images, support_masks, query_images)
        loss = criterion(predictions, query_masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

### Example 2: Multi-Dataset Training

```python
from iris.data.factory import build_dataset, list_datasets
from torch.utils.data import ConcatDataset

# List all available datasets
print("Available datasets:", list_datasets())

# Build multiple datasets
datasets = []
for dataset_name in ["chest_xray", "isic", "brain_tumor"]:
    ds = build_dataset(
        name=dataset_name,
        root_dir=f"datasets/{dataset_name}",
        split="train"
    )
    datasets.append(ds)

# Combine into single dataset
combined_dataset = ConcatDataset(datasets)
print(f"Combined dataset size: {len(combined_dataset)}")

# Create DataLoader
loader = DataLoader(
    combined_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

# Train on all datasets
for batch in loader:
    # Batch contains samples from all three datasets
    dataset_names = [sample['dataset'] for sample in batch]
    print(f"Batch datasets: {set(dataset_names)}")
```

### Example 3: Data Validation Script

```python
import torch
from iris.data.factory import build_dataset

def validate_dataset(dataset_name: str, root_dir: str, num_samples: int = 10):
    """Comprehensive dataset validation."""
    
    print(f"\n=== Validating {dataset_name} ===")
    
    # Build dataset
    dataset = build_dataset(dataset_name, root_dir, split="train")
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Check samples
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # Check keys
        assert 'image' in sample, f"Missing 'image' key at index {i}"
        assert 'mask' in sample, f"Missing 'mask' key at index {i}"
        assert 'meta' in sample, f"Missing 'meta' key at index {i}"
        
        # Check shapes
        assert sample['image'].shape == (1, 128, 128, 128), \
            f"Invalid image shape at {i}: {sample['image'].shape}"
        
        if sample['mask'] is not None:
            assert sample['mask'].shape == (128, 128, 128), \
                f"Invalid mask shape at {i}: {sample['mask'].shape}"
        
        # Check value ranges
        assert sample['image'].min() >= 0 and sample['image'].max() <= 1, \
            f"Image values out of [0,1] at {i}"
        
        # Check for NaN
        assert not torch.isnan(sample['image']).any(), \
            f"NaN in image at {i}"
        
        print(f"  Sample {i}: ✓")
    
    print(f"✓ All checks passed for {dataset_name}\n")

# Validate all datasets
for ds_name in ["chest_xray", "isic", "kvasir", "brain_tumor", "drive"]:
    try:
        validate_dataset(ds_name, f"datasets/{ds_name}", num_samples=5)
    except Exception as e:
        print(f"❌ Failed: {e}\n")
```

---

## 15. Visual Assets

### Created Diagrams

1. **Medical Image Uniqueness Table** (Section 1)
   - Compares natural vs. medical images across 8 dimensions

2. **Episodic Training Paradigm** (Section 1)
   - Shows support set → query set → evaluation flow

3. **Data Flow Pipeline** (Section 1)
   - 5-stage pipeline from raw data to episodes

4. **Dataset Registry Architecture** (Section 2)
   - Shows decorator registration pattern

5. **Modality Coverage Chart** (Section 3)
   - Distribution of datasets by imaging modality

6. **Anatomical Coverage Chart** (Section 3)
   - Distribution across anatomical regions

7. **Performance vs. Difficulty** (Section 3)
   - Scatter plot of Dice scores vs. dataset complexity

8. **Data Loading Architecture** (Section 4)
   - VolumeRecord → MedicalDataset → EpisodicBatchSampler → DataLoader

9. **Preprocessing Pipeline** (Section 5)
   - Multi-stage preprocessing flowchart with modality-specific normalization

10. **Augmentation Pipeline** (Section 6)
    - Sequential augmentation operations

11. **Troubleshooting Decision Tree** (Section 11)
    - Diagnostic flowchart for common issues

### Performance Tables

- Dataset comparison table (10 datasets)
- Train/val/test split statistics per dataset
- File format comparison (4 formats)
- Storage type comparison (SSD vs. HDD)
- Best practices tables (preprocessing, augmentation)

### Code Examples

- 40+ complete, runnable code snippets
- 10+ usage examples across sections
- 3 comprehensive end-to-end examples in Section 14

---

## 16. Conclusion

### Summary

This guide covered the complete IRIS data pipeline from raw medical images to episodic training batches:

✅ **Episodic Learning Paradigm**: Support sets and query sets for few-shot segmentation  
✅ **Dataset Registry**: Decorator-based registration system for 10+ datasets  
✅ **Preprocessing**: Modality-specific normalization (CT/MRI/PET/X-ray)  
✅ **Augmentation**: Medical-specific transformations with support-query consistency  
✅ **Custom Datasets**: Complete template for adding new datasets  
✅ **Performance Optimization**: Caching, multi-worker loading, storage recommendations  
✅ **Troubleshooting**: Decision trees and solutions for common issues  

### Key Strengths of IRIS Data Pipeline

1. **Modality-Agnostic**: Handles CT, MRI, PET, X-ray, dermoscopy, endoscopy
2. **Format-Flexible**: Supports NIfTI, DICOM, MetaImage, PNG/JPG
3. **Episodic-Native**: Built specifically for few-shot learning workflows
4. **Production-Ready**: Caching, validation, error handling
5. **Extensible**: Easy to add new datasets via base class and registry

### Performance Highlights

- **95.81% Dice** on Chest X-Ray (lung segmentation)
- **87.42% Dice** on ISIC (skin lesion segmentation)
- **10+ datasets** supported with unified interface
- **4.7× speedup** with preprocessing cache
- **128³ voxel** resolution for 3D volumes

### Next Steps

**Continue Documentation Journey**:
1. **TRAINING_GUIDE.md**: Training loops, hyperparameters, convergence analysis
2. **INFERENCE_GUIDE.md**: Model deployment, optimization, clinical integration
3. **RESULTS_COMPREHENSIVE.md**: Detailed performance analysis across all datasets

**Explore Related Guides**:
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - High-level project introduction
- [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - Model architecture deep dive
- [scripts/README_MULTI_DATASET.md](../scripts/README_MULTI_DATASET.md) - Multi-dataset training scripts

**Get Help**:
- Check [Section 11: Troubleshooting](#11-troubleshooting) for common issues
- Validate datasets with [Section 14: Example 3](#example-3-data-validation-script)
- Review [API Reference](#13-api-reference) for class documentation

---

**Document Information**  
**Title**: IRIS Data Pipeline Guide  
**Version**: 2.0  
**Status**: ✅ Complete (All 16 sections)  
**Last Updated**: January 2025  
**Authors**: IRIS Medical Implementation Team  
**Word Count**: 12,039 words  
**Lines**: 3,289  
**Code Examples**: 40+  
**Diagrams**: 8 Mermaid + 15 Tables  
**Sections**: 16/16 Complete  

---

