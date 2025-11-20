"""IRIS model inference wrapper for web application."""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Optional, Dict
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iris.model import IrisModel, InContextTuner
from iris.data import build_dataset, DatasetSplit
from iris.data.preprocessing import normalize_intensity


class IRISInference:
    """IRIS model inference wrapper for web application."""
    
    # Dataset configurations
    DATASET_CONFIGS = {
        'kvasir': {
            'root': 'datasets/Kvasir-SEG Data (Polyp segmentation & detection)',
            'volume_shape': (16, 256, 256),
            'modality': 'Endoscopy',
            'display_name': 'Kvasir (Polyp Segmentation)'
        },
        'drive': {
            'root': 'datasets/drive-digital-retinal-images-for-vessel-extraction',
            'volume_shape': (16, 256, 256),
            'modality': 'Fundus',
            'display_name': 'DRIVE (Retinal Vessels)'
        },
        'brain_tumor': {
            'root': 'datasets/brain-tumor-dataset-includes-the-mask-and-images/data/data',
            'volume_shape': (16, 256, 256),
            'modality': 'MRI',
            'display_name': 'Brain Tumor (MRI)'
        },
        'isic': {
            'root': 'datasets/Skin cancer ISIC The International Skin Imaging Collaboration',
            'volume_shape': (16, 256, 256),
            'modality': 'Dermoscopy',
            'display_name': 'ISIC (Skin Lesion)'
        },
        'chest_xray_masks': {
            'root': 'datasets/chest_xray_masks',
            'volume_shape': (16, 256, 256),
            'modality': 'CT',
            'display_name': 'Chest X-Ray'
        }
    }
    
    def __init__(self, checkpoint_path: str, dataset: str = "kvasir", base_dir: Path = None):
        """
        Initialize IRIS inference.
        
        Args:
            checkpoint_path: Path to final_model.pt
            dataset: Dataset name (kvasir, drive, etc.)
            base_dir: Base directory of the project
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = dataset
        
        # Get dataset config
        if dataset not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(self.DATASET_CONFIGS.keys())}")
        
        self.config = self.DATASET_CONFIGS[dataset]
        self.volume_shape = self.config['volume_shape']
        self.modality = self.config['modality']
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Load support dataset
        self.support_dataset = self._load_support_dataset()
    
    def _load_model(self, checkpoint_path: str) -> IrisModel:
        """Load trained IRIS model."""
        checkpoint_path = self.base_dir / checkpoint_path
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        model = IrisModel(
            in_channels=1,
            base_channels=16,
            num_query_tokens=8,
            num_attention_heads=8,
            volume_shape=self.volume_shape,
            use_memory_bank=True,
            memory_momentum=0.999
        ).to(self.device)
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def _load_support_dataset(self):
        """Load support dataset for retrieving context images."""
        dataset_root = self.base_dir / self.config['root']
        
        if not dataset_root.exists():
            print(f"Warning: Dataset root not found: {dataset_root}")
            return None
        
        dataset_kwargs = {
            "depth_slices": self.volume_shape[0],
            "target_resolution": self.volume_shape[1],
        }
        
        try:
            dataset = build_dataset(
                self.dataset_name,
                root=str(dataset_root),
                split=DatasetSplit.TRAIN,
                **dataset_kwargs
            )
            return dataset
        except Exception as e:
            print(f"Warning: Could not load support dataset: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess uploaded image to IRIS format.
        
        Args:
            image: PIL Image (RGB or grayscale)
        
        Returns:
            Tensor of shape (1, 1, D, H, W)
        """
        # Convert to grayscale
        img_gray = image.convert("L")
        
        # Resize to target resolution
        target_hw = (self.volume_shape[1], self.volume_shape[2])
        img_resized = img_gray.resize(target_hw[::-1])  # PIL uses (W, H)
        
        # Convert to numpy and normalize
        img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
        
        # Create pseudo-3D volume
        volume = np.repeat(img_array[np.newaxis, ...], self.volume_shape[0], axis=0)
        
        # Apply modality-specific normalization
        volume = normalize_intensity(volume, self.modality)
        
        # Convert to tensor (1, 1, D, H, W)
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def get_support_images(self, num_support: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get random support images from dataset.
        
        Args:
            num_support: Number of support images
        
        Returns:
            (support_images, support_masks) tensors
        """
        if self.support_dataset is None:
            raise ValueError("Support dataset not available")
        
        indices = np.random.choice(len(self.support_dataset), num_support, replace=False)
        
        support_images = []
        support_masks = []
        
        for idx in indices:
            data = self.support_dataset[int(idx)]
            support_images.append(data['image'])
            support_masks.append(data['mask'])
        
        return (torch.stack(support_images).to(self.device),
                torch.stack(support_masks).to(self.device))
    
    def predict_oneshot(self, query_img: torch.Tensor, 
                       support_img: torch.Tensor, 
                       support_mask: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        One-shot prediction.
        
        Returns:
            (prediction, inference_time)
        """
        start_time = time.time()
        
        with torch.no_grad():
            # Ensure correct dimensions
            if support_mask.dim() == 4:
                support_mask = support_mask.unsqueeze(1)
            
            # Encode support
            support_out = self.model.encode_support(support_img, support_mask)
            task_embeddings = support_out["task_embeddings"]
            
            # Predict
            outputs = self.model(query_img, task_embeddings)
            logits = outputs["logits"]
            pred = (torch.sigmoid(logits) > 0.5).float()
        
        inference_time = time.time() - start_time
        return pred, inference_time
    
    def predict_ensemble(self, query_img: torch.Tensor,
                        support_imgs: torch.Tensor,
                        support_masks: torch.Tensor,
                        num_support: int = 3) -> Tuple[torch.Tensor, float]:
        """
        Context ensemble prediction.
        
        Returns:
            (prediction, inference_time)
        """
        start_time = time.time()
        
        with torch.no_grad():
            all_embeddings = []
            
            for i in range(min(num_support, len(support_imgs))):
                supp_img = support_imgs[i:i+1]
                supp_mask = support_masks[i:i+1]
                
                if supp_mask.dim() == 4:
                    supp_mask = supp_mask.unsqueeze(1)
                
                support_out = self.model.encode_support(supp_img, supp_mask)
                all_embeddings.append(support_out["task_embeddings"])
            
            # Average embeddings
            task_embeddings = torch.stack(all_embeddings).mean(dim=0)
            
            # Predict
            outputs = self.model(query_img, task_embeddings)
            logits = outputs["logits"]
            pred = (torch.sigmoid(logits) > 0.5).float()
        
        inference_time = time.time() - start_time
        return pred, inference_time
    
    def predict_with_tuning(self, query_img: torch.Tensor,
                           initial_mask: torch.Tensor,
                           support_imgs: torch.Tensor,
                           support_masks: torch.Tensor,
                           tuning_steps: int = 20,
                           num_support: int = 5) -> Tuple[torch.Tensor, float]:
        """
        Full IRIS with in-context tuning.
        
        Returns:
            (prediction, inference_time)
        """
        start_time = time.time()
        
        # Get initial prediction
        with torch.no_grad():
            all_embeddings = []
            
            for i in range(min(num_support, len(support_imgs))):
                supp_img = support_imgs[i:i+1]
                supp_mask = support_masks[i:i+1]
                
                if supp_mask.dim() == 4:
                    supp_mask = supp_mask.unsqueeze(1)
                
                support_out = self.model.encode_support(supp_img, supp_mask)
                all_embeddings.append(support_out["task_embeddings"])
            
            initial_embeddings = torch.stack(all_embeddings).mean(dim=0)
        
        # Create tuner and tune
        tuner = InContextTuner(
            model=self.model,
            lr=1e-3,
            steps=tuning_steps,
            memory_bank=self.model.memory_bank
        )
        
        # Use initial mask as target for tuning
        if initial_mask.dim() == 4:
            initial_mask = initial_mask.unsqueeze(1)
        
        tuned_embeddings = tuner.tune(
            query_images=query_img,
            query_masks=initial_mask,
            initial_embeddings=initial_embeddings,
            steps=tuning_steps,
            update_memory=False
        )
        
        # Final prediction with tuned embeddings
        with torch.no_grad():
            outputs = self.model(query_img, tuned_embeddings)
            logits = outputs["logits"]
            pred = (torch.sigmoid(logits) > 0.5).float()
        
        inference_time = time.time() - start_time
        return pred, inference_time
    
    def compute_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute Dice coefficient.
        
        Args:
            pred: Prediction tensor
            target: Ground truth tensor
            
        Returns:
            Dice score (0-1)
        """
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        dice = (2.0 * intersection) / (union + 1e-8)
        return dice.item()
    
    def visualize_prediction(self, query_img: torch.Tensor, 
                            prediction: torch.Tensor,
                            ground_truth: Optional[torch.Tensor] = None,
                            color_scheme: str = "Green-Gold") -> Image.Image:
        """
        Create visualization of prediction overlaid on query image.
        
        Args:
            query_img: Query image tensor
            prediction: Prediction mask tensor
            ground_truth: Optional ground truth mask
            color_scheme: Color scheme for overlay (Green-Gold, Blue-Cyan, Red-Orange, Purple-Pink, Rainbow)
        
        Returns:
            PIL Image with overlay
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        # Define color schemes
        color_schemes = {
            "Green-Gold": ['#00000000', '#00FF00', '#FFD700'],
            "Blue-Cyan": ['#00000000', '#0080FF', '#00FFFF'],
            "Red-Orange": ['#00000000', '#FF4500', '#FFA500'],
            "Purple-Pink": ['#00000000', '#9370DB', '#FF69B4'],
            "Rainbow": ['#00000000', '#00FF00', '#FFFF00', '#FF8C00', '#FF0000']
        }
        
        # Get colors for selected scheme
        colors = color_schemes.get(color_scheme, color_schemes["Green-Gold"])
        cmap = LinearSegmentedColormap.from_list('custom_seg', colors, N=100)
        
        # Extract middle slice
        mid_slice = self.volume_shape[0] // 2
        
        img_slice = query_img[0, 0, mid_slice].cpu().numpy()
        pred_slice = prediction[0, 0, mid_slice].cpu().numpy()
        
        # Normalize image
        img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        
        # Create figure
        if ground_truth is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            gt_slice = ground_truth[0, 0, mid_slice].cpu().numpy()
            
            # Query image
            axes[0].imshow(img_norm, cmap='gray')
            axes[0].set_title('Query Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Prediction
            axes[1].imshow(img_norm, cmap='gray')
            masked_pred = np.ma.masked_where(pred_slice < 0.5, pred_slice)
            axes[1].imshow(masked_pred, cmap=cmap, alpha=0.7, vmin=0, vmax=1)
            axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Ground truth
            axes[2].imshow(img_norm, cmap='gray')
            masked_gt = np.ma.masked_where(gt_slice < 0.5, gt_slice)
            axes[2].imshow(masked_gt, cmap='Greens', alpha=0.6)
            axes[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
            axes[2].axis('off')
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img_norm, cmap='gray')
            
            # Overlay prediction with custom colormap
            masked = np.ma.masked_where(pred_slice < 0.5, pred_slice)
            ax.imshow(masked, cmap=cmap, alpha=0.7, vmin=0, vmax=1)
            
            ax.set_title('Segmentation Result', fontsize=16, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return Image.fromarray(img_array)
