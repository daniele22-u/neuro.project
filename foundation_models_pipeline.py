"""
Foundation Models Segmentation Pipeline for Medical Images

This script provides a GPU-based pipeline that runs multiple foundation models
sequentially for medical image segmentation. After each model completes,
GPU memory is freed before loading the next model.

Supported Models:
- SAM (Segment Anything Model)
- MedSAM (Medical SAM)
- CLIP (for feature extraction)
- Generic Vision Transformers (ViT)
"""

import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt


def cleanup_memory():
    """Clean up GPU and CPU memory."""
    print("🧹 Cleaning up memory...")
    
    # Garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    
    print("✓ Memory cleanup complete")


class BaseSegmentationModel:
    """Base class for segmentation models."""
    
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
    def load(self):
        """Load model and processor."""
        raise NotImplementedError
        
    def predict(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run prediction on image."""
        raise NotImplementedError
        
    def unload(self):
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        cleanup_memory()


class SAMModel(BaseSegmentationModel):
    """Segment Anything Model (SAM)."""
    
    def __init__(self, device="cuda", model_type="vit_b"):
        super().__init__(device)
        self.model_type = model_type
        
    def load(self):
        """Load SAM model."""
        print("📦 Loading SAM model...")
        try:
            from transformers import SamModel, SamProcessor
            
            # Use facebook/sam-vit-base as default
            model_name = "facebook/sam-vit-base"
            
            self.processor = SamProcessor.from_pretrained(model_name)
            self.model = SamModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            print(f"✓ SAM model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Error loading SAM: {e}")
            raise
            
    def predict(self, image: torch.Tensor, boxes: Optional[list] = None, 
                points: Optional[list] = None, use_automatic_prompts: bool = True) -> torch.Tensor:
        """
        Predict segmentation mask for image.
        
        Args:
            image: [1, H, W] or [3, H, W] tensor
            boxes: Optional list of bounding boxes [[x0, y0, x1, y1], ...]
                   Should ONLY be provided during training/validation.
                   For test mode, set to None to avoid data leakage.
            points: Optional list of point prompts
            use_automatic_prompts: If True and no boxes/points provided,
                   will use entire image as prompt. If False, will try
                   to generate prompts automatically from the image.
            
        Returns:
            Binary mask tensor [1, H, W]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        # Convert to PIL Image
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        
        # Handle grayscale images
        if image.shape[0] == 1:
            image_rgb = torch.cat([image, image, image], dim=0)
        else:
            image_rgb = image
            
        pil_image = to_pil(image_rgb)
        
        # Prepare inputs
        if boxes is not None:
            # WARNING: boxes should only be used during training/validation
            # For test mode, boxes should be None to avoid data leakage
            inputs = self.processor(
                images=pil_image,
                input_boxes=[boxes],
                return_tensors="pt"
            ).to(self.device)
        elif points is not None:
            inputs = self.processor(
                images=pil_image,
                input_points=[points],
                return_tensors="pt"
            ).to(self.device)
        else:
            # Test mode: no ground truth information available
            if use_automatic_prompts:
                # Default: segment entire image (no data leakage)
                H, W = image.shape[-2:]
                boxes = [[[0, 0, W-1, H-1]]]
                inputs = self.processor(
                    images=pil_image,
                    input_boxes=boxes,
                    return_tensors="pt"
                ).to(self.device)
            else:
                # Could add automatic prompt generation here
                # (e.g., grid of points, edge detection, etc.)
                H, W = image.shape[-2:]
                boxes = [[[0, 0, W-1, H-1]]]
                inputs = self.processor(
                    images=pil_image,
                    input_boxes=boxes,
                    return_tensors="pt"
                ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        
        # Post-process masks
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Return first mask
        if len(masks) > 0:
            mask = masks[0]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            
            # Ensure correct shape [1, H, W]
            if mask.ndim == 4:
                mask = mask[0, 0]
            elif mask.ndim == 3:
                mask = mask[0]
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.zeros_like(image[:1])
            
        return mask


class MedSAMModel(BaseSegmentationModel):
    """Medical SAM Model."""
    
    def load(self):
        """Load MedSAM model."""
        print("📦 Loading MedSAM model...")
        try:
            from transformers import SamModel, SamProcessor
            
            model_name = "flaviagiammarino/medsam-vit-base"
            
            self.processor = SamProcessor.from_pretrained(model_name)
            self.model = SamModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            print(f"✓ MedSAM model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Error loading MedSAM: {e}")
            raise
            
    def predict(self, image: torch.Tensor, boxes: Optional[list] = None, 
                use_automatic_prompts: bool = True) -> torch.Tensor:
        """
        Predict segmentation mask for medical image.
        
        Args:
            image: [1, H, W] or [3, H, W] tensor
            boxes: Optional list of bounding boxes [[x0, y0, x1, y1], ...]
                   Should ONLY be provided during training/validation.
                   For test mode, set to None to avoid data leakage.
            use_automatic_prompts: If True and no boxes provided,
                   will use entire image as prompt.
            
        Returns:
            Binary mask tensor [1, H, W]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        
        # Handle grayscale images
        if image.shape[0] == 1:
            image_rgb = torch.cat([image, image, image], dim=0)
        else:
            image_rgb = image
            
        pil_image = to_pil(image_rgb)
        
        # Prepare inputs with box prompt
        if boxes is None:
            # Test mode: no ground truth information
            # Use entire image as prompt (no data leakage)
            H, W = image.shape[-2:]
            boxes = [[[0, 0, W-1, H-1]]]
        else:
            # WARNING: boxes should only be used during training/validation
            # For test mode, boxes should be None to avoid data leakage
            pass
            
        inputs = self.processor(
            pil_image,
            input_boxes=[boxes],
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        
        # Post-process masks
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Return first mask
        if len(masks) > 0:
            mask = masks[0]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            
            # Ensure correct shape [1, H, W]
            if mask.ndim == 4:
                mask = mask[0, 0]
            elif mask.ndim == 3:
                mask = mask[0]
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.zeros_like(image[:1])
            
        return mask


class CLIPViTModel(BaseSegmentationModel):
    """CLIP + ViT for feature extraction and segmentation."""
    
    def __init__(self, device="cuda", model_name="openai/clip-vit-base-patch32"):
        super().__init__(device)
        self.model_name = model_name
        
    def load(self):
        """Load CLIP model."""
        print("📦 Loading CLIP model...")
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
            print(f"✓ CLIP model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Error loading CLIP: {e}")
            raise
            
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract image features using CLIP.
        
        Args:
            image: [1, H, W] or [3, H, W] tensor
            
        Returns:
            Feature vector
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        
        # Handle grayscale images
        if image.shape[0] == 1:
            image_rgb = torch.cat([image, image, image], dim=0)
        else:
            image_rgb = image
            
        pil_image = to_pil(image_rgb)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Extract features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        return image_features
    
    def predict(self, image: torch.Tensor, text_prompt: str = "medical image") -> Dict[str, Any]:
        """
        Analyze image with CLIP using text prompt.
        
        Args:
            image: [1, H, W] or [3, H, W] tensor
            text_prompt: Text description for comparison
            
        Returns:
            Dictionary with features and similarity scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        
        # Handle grayscale images
        if image.shape[0] == 1:
            image_rgb = torch.cat([image, image, image], dim=0)
        else:
            image_rgb = image
            
        pil_image = to_pil(image_rgb)
        
        # Process inputs
        inputs = self.processor(
            text=[text_prompt],
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get features and similarity
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Compute similarity
            similarity = F.cosine_similarity(image_features, text_features)
            
        return {
            "image_features": image_features.cpu(),
            "text_features": text_features.cpu(),
            "similarity": similarity.cpu().item()
        }


class GenericViTModel(BaseSegmentationModel):
    """Generic Vision Transformer for segmentation."""
    
    def __init__(self, device="cuda", model_name="google/vit-base-patch16-224"):
        super().__init__(device)
        self.model_name = model_name
        
    def load(self):
        """Load ViT model."""
        print("📦 Loading ViT model...")
        try:
            from transformers import ViTModel, ViTImageProcessor
            
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
            print(f"✓ ViT model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Error loading ViT: {e}")
            raise
            
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract features using ViT.
        
        Args:
            image: [1, H, W] or [3, H, W] tensor
            
        Returns:
            Feature tensor
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        
        # Handle grayscale images
        if image.shape[0] == 1:
            image_rgb = torch.cat([image, image, image], dim=0)
        else:
            image_rgb = image
            
        pil_image = to_pil(image_rgb)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state
            
        return features
    
    def predict(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features and pooled representation.
        
        Args:
            image: [1, H, W] or [3, H, W] tensor
            
        Returns:
            Dictionary with features
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        
        # Handle grayscale images
        if image.shape[0] == 1:
            image_rgb = torch.cat([image, image, image], dim=0)
        else:
            image_rgb = image
            
        pil_image = to_pil(image_rgb)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return {
            "last_hidden_state": outputs.last_hidden_state.cpu(),
            "pooler_output": outputs.pooler_output.cpu() if hasattr(outputs, "pooler_output") else None
        }


class FoundationModelsPipeline:
    """Pipeline to run multiple foundation models sequentially."""
    
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.results = {}
        
    def run_sam(self, image: torch.Tensor, boxes: Optional[list] = None) -> torch.Tensor:
        """Run SAM model."""
        print("\n" + "="*60)
        print("Running SAM Model")
        print("="*60)
        
        model = SAMModel(device=self.device)
        model.load()
        
        try:
            mask = model.predict(image, boxes=boxes)
            self.results["sam"] = mask
            print(f"✓ SAM prediction complete. Mask shape: {mask.shape}")
        finally:
            model.unload()
            
        return mask
    
    def run_medsam(self, image: torch.Tensor, boxes: Optional[list] = None) -> torch.Tensor:
        """Run MedSAM model."""
        print("\n" + "="*60)
        print("Running MedSAM Model")
        print("="*60)
        
        model = MedSAMModel(device=self.device)
        model.load()
        
        try:
            mask = model.predict(image, boxes=boxes)
            self.results["medsam"] = mask
            print(f"✓ MedSAM prediction complete. Mask shape: {mask.shape}")
        finally:
            model.unload()
            
        return mask
    
    def run_clip(self, image: torch.Tensor, text_prompt: str = "medical image") -> Dict[str, Any]:
        """Run CLIP model."""
        print("\n" + "="*60)
        print("Running CLIP Model")
        print("="*60)
        
        model = CLIPViTModel(device=self.device)
        model.load()
        
        try:
            result = model.predict(image, text_prompt=text_prompt)
            self.results["clip"] = result
            print(f"✓ CLIP analysis complete. Similarity: {result['similarity']:.4f}")
        finally:
            model.unload()
            
        return result
    
    def run_vit(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run generic ViT model."""
        print("\n" + "="*60)
        print("Running Generic ViT Model")
        print("="*60)
        
        model = GenericViTModel(device=self.device)
        model.load()
        
        try:
            features = model.predict(image)
            self.results["vit"] = features
            print(f"✓ ViT feature extraction complete.")
            if features["last_hidden_state"] is not None:
                print(f"  Hidden state shape: {features['last_hidden_state'].shape}")
            if features["pooler_output"] is not None:
                print(f"  Pooler output shape: {features['pooler_output'].shape}")
        finally:
            model.unload()
            
        return features
    
    def run_all(self, image: torch.Tensor, boxes: Optional[list] = None, 
                text_prompt: str = "medical image", mode: str = "test") -> Dict[str, Any]:
        """
        Run all foundation models sequentially.
        
        Args:
            image: Input image tensor [C, H, W]
            boxes: Optional bounding boxes for SAM models.
                   Should ONLY be provided when mode="train" or mode="val".
                   For mode="test", boxes will be ignored to avoid data leakage.
            text_prompt: Text prompt for CLIP
            mode: Execution mode - "train", "val", or "test" (default: "test")
                  In "test" mode, boxes are always set to None to prevent data leakage.
            
        Returns:
            Dictionary with all results
        """
        print("\n" + "🚀"*30)
        print("FOUNDATION MODELS PIPELINE - RUN ALL")
        print("🚀"*30)
        print(f"Device: {self.device}")
        print(f"Image shape: {image.shape}")
        print(f"Mode: {mode.upper()}")
        
        # DATA LEAKAGE PREVENTION: In test mode, ignore boxes
        if mode == "test":
            if boxes is not None:
                print("⚠️  WARNING: Boxes provided in test mode. Setting to None to prevent data leakage.")
            boxes = None
            print("✓ Test mode: Using no ground truth information (boxes=None)")
        else:
            if boxes is not None:
                print(f"✓ {mode.capitalize()} mode: Using provided bounding boxes")
            else:
                print(f"✓ {mode.capitalize()} mode: No boxes provided, using full image")
        
        # Clear previous results
        self.results = {}
        
        # Run each model sequentially with memory cleanup
        self.run_sam(image, boxes=boxes)
        self.run_medsam(image, boxes=boxes)
        self.run_clip(image, text_prompt=text_prompt)
        self.run_vit(image)
        
        print("\n" + "✅"*30)
        print("ALL MODELS COMPLETED SUCCESSFULLY")
        print("✅"*30)
        
        return self.results
    
    def visualize_results(self, original_image: torch.Tensor, save_path: Optional[str] = None):
        """Visualize segmentation results."""
        if "sam" not in self.results and "medsam" not in self.results:
            print("No segmentation results to visualize.")
            return
            
        # Prepare original image for display
        if original_image.shape[0] == 1:
            img_display = original_image[0].cpu().numpy()
        else:
            img_display = original_image.permute(1, 2, 0).cpu().numpy()
            
        # Create figure
        n_plots = 1 + sum([k in self.results for k in ["sam", "medsam"]])
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        if n_plots == 1:
            axes = [axes]
            
        # Plot original image
        idx = 0
        if img_display.ndim == 2:
            axes[idx].imshow(img_display, cmap="gray")
        else:
            axes[idx].imshow(img_display)
        axes[idx].set_title("Original Image")
        axes[idx].axis("off")
        idx += 1
        
        # Plot SAM result
        if "sam" in self.results:
            mask = self.results["sam"][0].cpu().numpy()
            axes[idx].imshow(img_display, cmap="gray" if img_display.ndim == 2 else None)
            axes[idx].imshow(mask, alpha=0.5, cmap="Reds")
            axes[idx].set_title("SAM Segmentation")
            axes[idx].axis("off")
            idx += 1
            
        # Plot MedSAM result
        if "medsam" in self.results:
            mask = self.results["medsam"][0].cpu().numpy()
            axes[idx].imshow(img_display, cmap="gray" if img_display.ndim == 2 else None)
            axes[idx].imshow(mask, alpha=0.5, cmap="Blues")
            axes[idx].set_title("MedSAM Segmentation")
            axes[idx].axis("off")
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Visualization saved to {save_path}")
            
        plt.show()


def demo():
    """Demo function to test the pipeline."""
    print("Foundation Models Segmentation Pipeline Demo")
    print("=" * 60)
    
    # Create synthetic test image
    H, W = 256, 256
    test_image = torch.rand(1, H, W)
    
    # Create pipeline
    pipeline = FoundationModelsPipeline()
    
    # Run all models
    results = pipeline.run_all(test_image)
    
    # Visualize
    pipeline.visualize_results(test_image)
    
    return results


if __name__ == "__main__":
    demo()
