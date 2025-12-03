#!/usr/bin/env python3
"""
Fine-tuning and Inference Pipeline for Foundation Models

This script provides:
1. Fine-tuning capability for foundation models on the full dataset
2. Average Dice score computation during and after training
3. Pure inference mode (no training)
4. Integration with ASOCA + ImageCAS dataset

Usage:
    # Fine-tuning mode
    python train_foundation_models.py --mode train --model medsam --epochs 10
    
    # Inference only mode
    python train_foundation_models.py --mode inference --model medsam --checkpoint best_model.pth
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, Literal
import matplotlib.pyplot as plt

# Import foundation models
from foundation_models_pipeline import (
    SAMModel, 
    MedSAMModel,
    cleanup_memory
)

# Import segmentation metrics
from segmentation import DiceLoss


class FoundationModelTrainer:
    """
    Trainer class for fine-tuning foundation models on medical image segmentation.
    """
    
    def __init__(
        self,
        model_type: Literal["sam", "medsam"] = "medsam",
        device: str = "cuda",
        learning_rate: float = 1e-4,
        output_dir: str = "checkpoints"
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of foundation model ("sam" or "medsam")
            device: Device to use for training
            learning_rate: Learning rate for optimizer
            output_dir: Directory to save checkpoints
        """
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize model
        print(f"\n{'='*60}")
        print(f"Initializing {model_type.upper()} model for training")
        print(f"{'='*60}")
        
        if model_type == "sam":
            self.model_wrapper = SAMModel(device=self.device)
        elif model_type == "medsam":
            self.model_wrapper = MedSAMModel(device=self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_wrapper.load()
        self.model = self.model_wrapper.model
        
        # Set model to training mode
        self.model.train()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Initialize loss functions
        self.dice_loss = DiceLoss(
            smooth=1,
            class_reduction='foreground mean',
            batch_reduction='mean',
            which='loss'
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Training metrics
        self.train_losses = []
        self.train_dice_scores = []
        self.val_losses = []
        self.val_dice_scores = []
        
        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Optimizer: AdamW (lr={learning_rate})")
        print(f"✓ Loss functions: Dice + BCE")
    
    def compute_dice_score(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
        """
        Compute Dice score between prediction and ground truth.
        
        Args:
            pred_mask: Predicted mask [B, 1, H, W] (logits or probabilities)
            gt_mask: Ground truth mask [B, 1, H, W] (binary)
            
        Returns:
            Dice score (0-1, higher is better)
        """
        # Apply sigmoid if needed
        if pred_mask.min() < 0 or pred_mask.max() > 1:
            pred_mask = torch.sigmoid(pred_mask)
        
        # Binarize prediction
        pred_binary = (pred_mask > 0.5).float()
        
        # Compute Dice score
        intersection = (pred_binary * gt_mask).sum()
        union = pred_binary.sum() + gt_mask.sum()
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        
        return dice.item()
    
    def train_step(self, image: torch.Tensor, gt_mask: torch.Tensor, boxes: Optional[list] = None) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            image: Input image [1, H, W]
            gt_mask: Ground truth mask [1, H, W]
            boxes: Optional bounding boxes for prompting
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Convert to PIL for processing
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        
        # Handle grayscale images
        if image.shape[0] == 1:
            image_rgb = torch.cat([image, image, image], dim=0)
        else:
            image_rgb = image
        
        pil_image = to_pil(image_rgb)
        
        # Prepare inputs with bounding boxes
        if boxes is None:
            H, W = image.shape[-2:]
            boxes = [[[0, 0, W-1, H-1]]]
        
        inputs = self.model_wrapper.processor(
            pil_image,
            input_boxes=[boxes],
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs, multimask_output=False)
        pred_masks = outputs.pred_masks  # [B, 1, H, W, ...]
        
        # Get the first mask
        if pred_masks.ndim == 5:
            pred_masks = pred_masks[:, 0]  # [B, H, W, ...]
        
        # Resize ground truth to match prediction size
        gt_mask_batch = gt_mask.unsqueeze(0).float()  # [1, 1, H, W]
        if gt_mask_batch.shape[-2:] != pred_masks.shape[-2:]:
            gt_mask_batch = F.interpolate(
                gt_mask_batch,
                size=pred_masks.shape[-2:],
                mode='nearest'
            )
        
        # Prepare masks for loss computation
        pred_masks_flat = pred_masks.view(-1, *pred_masks.shape[-2:]).unsqueeze(1)  # [B, 1, H, W]
        gt_mask_flat = gt_mask_batch.view(-1, *gt_mask_batch.shape[-2:])  # [B, 1, H, W]
        gt_mask_long = gt_mask_flat.long()  # For Dice loss
        
        # Compute losses
        bce = self.bce_loss(pred_masks_flat, gt_mask_flat)
        dice = self.dice_loss(
            torch.cat([1 - torch.sigmoid(pred_masks_flat), torch.sigmoid(pred_masks_flat)], dim=1),
            gt_mask_long,
            x_is_probabilities=True
        )
        
        # Combined loss
        loss = bce + dice
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute Dice score for metrics
        dice_score = self.compute_dice_score(pred_masks_flat, gt_mask_flat)
        
        return {
            "loss": loss.item(),
            "bce": bce.item(),
            "dice_loss": dice.item(),
            "dice_score": dice_score
        }
    
    def validate(self, dataset) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            dataset: Validation dataset
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        num_samples = 0
        
        print("\n🔍 Running validation...")
        
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Validating"):
                # Get validation sample
                val_data = dataset._get_val_test()
                image = val_data['image']  # [S, 1, H, W]
                label = val_data['label']  # [S, 1, H, W]
                
                if label is None:
                    continue
                
                # Process a few slices
                num_slices = min(5, image.shape[0])
                slice_indices = np.linspace(0, image.shape[0]-1, num_slices, dtype=int)
                
                for s_idx in slice_indices:
                    img_slice = image[s_idx]  # [1, H, W]
                    gt_slice = label[s_idx]  # [1, H, W]
                    
                    # Skip if no foreground
                    if gt_slice.sum() < 10:
                        continue
                    
                    # Predict
                    try:
                        pred_mask = self.model_wrapper.predict(img_slice, boxes=None)
                        
                        # Resize to match ground truth
                        if pred_mask.shape[-2:] != gt_slice.shape[-2:]:
                            pred_mask = F.interpolate(
                                pred_mask.unsqueeze(0),
                                size=gt_slice.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)
                        
                        # Compute dice score
                        dice_score = self.compute_dice_score(
                            pred_mask.unsqueeze(0),
                            gt_slice.unsqueeze(0)
                        )
                        
                        total_dice += dice_score
                        num_samples += 1
                        
                    except Exception as e:
                        print(f"Warning: Error processing slice {s_idx}: {e}")
                        continue
        
        avg_dice = total_dice / max(num_samples, 1)
        
        return {
            "val_dice": avg_dice,
            "num_samples": num_samples
        }
    
    def train(
        self,
        train_dataset,
        val_dataset=None,
        epochs: int = 10,
        samples_per_epoch: int = 100,
        val_every: int = 1,
        save_every: int = 1
    ):
        """
        Train the foundation model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            epochs: Number of epochs
            samples_per_epoch: Number of samples per epoch
            val_every: Validate every N epochs
            save_every: Save checkpoint every N epochs
        """
        print(f"\n{'🚀'*30}")
        print("STARTING FINE-TUNING")
        print(f"{'🚀'*30}")
        print(f"Model: {self.model_type.upper()}")
        print(f"Epochs: {epochs}")
        print(f"Samples per epoch: {samples_per_epoch}")
        print(f"Device: {self.device}\n")
        
        best_val_dice = 0.0
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Training
            self.model.train()
            epoch_losses = []
            epoch_dice_scores = []
            
            pbar = tqdm(range(samples_per_epoch), desc="Training")
            for step in pbar:
                try:
                    # Get a batch from dataset
                    img_batch, lab_batch = train_dataset.get(minibatch_size=1, out_side=256)
                    
                    # Train on first sample
                    img = img_batch[0]  # [1, H, W]
                    gt = lab_batch[0]  # [1, H, W]
                    
                    # Skip if no foreground
                    if gt.sum() < 10:
                        continue
                    
                    # Compute bounding box from ground truth
                    nonzero = torch.nonzero(gt[0] > 0)
                    if len(nonzero) > 0:
                        y_min, x_min = nonzero.min(dim=0)[0].tolist()
                        y_max, x_max = nonzero.max(dim=0)[0].tolist()
                        # Add margin
                        margin = 10
                        H, W = gt.shape[-2:]
                        boxes = [[
                            max(0, x_min - margin),
                            max(0, y_min - margin),
                            min(W-1, x_max + margin),
                            min(H-1, y_max + margin)
                        ]]
                    else:
                        boxes = None
                    
                    # Training step
                    metrics = self.train_step(img, gt, boxes=boxes)
                    
                    epoch_losses.append(metrics['loss'])
                    epoch_dice_scores.append(metrics['dice_score'])
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'dice': f"{metrics['dice_score']:.4f}"
                    })
                    
                except Exception as e:
                    print(f"\nWarning: Error in training step {step}: {e}")
                    continue
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            avg_dice = np.mean(epoch_dice_scores) if epoch_dice_scores else 0.0
            
            self.train_losses.append(avg_loss)
            self.train_dice_scores.append(avg_dice)
            
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {avg_loss:.4f}")
            print(f"   Train Dice: {avg_dice:.4f}")
            
            # Validation
            if val_dataset is not None and (epoch + 1) % val_every == 0:
                val_metrics = self.validate(val_dataset)
                val_dice = val_metrics['val_dice']
                
                self.val_dice_scores.append(val_dice)
                
                print(f"   Val Dice:   {val_dice:.4f} (on {val_metrics['num_samples']} samples)")
                
                # Save best model
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    self.save_checkpoint("best_model.pth", epoch, best_val_dice)
                    print(f"   💾 New best model saved! (Dice: {best_val_dice:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", epoch, avg_dice)
        
        print(f"\n{'✅'*30}")
        print("FINE-TUNING COMPLETED")
        print(f"{'✅'*30}")
        print(f"\nFinal Training Dice: {self.train_dice_scores[-1]:.4f}")
        if self.val_dice_scores:
            print(f"Best Validation Dice: {max(self.val_dice_scores):.4f}")
        
        return {
            "train_losses": self.train_losses,
            "train_dice_scores": self.train_dice_scores,
            "val_dice_scores": self.val_dice_scores,
            "best_val_dice": best_val_dice
        }
    
    def save_checkpoint(self, filename: str, epoch: int, dice_score: float):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dice_score': dice_score,
        }, checkpoint_path)
        print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"\n📦 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Checkpoint loaded (Epoch: {checkpoint.get('epoch', 'N/A')}, Dice: {checkpoint.get('dice_score', 'N/A'):.4f})")
    
    def inference(self, dataset, save_results: bool = True) -> Dict[str, Any]:
        """
        Run pure inference on dataset without training.
        
        Args:
            dataset: Dataset to run inference on
            save_results: Whether to save prediction results
            
        Returns:
            Dictionary with inference metrics and predictions
        """
        print(f"\n{'🔍'*30}")
        print("RUNNING INFERENCE")
        print(f"{'🔍'*30}")
        print(f"Model: {self.model_type.upper()}")
        print(f"Mode: Inference only (no training)")
        print(f"Device: {self.device}\n")
        
        self.model.eval()
        
        all_dice_scores = []
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Inference"):
                # Get sample
                data = dataset._get_val_test()
                image = data['image']  # [S, 1, H, W]
                label = data['label']  # [S, 1, H, W] or None
                sample_id = data['id']
                
                # Process middle slice
                mid_slice = image.shape[0] // 2
                img_slice = image[mid_slice]  # [1, H, W]
                
                # Predict
                try:
                    pred_mask = self.model_wrapper.predict(img_slice, boxes=None)
                    
                    # Compute Dice if label available
                    if label is not None:
                        gt_slice = label[mid_slice]  # [1, H, W]
                        
                        # Resize to match
                        if pred_mask.shape[-2:] != gt_slice.shape[-2:]:
                            pred_mask_resized = F.interpolate(
                                pred_mask.unsqueeze(0),
                                size=gt_slice.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)
                        else:
                            pred_mask_resized = pred_mask
                        
                        dice_score = self.compute_dice_score(
                            pred_mask_resized.unsqueeze(0),
                            gt_slice.unsqueeze(0)
                        )
                        all_dice_scores.append(dice_score)
                    else:
                        dice_score = None
                    
                    predictions.append({
                        'id': sample_id,
                        'prediction': pred_mask.cpu().numpy(),
                        'dice': dice_score
                    })
                    
                except Exception as e:
                    print(f"\nWarning: Error processing sample {i}: {e}")
                    continue
        
        # Compute average Dice
        avg_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0
        
        print(f"\n{'📊'*30}")
        print("INFERENCE RESULTS")
        print(f"{'📊'*30}")
        print(f"Number of samples: {len(predictions)}")
        if all_dice_scores:
            print(f"Average Dice Score: {avg_dice:.4f}")
            print(f"Std Dice Score: {np.std(all_dice_scores):.4f}")
            print(f"Min Dice Score: {np.min(all_dice_scores):.4f}")
            print(f"Max Dice Score: {np.max(all_dice_scores):.4f}")
        
        # Save results
        if save_results:
            results_dir = Path("inference_results")
            results_dir.mkdir(exist_ok=True)
            results_path = results_dir / f"inference_{self.model_type}.pth"
            torch.save({
                'predictions': predictions,
                'avg_dice': avg_dice,
                'all_dice_scores': all_dice_scores
            }, results_path)
            print(f"\n💾 Results saved to: {results_path}")
        
        return {
            'predictions': predictions,
            'avg_dice': avg_dice,
            'all_dice_scores': all_dice_scores
        }
    
    def cleanup(self):
        """Clean up model and free memory."""
        self.model_wrapper.unload()
        cleanup_memory()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fine-tune or run inference with foundation models"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'inference'],
        default='train',
        help='Mode: train (fine-tuning) or inference (no training)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['sam', 'medsam'],
        default='medsam',
        help='Foundation model to use'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--samples-per-epoch',
        type=int,
        default=100,
        help='Number of samples per epoch'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for inference or resume training'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='checkpoints',
        help='Output directory for checkpoints'
    )
    
    args = parser.parse_args()
    
    # Import dataset
    print("\nLoading dataset...")
    
    try:
        # Try to import the actual dataset
        from dataset_asoca_cas import DatasetMerged_2d
        
        train_dataset = DatasetMerged_2d(split='train', img_side=256)
        val_dataset = DatasetMerged_2d(split='val', img_side=256)
        test_dataset = DatasetMerged_2d(split='test', img_side=256)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nPlease ensure the dataset code is available and properly configured.")
        sys.exit(1)
    
    # Initialize trainer
    trainer = FoundationModelTrainer(
        model_type=args.model,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    try:
        if args.mode == 'train':
            # Fine-tuning mode
            results = trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=args.epochs,
                samples_per_epoch=args.samples_per_epoch,
                val_every=1,
                save_every=2
            )
            
            # Plot training curves
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(results['train_losses'], label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(results['train_dice_scores'], label='Train Dice')
            if results['val_dice_scores']:
                plt.plot(results['val_dice_scores'], label='Val Dice')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Score')
            plt.title('Dice Score')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plot_path = Path(args.output_dir) / 'training_curves.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\n📈 Training curves saved to: {plot_path}")
            
        else:
            # Inference mode
            results = trainer.inference(
                dataset=test_dataset,
                save_results=True
            )
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during {args.mode}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        trainer.cleanup()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()
