#!/usr/bin/env python3
"""
Example: Fine-tuning and Inference with Foundation Models

This script demonstrates:
1. How to fine-tune foundation models on the ASOCA+ImageCAS dataset
2. How to run pure inference without training
3. How to compute and display average Dice scores

Usage examples:

# Fine-tune MedSAM for 10 epochs
python example_finetuning.py --mode train --model medsam --epochs 10

# Fine-tune SAM for 5 epochs
python example_finetuning.py --mode train --model sam --epochs 5

# Run inference only with a trained model
python example_finetuning.py --mode inference --model medsam --checkpoint checkpoints/best_model.pth

# Run inference on test set without prior training (using pretrained weights)
python example_finetuning.py --mode inference --model medsam
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70 + "\n")


def example_finetuning():
    """
    Example 1: Fine-tune a foundation model on the full dataset
    and display average Dice score.
    """
    print_header("EXAMPLE 1: FINE-TUNING FOUNDATION MODEL")
    
    print("This example shows how to:")
    print("  1. Load the ASOCA + ImageCAS dataset")
    print("  2. Fine-tune a foundation model (MedSAM)")
    print("  3. Track Dice scores during training")
    print("  4. Save the best model")
    print()
    
    # Import required modules
    try:
        from dataset_asoca_cas import DatasetMerged_2d
        from train_foundation_models import FoundationModelTrainer
        
        # Load datasets
        print("📦 Loading datasets...")
        train_dataset = DatasetMerged_2d(split='train', img_side=256)
        val_dataset = DatasetMerged_2d(split='val', img_side=256)
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val: {len(val_dataset)} samples")
        
        # Initialize trainer
        print("\n🤖 Initializing MedSAM trainer...")
        trainer = FoundationModelTrainer(
            model_type='medsam',
            learning_rate=1e-4,
            output_dir='checkpoints'
        )
        
        # Fine-tune
        print("\n🚀 Starting fine-tuning...")
        results = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,  # Small number for demo
            samples_per_epoch=50,
            val_every=1,
            save_every=1
        )
        
        # Display results
        print_header("TRAINING RESULTS")
        print(f"Final Train Dice: {results['train_dice_scores'][-1]:.4f}")
        if results['val_dice_scores']:
            print(f"Best Val Dice: {max(results['val_dice_scores']):.4f}")
        
        # Cleanup
        trainer.cleanup()
        
        print("\n✅ Fine-tuning example completed!")
        
    except ImportError as e:
        print(f"❌ Error: Could not import required modules: {e}")
        print("   Make sure dataset_asoca_cas.py and train_foundation_models.py are available.")
        return
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return


def example_inference_only():
    """
    Example 2: Run pure inference without training.
    """
    print_header("EXAMPLE 2: PURE INFERENCE (NO TRAINING)")
    
    print("This example shows how to:")
    print("  1. Load a pretrained foundation model")
    print("  2. Run inference on test data")
    print("  3. Compute average Dice score")
    print()
    
    try:
        from dataset_asoca_cas import DatasetMerged_2d
        from train_foundation_models import FoundationModelTrainer
        
        # Load test dataset
        print("📦 Loading test dataset...")
        test_dataset = DatasetMerged_2d(split='test', img_side=256)
        print(f"   Test: {len(test_dataset)} samples")
        
        # Initialize trainer (will load pretrained weights)
        print("\n🤖 Loading pretrained MedSAM...")
        trainer = FoundationModelTrainer(
            model_type='medsam',
            output_dir='inference_results'
        )
        
        # Optional: Load fine-tuned checkpoint if available
        checkpoint_path = 'checkpoints/best_model.pth'
        if os.path.exists(checkpoint_path):
            print(f"   Loading fine-tuned weights from {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)
        else:
            print("   Using pretrained weights (no fine-tuning)")
        
        # Run inference
        print("\n🔍 Running inference...")
        results = trainer.inference(
            dataset=test_dataset,
            save_results=True
        )
        
        # Display results
        print_header("INFERENCE RESULTS")
        print(f"Number of samples: {len(results['predictions'])}")
        if results['all_dice_scores']:
            print(f"Average Dice Score: {results['avg_dice']:.4f}")
            print(f"Std Dice Score: {np.std(results['all_dice_scores']):.4f}")
            print(f"Min Dice Score: {np.min(results['all_dice_scores']):.4f}")
            print(f"Max Dice Score: {np.max(results['all_dice_scores']):.4f}")
        
        # Cleanup
        trainer.cleanup()
        
        print("\n✅ Inference example completed!")
        
    except ImportError as e:
        print(f"❌ Error: Could not import required modules: {e}")
        return
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return


def example_compare_models():
    """
    Example 3: Compare SAM vs MedSAM on inference.
    """
    print_header("EXAMPLE 3: COMPARE SAM vs MEDSAM")
    
    print("This example shows how to:")
    print("  1. Run inference with both SAM and MedSAM")
    print("  2. Compare their Dice scores")
    print()
    
    try:
        from dataset_asoca_cas import DatasetMerged_2d
        from train_foundation_models import FoundationModelTrainer
        
        # Load test dataset
        print("📦 Loading test dataset...")
        test_dataset = DatasetMerged_2d(split='test', img_side=256)
        print(f"   Test: {len(test_dataset)} samples")
        
        results_comparison = {}
        
        for model_type in ['sam', 'medsam']:
            print(f"\n{'='*70}")
            print(f"Testing {model_type.upper()}")
            print(f"{'='*70}")
            
            # Initialize trainer
            trainer = FoundationModelTrainer(
                model_type=model_type,
                output_dir=f'inference_results_{model_type}'
            )
            
            # Run inference (limited samples for demo)
            print(f"\n🔍 Running inference with {model_type.upper()}...")
            
            # For demo, just test on a few samples
            dice_scores = []
            for i in range(min(5, len(test_dataset))):
                try:
                    data = test_dataset._get_val_test()
                    image = data['image']
                    label = data['label']
                    
                    if label is None:
                        continue
                    
                    # Test on middle slice
                    mid = image.shape[0] // 2
                    img_slice = image[mid]
                    gt_slice = label[mid]
                    
                    # Predict
                    pred_mask = trainer.model_wrapper.predict(img_slice, boxes=None)
                    
                    # Compute Dice
                    dice = trainer.compute_dice_score(
                        pred_mask.unsqueeze(0),
                        gt_slice.unsqueeze(0)
                    )
                    dice_scores.append(dice)
                    
                except Exception as e:
                    print(f"   Warning: Error on sample {i}: {e}")
                    continue
            
            avg_dice = np.mean(dice_scores) if dice_scores else 0.0
            results_comparison[model_type] = avg_dice
            
            print(f"   {model_type.upper()} Average Dice: {avg_dice:.4f}")
            
            # Cleanup
            trainer.cleanup()
        
        # Final comparison
        print_header("COMPARISON RESULTS")
        for model, dice in results_comparison.items():
            print(f"{model.upper()}: Dice = {dice:.4f}")
        
        best_model = max(results_comparison, key=results_comparison.get)
        print(f"\n🏆 Best model: {best_model.upper()}")
        
        print("\n✅ Comparison example completed!")
        
    except ImportError as e:
        print(f"❌ Error: Could not import required modules: {e}")
        return
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Examples for fine-tuning and inference with foundation models"
    )
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3],
        default=None,
        help='Example to run: 1=fine-tuning, 2=inference, 3=comparison'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all examples'
    )
    
    args = parser.parse_args()
    
    print_header("FOUNDATION MODELS - FINE-TUNING & INFERENCE EXAMPLES")
    print("This script demonstrates how to use foundation models for medical image segmentation.")
    print()
    print("Available examples:")
    print("  1. Fine-tuning on full dataset with Dice tracking")
    print("  2. Pure inference without training")
    print("  3. Compare SAM vs MedSAM")
    print()
    
    if args.all:
        # Run all examples
        example_finetuning()
        example_inference_only()
        example_compare_models()
    elif args.example == 1:
        example_finetuning()
    elif args.example == 2:
        example_inference_only()
    elif args.example == 3:
        example_compare_models()
    else:
        # Interactive menu
        print("Please select an example to run:")
        print("  1. Fine-tuning")
        print("  2. Inference only")
        print("  3. Comparison")
        print("  4. All examples")
        
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                example_finetuning()
            elif choice == '2':
                example_inference_only()
            elif choice == '3':
                example_compare_models()
            elif choice == '4':
                example_finetuning()
                example_inference_only()
                example_compare_models()
            else:
                print("Invalid choice. Run with --example 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


if __name__ == "__main__":
    main()
