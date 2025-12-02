#!/usr/bin/env python3
"""
Run All Foundation Models - Complete Pipeline

This script executes all foundation models sequentially on medical image data.
It integrates with the existing dataset infrastructure and provides proper
GPU memory management between model executions.

Usage:
    python run_all_foundation_models.py

Features:
- Sequential execution of SAM, MedSAM, CLIP, and ViT models
- Automatic GPU memory cleanup between models
- Results visualization and saving
- Support for both 2D slices and full volumes
"""

import os
import sys
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the foundation models pipeline
from foundation_models_pipeline import (
    FoundationModelsPipeline,
    cleanup_memory
)


def create_sample_data(device="cuda"):
    """
    Create sample medical image data for testing.
    Returns a dictionary with image and optional ground truth.
    """
    print("Creating sample medical image data...")
    
    # Create synthetic medical image (grayscale, 256x256)
    H, W = 256, 256
    
    # Simulate a medical scan with some anatomical structures
    image = torch.zeros(1, H, W, dtype=torch.float32)
    
    # Add some circular structures (simulating vessels or organs)
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    
    # Central structure
    center_y, center_x = H // 2, W // 2
    radius = 60
    dist = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
    image[0, dist < radius] = 0.8
    
    # Add some vessel-like structures
    for i in range(3):
        angle = i * np.pi * 2 / 3
        cx = int(center_x + 40 * np.cos(angle))
        cy = int(center_y + 40 * np.sin(angle))
        dist = torch.sqrt((y - cy)**2 + (x - cx)**2)
        image[0, dist < 20] = 0.6
    
    # Add some noise
    noise = torch.randn(1, H, W) * 0.05
    image = torch.clamp(image + noise, 0, 1)
    
    # Create a bounding box around the central structure
    boxes = [[center_x - radius - 10, center_y - radius - 10, 
              center_x + radius + 10, center_y + radius + 10]]
    
    print(f"✓ Sample data created: image shape {image.shape}")
    
    return {
        'image': image,
        'boxes': boxes,
        'text_prompt': 'medical anatomical structure'
    }


def run_pipeline_on_sample_data():
    """
    Main function to run all foundation models on sample data.
    """
    print("\n" + "="*70)
    print("FOUNDATION MODELS PIPELINE - RUN ALL")
    print("="*70)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create sample data
    data = create_sample_data(device=device)
    image = data['image']
    boxes = data['boxes']
    text_prompt = data['text_prompt']
    
    # Initialize pipeline
    pipeline = FoundationModelsPipeline(device=device)
    
    # Run all models sequentially
    print("\n" + "🚀"*35)
    print("Starting sequential model execution...")
    print("🚀"*35 + "\n")
    
    try:
        results = pipeline.run_all(
            image=image,
            boxes=boxes,
            text_prompt=text_prompt
        )
        
        # Display results summary
        print("\n" + "📊"*35)
        print("RESULTS SUMMARY")
        print("📊"*35)
        
        for model_name, result in results.items():
            print(f"\n{model_name.upper()}:")
            if isinstance(result, torch.Tensor):
                print(f"  Output shape: {result.shape}")
                print(f"  Output dtype: {result.dtype}")
                if result.numel() < 10:
                    print(f"  Values: {result}")
            elif isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key} shape: {value.shape}")
                    else:
                        print(f"  {key}: {value}")
        
        # Visualize results
        print("\n" + "🎨"*35)
        print("Generating visualization...")
        print("🎨"*35)
        
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "foundation_models_results.png"
        
        pipeline.visualize_results(image, save_path=str(output_path))
        
        print(f"\n✅ Results saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Final cleanup
    print("\n" + "🧹"*35)
    print("Final cleanup...")
    print("🧹"*35)
    cleanup_memory()
    
    print("\n" + "✅"*35)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("✅"*35 + "\n")
    
    return 0


def run_pipeline_with_custom_image(image_path: str):
    """
    Run pipeline on a custom image file.
    
    Args:
        image_path: Path to the image file
    """
    from PIL import Image
    from torchvision import transforms
    
    print(f"\nLoading image from: {image_path}")
    
    # Load image
    img = Image.open(image_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize to 256x256
    img = img.resize((256, 256))
    
    # Convert to tensor
    transform = transforms.ToTensor()
    image = transform(img)  # [1, H, W]
    
    print(f"✓ Image loaded: shape {image.shape}")
    
    # Create default bounding box (entire image)
    H, W = image.shape[-2:]
    boxes = [[0, 0, W-1, H-1]]
    
    # Initialize pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = FoundationModelsPipeline(device=device)
    
    # Run all models
    results = pipeline.run_all(
        image=image,
        boxes=boxes,
        text_prompt="medical image"
    )
    
    # Visualize
    output_path = Path("results") / f"results_{Path(image_path).stem}.png"
    pipeline.visualize_results(image, save_path=str(output_path))
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run foundation models pipeline for medical image segmentation"
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to input image (optional, uses synthetic data if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    try:
        if args.image:
            # Run on custom image
            results = run_pipeline_with_custom_image(args.image)
        else:
            # Run on sample data
            exit_code = run_pipeline_on_sample_data()
            sys.exit(exit_code)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        cleanup_memory()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        sys.exit(1)


if __name__ == "__main__":
    main()
