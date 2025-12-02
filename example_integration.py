"""
Example: Integration with Neuro_Dani-4 Dataset

This script shows how to integrate the foundation models pipeline
with the existing ASOCA dataset infrastructure from Neuro_Dani-4.py
"""

import torch
import numpy as np
from foundation_models_pipeline import FoundationModelsPipeline, cleanup_memory


def find_heart_bbox_2d(img_t_norm: torch.Tensor,
                       thresh: float = 0.3,
                       min_area: int = 4000,
                       margin: int = 32):
    """
    Find heart bounding box from normalized image volume.
    (Copied from Neuro_Dani-4.py for compatibility)
    """
    from skimage.measure import label as sk_label, regionprops
    
    S, C, H, W = img_t_norm.shape
    assert C == 1
    
    mid = S // 2
    sl = img_t_norm[mid, 0].cpu().numpy()
    
    mask = sl > thresh
    lab = sk_label(mask)
    
    if lab.max() == 0:
        return 0, H, 0, W
    
    regions = regionprops(lab)
    reg = max(regions, key=lambda r: r.area)
    if reg.area < min_area:
        return 0, H, 0, W
    
    minr, minc, maxr, maxc = reg.bbox
    
    minr = max(minr - margin, 0)
    minc = max(minc - margin, 0)
    maxr = min(maxr + margin, H)
    maxc = min(maxc + margin, W)
    
    h = maxr - minr
    w = maxc - minc
    side = max(h, w)
    
    cy = (minr + maxr) / 2.0
    cx = (minc + maxc) / 2.0
    
    y0 = int(round(cy - side / 2.0))
    x0 = int(round(cx - side / 2.0))
    
    y0 = max(0, min(H - side, y0))
    x0 = max(0, min(W - side, x0))
    y1 = y0 + side
    x1 = x0 + side
    
    return y0, y1, x0, x1


def run_foundation_models_on_asoca_sample():
    """
    Example: Run foundation models on a sample from ASOCA dataset.
    
    This demonstrates the integration pattern. In practice, you would:
    1. Load your actual dataset using DatasetMerged_2d
    2. Extract slices from volumes
    3. Run the pipeline on each slice
    4. Aggregate results
    """
    
    print("="*70)
    print("FOUNDATION MODELS + ASOCA DATASET - INTEGRATION EXAMPLE")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Simulate loading from DatasetMerged_2d
    # In real usage: dataset = DatasetMerged_2d(split='val', img_side=256)
    print("\n📦 Simulating dataset loading...")
    
    # Create synthetic CT-like volume (simulating normalized ASOCA data)
    S, H, W = 100, 512, 512  # Slices, Height, Width
    volume = torch.rand(S, 1, H, W) * 0.5  # Normalized to [0, 0.5]
    
    # Add a "heart-like" structure in the center
    center_s = S // 2
    for s in range(max(0, center_s - 20), min(S, center_s + 20)):
        y_center, x_center = H // 2, W // 2
        radius = 100
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist = torch.sqrt((y - y_center)**2 + (x - x_center)**2)
        volume[s, 0, dist < radius] = 0.7 + torch.rand(1).item() * 0.1
    
    print(f"✓ Volume shape: {volume.shape}")
    
    # Find heart bounding box (as done in Neuro_Dani-4)
    print("\n🔍 Finding heart bounding box...")
    bbox = find_heart_bbox_2d(volume, thresh=0.3, min_area=4000, margin=32)
    y0, y1, x0, x1 = bbox
    print(f"✓ Heart bbox: y=[{y0}:{y1}], x=[{x0}:{x1}]")
    
    # Crop and resize to 256x256 (as in training)
    print("\n✂️  Cropping and resizing...")
    vol_crop = volume[:, :, y0:y1, x0:x1]
    
    # Resize to 256x256
    import torch.nn.functional as F
    vol_crop_resized = F.interpolate(
        vol_crop.float(),
        size=(256, 256),
        mode='bilinear',
        align_corners=False
    )
    
    print(f"✓ Cropped volume shape: {vol_crop_resized.shape}")
    
    # Select a slice with anatomical structures (middle slice)
    slice_idx = vol_crop_resized.shape[0] // 2
    slice_img = vol_crop_resized[slice_idx]  # [1, 256, 256]
    
    print(f"✓ Selected slice {slice_idx}: shape {slice_img.shape}")
    
    # Create bounding box for vessels (simulate GT-based box)
    # In real scenario, this would come from ground truth label
    H_crop, W_crop = slice_img.shape[-2:]
    vessel_bbox = [[
        W_crop // 4,      # x_min
        H_crop // 4,      # y_min
        3 * W_crop // 4,  # x_max
        3 * H_crop // 4   # y_max
    ]]
    
    print(f"✓ Vessel bbox (for SAM/MedSAM): {vessel_bbox}")
    
    # Initialize foundation models pipeline
    print("\n🚀 Initializing Foundation Models Pipeline...")
    pipeline = FoundationModelsPipeline(device=device)
    
    # Run all models on the selected slice
    print("\n" + "="*70)
    print("RUNNING ALL FOUNDATION MODELS")
    print("="*70)
    
    try:
        results = pipeline.run_all(
            image=slice_img,
            boxes=vessel_bbox,
            text_prompt="coronary artery in CT scan"
        )
        
        # Analyze results
        print("\n" + "📊"*35)
        print("RESULTS ANALYSIS")
        print("📊"*35)
        
        # SAM results
        if 'sam' in results:
            sam_mask = results['sam']
            sam_coverage = (sam_mask > 0.5).float().mean().item()
            print(f"\n✓ SAM Segmentation:")
            print(f"  - Mask shape: {sam_mask.shape}")
            print(f"  - Coverage: {sam_coverage*100:.2f}%")
        
        # MedSAM results
        if 'medsam' in results:
            medsam_mask = results['medsam']
            medsam_coverage = (medsam_mask > 0.5).float().mean().item()
            print(f"\n✓ MedSAM Segmentation:")
            print(f"  - Mask shape: {medsam_mask.shape}")
            print(f"  - Coverage: {medsam_coverage*100:.2f}%")
        
        # CLIP results
        if 'clip' in results:
            clip_result = results['clip']
            print(f"\n✓ CLIP Analysis:")
            print(f"  - Image-Text Similarity: {clip_result['similarity']:.4f}")
            print(f"  - Image features shape: {clip_result['image_features'].shape}")
        
        # ViT results
        if 'vit' in results:
            vit_result = results['vit']
            print(f"\n✓ ViT Features:")
            print(f"  - Hidden state shape: {vit_result['last_hidden_state'].shape}")
            if vit_result['pooler_output'] is not None:
                print(f"  - Pooled features shape: {vit_result['pooler_output'].shape}")
        
        # Visualize results
        print("\n" + "🎨"*35)
        print("VISUALIZATION")
        print("🎨"*35)
        
        from pathlib import Path
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "asoca_foundation_models_example.png"
        pipeline.visualize_results(slice_img, save_path=str(output_path))
        
        print(f"\n✅ Results saved to: {output_path}")
        
        # Compare SAM vs MedSAM if both available
        if 'sam' in results and 'medsam' in results:
            sam_mask = (results['sam'] > 0.5).float()
            medsam_mask = (results['medsam'] > 0.5).float()
            
            # Compute overlap (IoU)
            intersection = (sam_mask * medsam_mask).sum().item()
            union = (sam_mask + medsam_mask).clamp(0, 1).sum().item()
            
            if union > 0:
                iou = intersection / union
                print(f"\n📊 SAM vs MedSAM Overlap (IoU): {iou:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Cleanup
    print("\n" + "🧹"*35)
    cleanup_memory()
    
    print("\n" + "✅"*35)
    print("INTEGRATION EXAMPLE COMPLETED")
    print("✅"*35)
    
    return 0


def batch_process_multiple_patients():
    """
    Example: Process multiple patients/volumes sequentially.
    
    Shows how to:
    1. Load multiple volumes
    2. Process each with foundation models
    3. Maintain memory efficiency
    4. Aggregate results
    """
    
    print("\n" + "="*70)
    print("BATCH PROCESSING EXAMPLE")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulate multiple patient IDs
    patient_ids = ["Normal_1", "Normal_2", "Diseased_1"]
    
    all_results = {}
    
    for patient_id in patient_ids:
        print(f"\n{'='*70}")
        print(f"Processing: {patient_id}")
        print(f"{'='*70}")
        
        # Simulate loading patient volume
        # In practice: img_t, lab_t = dataset._load_image_label(patient_id)
        volume = torch.rand(80, 1, 512, 512) * 0.5
        
        # Find heart and crop
        bbox = find_heart_bbox_2d(volume)
        y0, y1, x0, x1 = bbox
        vol_crop = volume[:, :, y0:y1, x0:x1]
        
        # Resize
        import torch.nn.functional as F
        vol_crop = F.interpolate(
            vol_crop.float(),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )
        
        # Process middle slice
        slice_img = vol_crop[vol_crop.shape[0] // 2]
        
        # Run pipeline
        pipeline = FoundationModelsPipeline(device=device)
        
        try:
            results = pipeline.run_all(
                image=slice_img,
                text_prompt="medical structure"
            )
            
            all_results[patient_id] = results
            
            # Save visualization
            from pathlib import Path
            output_path = Path("results") / f"{patient_id}_results.png"
            pipeline.visualize_results(slice_img, save_path=str(output_path))
            
            print(f"✓ {patient_id} processed successfully")
            
        except Exception as e:
            print(f"✗ Error processing {patient_id}: {e}")
            all_results[patient_id] = None
        
        # Important: Memory cleanup between patients
        cleanup_memory()
    
    # Summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    
    successful = sum(1 for v in all_results.values() if v is not None)
    print(f"✓ Successfully processed: {successful}/{len(patient_ids)} patients")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    print("\n🔬 Foundation Models Integration Examples\n")
    
    # Run single example
    print("\n" + "🔹"*35)
    print("EXAMPLE 1: Single Slice Processing")
    print("🔹"*35)
    
    try:
        exit_code = run_foundation_models_on_asoca_sample()
    except Exception as e:
        print(f"\n❌ Error in Example 1: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    if exit_code == 0:
        # Run batch example
        print("\n\n" + "🔹"*35)
        print("EXAMPLE 2: Batch Processing")
        print("🔹"*35)
        
        try:
            batch_process_multiple_patients()
        except Exception as e:
            print(f"\n❌ Error in Example 2: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "🎉"*35)
    print("ALL EXAMPLES COMPLETED")
    print("🎉"*35 + "\n")
