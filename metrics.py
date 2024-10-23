import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from scipy.spatial.distance import directed_hausdorff
from utils import intersection, union

# Directory to save computed metrics
output_dir = 'metrics_output/enet/'
os.makedirs(output_dir, exist_ok=True)

# Directories for actual (ground truth) and predicted segmentations
ground_truths_dir = 'data/segthor_train_fixed/train/'
predictions_dir = 'volumes/segthor/enet/'

# Initialize arrays for storing the actual and predicted segmentations
ground_truths = []
predictions = []

for patient_file in sorted(os.listdir(ground_truths_dir)):
    actual_file_path = os.path.join(ground_truths_dir, patient_file, 'GT2.nii.gz')  # GT2.nii.gz inside patient folder
    predicted_file_path = os.path.join(predictions_dir, f'{patient_file}.nii.gz')   # Prediction file matches patient folder name

    # Check for the existence of both actual and predicted files
    if os.path.exists(actual_file_path) and os.path.exists(predicted_file_path):
        # Load images using SimpleITK and nibabel
        actual_image = sitk.ReadImage(actual_file_path)
        
        predicted_image = sitk.ReadImage(predicted_file_path)
        
        # Add the loaded images to lists
        ground_truths.append(actual_image)
        predictions.append(predicted_image)
    else:
        print(f"Warning: Files missing for {patient_file} in one or both directories.")

# Check how many images were loaded
print(f"Loaded {len(ground_truths)} actual images and {len(predictions)} predicted images.")

# Label mapping (this ensures the predicted labels align with the ground truth)
actual_labels = [1, 2, 3, 4]  # Ground truth classes
predicted_labels = [63, 126, 189, 252]  # Example predicted labels mapping

def hausdorff_distance_computation(gt_mask, pred_mask):
    """
    Computes the Hausdorff distance between ground truth and predicted masks.
    """
    coordinates_gt = np.column_stack(np.where(gt_mask))
    coordinates_pred = np.column_stack(np.where(pred_mask))
    
    if len(coordinates_gt) == 0 or len(coordinates_pred) == 0:
        return np.nan  # No Hausdorff distance if masks are empty
    
    dist_gt_to_pred = directed_hausdorff(coordinates_gt, coordinates_pred)[0]
    dist_pred_to_gt = directed_hausdorff(coordinates_pred, coordinates_gt)[0]
    
    return max(dist_gt_to_pred, dist_pred_to_gt)

def compute_metrics(predictions, ground_truths, actual_labels, predicted_labels):
    """
    Computes Hausdorff distance and IoU for each scan and class.
    """
    N = len(predictions)  # Number of scans
    K = len(actual_labels) + 1  # Number of classes (excluding background)
    D = 1  # Metric dimensionality (simple scalar metrics)

    # Initialize the metrics arrays
    hausdorff_metrics = np.zeros((N, K, D))
    iou_metrics = np.zeros((N, K, D))
    dice_metrics = np.zeros((N, K, D))

    for n in range(N):
        pred_scan = sitk.GetArrayFromImage(predictions[n])  # 3D prediction for scan n
        gt_scan = sitk.GetArrayFromImage(ground_truths[n])  # 3D ground truth for scan n
        
        print(f"\nProcessing Scan {n+1}/{N}")

        # Loop through each class, including background (class 0)
        for k in range(K):
            if k == 0:  # Background
                print("Processing Background Class (0)...")
                pred_mask = (pred_scan == 0).astype(np.uint8)
                gt_mask = (gt_scan == 0).astype(np.uint8)
            else:
                # Map actual label to predicted label for non-background classes
                actual_label = actual_labels[k-1]
                predicted_label = predicted_labels[k-1]
                print(f"Processing Class {actual_label} (Predicted as {predicted_label})...")
                pred_mask = (pred_scan == predicted_label).astype(np.uint8)
                gt_mask = (gt_scan == actual_label).astype(np.uint8)

            # Hausdorff Distance
            print(f"  Calculating Hausdorff Distance for Class {k}...")
            if np.any(pred_mask) and np.any(gt_mask):
                hausdorff_metrics[n, k, 0] = hausdorff_distance_computation(gt_mask, pred_mask)
            else:
                hausdorff_metrics[n, k, 0] = np.nan  # No meaningful Hausdorff distance
                print(f"  No mask present for Class {k}, skipping Hausdorff Distance.")

            # IoU Calculation
            print(f"  Calculating IoU for Class {k}...")
            pred_mask_tensor = torch.tensor(pred_mask, dtype=torch.uint8).cuda()
            gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.uint8).cuda()

            intersection_mask = intersection(pred_mask_tensor, gt_mask_tensor).cuda()
            union_mask = union(pred_mask_tensor, gt_mask_tensor).cuda()

            intersection_sum = intersection_mask.sum().item()
            union_sum = union_mask.sum().item()

            if union_sum == 0:
                iou_metrics[n, k, 0] = np.nan  # No meaningful IoU
                print(f"  No overlap for Class {k}, skipping IoU.")
            else:
                iou_metrics[n, k, 0] = intersection_sum / union_sum

            # Dice Coefficient Calculation
            print(f"  Calculating Dice Coefficient for Class {k}...")
            pred_sum = pred_mask_tensor.sum().item()
            gt_sum = gt_mask_tensor.sum().item()

            if pred_sum + gt_sum == 0:
                dice_metrics[n, k, 0] = np.nan  # No meaningful Dice score
                print(f"  No mask present for Class {k}, skipping Dice.")
            else:
                dice_metrics[n, k, 0] = 2 * intersection_sum / (pred_sum + gt_sum)

        # Save intermediate results after each scan
        np.save(os.path.join(output_dir, 'hausdorff_per_class.npy'), hausdorff_metrics)
        np.save(os.path.join(output_dir, 'iou_per_class.npy'), iou_metrics)
        np.save(os.path.join(output_dir, 'dice_per_class.npy'), dice_metrics)

        print(f"Saved intermediate results after Scan {n+1}")
    return hausdorff_metrics, iou_metrics, dice_metrics

# Compute metrics, including background
hausdorff, iou, dice = compute_metrics(predictions, ground_truths, actual_labels, predicted_labels)

# Save metrics to output directory
np.save(os.path.join(output_dir, 'hausdorff_per_class.npy'), hausdorff)
np.save(os.path.join(output_dir, 'iou_per_class.npy'), iou)
np.save(os.path.join(output_dir, 'dice_per_class.npy'), dice)

print(f"Hausdorff Metrics Shape: {hausdorff.shape}")
print(f"IoU Metrics Shape: {iou.shape}")
print(f"Dice Metrics Shape: {dice.shape}")