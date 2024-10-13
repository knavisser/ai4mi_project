import numpy as np
import os
import matplotlib.pyplot as plt

# Folder containing the .npy files
folder_path = 'results/segthor/ce'  # Replace with your folder path

# Folder to save the plots
output_folder = 'results/visualization'  # Replace with the path to where you want to save the plots
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Define the .npy files we are going to visualize
metrics_files = {
    "iou_test": "iou_test.npy",
    "iou_val": "iou_val.npy",
    "dice_test": "dice_test.npy",
    "dice_val": "dice_val.npy",
    "hausdorff_test": "hausdorff_test.npy",
    "hausdorff_val": "hausdorff_val.npy",
    "loss_tra": "loss_tra.npy",
    "loss_val": "loss_val.npy",
}

# Helper function to plot histograms and save them
def save_histogram(data, title, output_folder):
    plt.figure()
    plt.hist(data.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title(f"Histogram of {title}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Save the figure
    output_path = os.path.join(output_folder, f"{title}_histogram.png")
    plt.savefig(output_path)
    plt.close()  # Close the figure to avoid displaying it

# Load and save histograms for IoU, Dice, and Hausdorff
for metric, file_name in metrics_files.items():
    file_path = os.path.join(folder_path, file_name)
    data = np.load(file_path)
    
    # For IoU, Dice, and Hausdorff distances, save histograms
    if 'iou' in metric or 'dice' in metric or 'hausdorff' in metric:
        save_histogram(data, metric, output_folder)

# Function to save loss curves for training and validation
def save_loss_curves(train_data, val_data, title, output_folder):
    plt.figure()
    plt.plot(train_data.mean(axis=1), label="Training Loss", color='blue')
    plt.plot(val_data.mean(axis=1), label="Validation Loss", color='orange')
    plt.title(f"{title}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    output_path = os.path.join(output_folder, f"{title}_loss_curve.png")
    plt.savefig(output_path)
    plt.close()  # Close the figure to avoid displaying it

# Load and save loss curves
loss_tra = np.load(os.path.join(folder_path, metrics_files["loss_tra"]))
loss_val = np.load(os.path.join(folder_path, metrics_files["loss_val"]))

save_loss_curves(loss_tra, loss_val, "Training vs Validation Loss", output_folder)
