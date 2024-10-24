import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

# Define the output directory and the numpy files
metrics_dir = 'metrics_output/enet/'
#metrics_dir = 'metrics_output/unet/'
hausdorff_file = os.path.join(metrics_dir, 'hausdorff_per_class.npy')
iou_file = os.path.join(metrics_dir, 'iou_per_class.npy')
dice_file = os.path.join(metrics_dir, 'dice_per_class.npy')

# Check if the files exist
hausdorff_exists = os.path.exists(hausdorff_file)
iou_exists = os.path.exists(iou_file)
dice_exists = os.path.exists(dice_file)

# Load the metrics if the files exist
hausdorff_data = np.load(hausdorff_file) if hausdorff_exists else None
iou_data = np.load(iou_file) if iou_exists else None
dice_data = np.load(dice_file) if dice_exists else None

# Prepare information about the numpy arrays
info = {
    "Hausdorff Metrics File": hausdorff_file,
    "Hausdorff Metrics Shape": hausdorff_data.shape if hausdorff_data is not None else "File not found",
    "IoU Metrics File": iou_file,
    "IoU Metrics Shape": iou_data.shape if iou_data is not None else "File not found",
    "Dice Metrics File": dice_file,
    "Dice Metrics Shape": dice_data.shape if dice_data is not None else "File not found"
}

# Convert the information to a DataFrame for better visualization
info_df = pd.DataFrame([info])
print(info_df)


# Function to print metrics per class
def print_metrics_per_class(metric_data, metric_name):
    if metric_data is not None:
        N, K, D = metric_data.shape
        for n in range(N):
            print(f"\nScan {n+1} {metric_name}:")
            for k in range(K):
                print(f"  Class {k}: {metric_data[n, k, :]}")


# Function to compute mean and standard deviation per class
def compute_mean_std_per_class(metric_data, metric_name=None):
    if metric_data is not None:
        _, K, _ = metric_data.shape
        
        # Compute mean and standard deviation
        mean_per_class = np.nanmean(metric_data[:, :, 0], axis=0)
        std_per_class = np.nanstd(metric_data[:, :, 0], axis=0)
        
        # Compute "All classes" by averaging across all existing classes (skip background)
        all_classes_mean = np.nanmean(mean_per_class[1:])
        all_classes_std = np.nanmean(std_per_class[1:])

        # Append "All classes" mean and std
        mean_per_class = np.append(mean_per_class, all_classes_mean)
        std_per_class = np.append(std_per_class, all_classes_std)

        if metric_name:
            print(f"\n{metric_name} Mean and Standard Deviation Per Class:")
            for k in range(K):
                print(f"  Class {k}: Mean = {mean_per_class[k]:.4f}, Std = {std_per_class[k]:.4f}")
            print(f"  All classes: Mean = {all_classes_mean:.4f}, Std = {all_classes_std:.4f}")
        
        return mean_per_class, std_per_class
    return None, None


# Function to combine all classes for "All classes" plots
def combine_all_classes(metric_data):
    if metric_data is not None:
        N, K, _ = metric_data.shape
        combined_data = np.nanmean(metric_data[:, 1:, 0], axis=1)  # Average across all non-background classes
        return combined_data
    return None


# Function to plot boxplot and line plot and save to file
def plot_metrics(metric_data, metric_name, class_labels, class_colors, save_filename):
    if metric_data is not None:
        N, K, _ = metric_data.shape
        
        # Boxplot for each class with custom colors
        plt.figure(figsize=(10, 6))
        data_per_class = [metric_data[:, k, 0] for k in range(K)]
        
        # Add combined "All classes" data to the boxplot
        combined_data = combine_all_classes(metric_data)
        data_per_class.append(combined_data)

        # Create boxplot and color each box manually with black median line
        boxprops = dict(color='black', facecolor='white')
        medianprops = dict(color='black')
        bp = plt.boxplot(data_per_class, labels=class_labels, patch_artist=True, medianprops=medianprops)

        for patch, color in zip(bp['boxes'], class_colors):
            patch.set_facecolor(color)
        
        plt.title(f"{metric_name} Distribution per Class (Boxplot)")
        plt.xlabel("Class")
        plt.ylabel(f"{metric_name} Value")
        
        plt.savefig(os.path.join(metrics_dir, f'{save_filename}_boxplot.png'))
        plt.show()
        plt.close()

        # Line plot for Mean and Std per class with custom colors and connecting lines
        mean_per_class, std_per_class = compute_mean_std_per_class(metric_data)
        if mean_per_class is not None:
            plt.figure(figsize=(10, 6))
            
            plt.plot(range(K+1), mean_per_class, '-o', color='grey', label='Mean')
            for k in range(K+1):
                plt.scatter(k, mean_per_class[k], color=class_colors[k], s=100, zorder=3)
            
            plt.errorbar(range(K+1), mean_per_class, yerr=std_per_class, fmt='none', color='black', capsize=5)
            
            plt.title(f"Mean and Std of {metric_name} per Class")
            plt.xlabel("Class")
            plt.ylabel(f"{metric_name} Value")
            plt.xticks(range(K+1), class_labels)
            plt.grid(True)

            plt.savefig(os.path.join(metrics_dir, f'{save_filename}_lineplot.png'))
            plt.show()
            plt.close()


# Function to plot violin plot and save to file
def plot_violin_metrics(metric_data, metric_name, class_labels, class_colors, save_filename, enforce_non_negative=False):
    if metric_data is not None:
        N, K, _ = metric_data.shape
        
        plt.figure(figsize=(10, 6))
        data_per_class = [metric_data[:, k, 0] for k in range(K)]
        
        combined_data = combine_all_classes(metric_data)
        data_per_class.append(combined_data)

        sns.violinplot(data=data_per_class, palette=class_colors, inner='quartile', linewidth=2)

        plt.title(f"{metric_name} Distribution per Class (Violin Plot)")
        plt.xlabel("Class")
        plt.ylabel(f"{metric_name} Value")
        plt.xticks(range(K+1), class_labels)

        if enforce_non_negative:
            plt.ylim(0, None)

        median_line = mlines.Line2D([], [], color='black', linestyle='-', label='Median')
        quartile_line = mlines.Line2D([], [], color='black', linestyle='--', label='Quartiles')
        plt.legend(handles=[median_line, quartile_line], loc='upper right')

        plt.savefig(os.path.join(metrics_dir, f'{save_filename}_violinplot.png'))
        plt.show()
        plt.close()


# Class labels for the X-axis and corresponding colors
class_labels = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta', 'All classes']
class_colors = ['grey', 'green', 'yellow', 'red', 'blue', 'purple']

# Print Metrics
print_metrics_per_class(hausdorff_data, "Hausdorff Distance")
print_metrics_per_class(iou_data, "IoU")
print_metrics_per_class(dice_data, "Dice Coefficient")

compute_mean_std_per_class(hausdorff_data, "Hausdorff Distance")
compute_mean_std_per_class(iou_data, "IoU")
compute_mean_std_per_class(dice_data, "Dice Coefficient")

# Plot and save 3D Metrics
plot_metrics(hausdorff_data, "Hausdorff Distance", class_labels, class_colors, "hausdorff")
plot_metrics(iou_data, "IoU", class_labels, class_colors, "iou")
plot_metrics(dice_data, "Dice Coefficient", class_labels, class_colors, "dice")

# Plot and save 3D Metrics: Violin Plots
plot_violin_metrics(hausdorff_data, "Hausdorff Distance", class_labels, class_colors, "hausdorff_violin", enforce_non_negative=True)
plot_violin_metrics(iou_data, "IoU", class_labels, class_colors, "iou_violin")
plot_violin_metrics(dice_data, "Dice Coefficient", class_labels, class_colors, "dice_violin")