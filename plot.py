#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(metric_file: Path, dest: Path, headless: bool = False) -> None:
    metrics = np.load(metric_file)
    
    if metrics.ndim == 1:
        # Case for 1D metrics (e.g., overall loss per epoch)
        E = metrics.shape[0]  # E is the number of epochs
        mean_metrics = metrics  # No need to compute mean, already 1D
        K = 1
    elif metrics.ndim == 2:
        E, N = metrics.shape  # E is epochs, N is the number of samples
        K = 1  # No multiple classes
        mean_metrics = metrics.mean(axis=1)  # Mean across all samples for each epoch

    elif metrics.ndim == 3:
        E, N, K = metrics.shape  # E is epochs, N is number of samples, K is number of classes
        mean_metrics = metrics.mean(axis=1)  # Mean across all samples for each epoch and class
    else:
        print(f"Skipping {metric_file}: Unsupported shape {metrics.shape}")
        return  # Skip files with unsupported dimensionalities

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(str(metric_file))

    epcs = np.arange(E)

    if K > 1:
        # Plot for Dice scores (multi-class)
        labels = ["Esophagus", "Heart", "Trachea", "Aorta"]
        colors = ["green", "yellow", "red", "blue"]

        for k in range(1, K):
            y = mean_metrics[:, k]  # Mean per epoch for each class
            ax.plot(epcs, y, label=labels[k-1], color=colors[k-1], linewidth=1.5)

        ax.plot(epcs, mean_metrics.mean(axis=1), label="All classes", color="purple", linewidth=3)
        ax.legend()

                # Get min and max values for the multi-class data
        y_min = mean_metrics.min()
        y_max = mean_metrics.max()
    else:
        # Plot for 1D or 2D metrics like Hausdorff, IoU, or Loss
        ax.plot(epcs, mean_metrics, label="Metric Average", linewidth=3)
        
        # Get min and max values for the single metric data
        y_min = mean_metrics.min()
        y_max = mean_metrics.max()

    # Set dynamic y-axis limits based on the min and max of the metrics
    margin = 0.05 * (y_max - y_min)  # Add a small margin to the limits
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.legend()  # Ensure the legend is added to the plot
    fig.tight_layout()
    
    if dest:
        fig.savefig(dest)
    
    if not headless:
        plt.show()


def process_all_metrics(metric_dir: Path, dest_dir: Path, headless: bool = False):
    metric_files = list(metric_dir.glob("*.npy"))
    
    for metric_file in metric_files:
        try:
            dest_file = dest_dir / (metric_file.stem + ".png")
            print(f"Processing {metric_file} -> {dest_file}")
            plot_metrics(metric_file, dest_file, headless)
        except Exception as e:
            print(f"Error processing {metric_file}: {e}")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot metrics and save as images')
    parser.add_argument('--metric_dir', type=Path, required=True, help="Directory with .npy metric files")
    parser.add_argument('--dest_dir', type=Path, required=True, help="Directory to save the plots")
    parser.add_argument("--headless", action="store_true", help="Does not display the plot, saves directly.")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    process_all_metrics(args.metric_dir, args.dest_dir, args.headless)