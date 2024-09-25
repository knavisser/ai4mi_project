import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse

def analyze_fourier_transform(pat):
    # Load NIfTI file (it can be .nii or .nii.gz)
    img = nib.load(f"data/segthor_train/train/{pat}/{pat}.nii.gz")
    data = img.get_fdata()

    # Select a single slice or average over slices
    slice_data = data[:, :, data.shape[2] // 2]  # Select the middle slice
    # You can also average over all slices: slice_data = np.mean(data, axis=2)

    # Perform Fourier Transform
    fourier_transformed = np.fft.fft2(slice_data)
    power_spectrum = np.abs(fourier_transformed) ** 2

    # Shift the zero frequency component to the center
    power_spectrum_shifted = np.fft.fftshift(power_spectrum)

    # Plot the power spectrum
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log1p(power_spectrum_shifted), cmap='gray')
    plt.title(f'Power Spectrum for {pat} (Log Scale)')
    plt.colorbar(label='Log Power Spectrum')
    plt.axis('off')
    plt.show()

    # Plot histogram of pixel intensities
    plt.figure(figsize=(10, 6))
    plt.hist(slice_data.flatten(), bins=256, color='blue', alpha=0.7)
    plt.title('Histogram of Pixel Intensities')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])  # Adjust based on your data range
    plt.grid()
    plt.show()

    # Calculate histogram and set a threshold based on percentiles
    hist, bin_edges = np.histogram(slice_data.flatten(), bins=256)
    cumulative_hist = np.cumsum(hist)

    # Use the 97th percentile to set a threshold
    threshold_value = bin_edges[np.where(cumulative_hist >= 0.97 * cumulative_hist[-1])[0][0]]

    signal_mask = slice_data > threshold_value  # Mask for signal areas

    # Step 2: Calculate mean signal and std noise using the mask
    mean_signal = np.mean(slice_data[signal_mask])
    std_noise = np.std(slice_data[~signal_mask])  # Noise from areas below the threshold

    # Step 3: Calculate SNR
    if std_noise > 0:
        snr = mean_signal / std_noise
    else:
        snr = float('inf')  # Handle division by zero

    print(f'Signal-to-Noise Ratio (SNR) for {pat}: {snr:.2f}')

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Analyze Fourier Transform of NIfTI images.')
    parser.add_argument('pat', type=str, nargs='+', help='Paths to the NIfTI files (.nii or .nii.gz)')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Loop through each file path and call the analysis function
    for file_path in args.pat:
        analyze_fourier_transform(file_path)