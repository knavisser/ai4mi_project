
import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, shift
from pathlib import Path
import shutil

# Load a NIfTI file and convert to a NumPy array.
def load_nifti_data(file_path):
    nifti = nib.load(str(file_path))
    return np.asarray(nifti.dataobj), nifti.affine

# Save a modified array as a NIfTI file.
def save_nifti_image(file_path, data, affine, header):
    nifti = nib.Nifti1Image(data, affine, header=header)
    nib.save(nifti, str(file_path))

# Calculate a correction matrix for affine transformation.
def calculate_correction_matrix():
    rotation_corr = np.eye(4)
    translation_corr = np.eye(4)
    translation_corr[:3, 3] = [0, 0, 10]  # Ten z-units
    correction_matrix = np.dot(translation_corr, rotation_corr)
    return correction_matrix

# Compute the shift vector for the heart region.
def compute_heart_shift(original_gt, corrected_gt):
    heart_original = (original_gt == 2).astype(np.float32)
    heart_corrected = (corrected_gt == 2).astype(np.float32)
    centroid_original = np.mean(np.argwhere(heart_original), axis=0)
    centroid_corrected = np.mean(np.argwhere(heart_corrected), axis=0)
    return centroid_corrected - centroid_original

# Apply affine transformation to the image data.
def apply_correction(image_data, correction_matrix):
    M = correction_matrix[:3, :3]
    t = correction_matrix[:3, 3]
    M_inv = np.linalg.inv(M)
    offset = -np.dot(M_inv, t)
    corrected_data = affine_transform(image_data, M_inv, offset=offset, order=1, mode='constant', cval=0.0)
    return corrected_data

# Shift the heart segment in the ground truth data.
def shift_heart_segment(gt_data, shift_vector):
    heart_segment = (gt_data == 2).astype(np.float32)
    shifted_heart = shift(heart_segment, shift=shift_vector, order=0, mode='constant', cval=0.0)
    gt_data_corrected = np.where(gt_data == 2, 0, gt_data)  # Remove the original heart segment
    gt_data_corrected = np.where(shifted_heart > 0, 2, gt_data_corrected)  # Add shifted heart segment
    return gt_data_corrected

#Process a single patient, applying affine corrections and heart shift.
def process_patient(patient_folder, correction_matrix, heart_shift_vector, output_folder):
    patient_output_folder = output_folder/patient_folder.name
    patient_output_folder.mkdir(parents=True, exist_ok=True)

    # Apply affine correction to CT image
    ct_path = patient_folder/f"{patient_folder.name}.nii.gz"
    ct_data, ct_affine = load_nifti_data(ct_path)
    corrected_ct = apply_correction(ct_data, correction_matrix)
    save_nifti_image(patient_output_folder/f"{patient_folder.name}.nii.gz", corrected_ct, ct_affine, nib.load(ct_path).header)

    # Apply affine correction to GT image and shift the heart segment
    gt_path = patient_folder/"GT.nii.gz"
    gt_data, gt_affine = load_nifti_data(gt_path)
    corrected_gt = apply_correction(gt_data, correction_matrix)
    corrected_gt = shift_heart_segment(corrected_gt, heart_shift_vector)
    save_nifti_image(patient_output_folder/"GT2.nii.gz", corrected_gt, gt_affine, nib.load(gt_path).header)

# Process all patients, excluding Patient_27.
def process_all_patients(data_root, output_root, correction_matrix, heart_shift_vector):
    
    patient_27_folder = data_root/"Patient_27"
    
    # Sort patient folders by name
    patient_folders = sorted(data_root.glob("Patient_*"), key=lambda x: int(x.name.split('_')[1]))

    for patient in patient_folders:
        if patient.name != "Patient_27":
            print(f"Processing {patient.name}")
            process_patient(patient, correction_matrix, heart_shift_vector, output_root)
        else:
            patient_27_output = output_root/"Patient_27"
            patient_27_output.mkdir(parents=True, exist_ok=True)
            for file in patient_27_folder.glob("*"):
                if file.name.endswith(".nii.gz"):
                    shutil.copy(str(file), str(patient_27_output / file.name))
    print("Done.")

def main():
    data_root = Path("data/segthor_train/train")
    output_root = Path("data/segthor_train_fixed/train")    
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Process Patient 27 first
    original_gt, _ = load_nifti_data("data/segthor_train/train/Patient_27/GT.nii.gz")
    corrected_gt, _ = load_nifti_data("data/segthor_train/train/Patient_27/GT2.nii.gz")
    
    correction_matrix = calculate_correction_matrix()
    heart_shift_vector = compute_heart_shift(original_gt, corrected_gt)

    process_all_patients(data_root, output_root, correction_matrix, heart_shift_vector)

if __name__ == "__main__":
    main()
