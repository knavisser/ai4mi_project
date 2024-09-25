import os
import nibabel as nib
import numpy as np

class NiftiProcessor:
    def __init__(self, reference_file, current_file):
        self.ref_data = self.load_nifti_data(reference_file)
        self.curr_data = self.load_nifti_data(current_file)
        self.offset_vector = self.calculate_offset(self.ref_data, self.curr_data)
    
    @staticmethod
    def load_nifti_data(file_path):
        """Load a NIfTI file and convert to a NumPy array."""
        return np.asarray(nib.load(file_path).dataobj)

    @staticmethod
    def extract_coordinates(binary_mask):
        """Get the coordinates of non-zero values in the mask."""
        return np.argwhere(binary_mask)

    @staticmethod
    def calculate_centroid(points_array):
        """Find centroid of 3D coordinates."""
        return np.mean(points_array, axis=0)

    @staticmethod
    def shift_coordinates(points_array, offset):
        """Apply shift to 3D coordinates."""
        return points_array + offset

    @staticmethod
    def generate_mask_from_points(points, target_shape):
        """Convert the shifted points back to a binary mask."""
        mask = np.zeros(target_shape, dtype=np.uint8)
        for coord in points.astype(int):
            mask[tuple(coord)] = 1
        return mask

    @staticmethod
    def save_nifti_image(mask_array, filepath, reference_img):
        """Save a modified array as a NIfTI file."""
        if os.path.exists(filepath):
            print(f"File {filepath} exists, replacing it with the new mask.")
        new_img = nib.Nifti1Image(mask_array, affine=reference_img.affine, header=reference_img.header)
        nib.save(new_img, filepath)

    def calculate_offset(self, ref_data, curr_data):
        """Calculate the shift required to align the centroids of the regions of interest."""
        ref_region = (ref_data == 2)
        curr_region = (curr_data == 2)

        ref_points = self.extract_coordinates(ref_region)
        curr_points = self.extract_coordinates(curr_region)

        ref_centroid = self.calculate_centroid(ref_points)
        curr_centroid = self.calculate_centroid(curr_points)

        return ref_centroid - curr_centroid

    def process_patient(self, patient_id_str, output_dir):
        """Process a single patient and save the aligned mask."""
        patient_path = os.path.join(output_dir, f"Patient_{patient_id_str}/GT.nii.gz")
        gt_patient = nib.load(patient_path)
        patient_data = np.asarray(gt_patient.dataobj)

        # Create binary mask and shift coordinates
        target_region_mask = (patient_data == 2)
        target_points = self.extract_coordinates(target_region_mask)
        shifted_points = self.shift_coordinates(target_points, self.offset_vector)
        aligned_target_mask = self.generate_mask_from_points(shifted_points, target_region_mask.shape)

        # Create modified patient data
        modified_patient_data = np.copy(patient_data)
        modified_patient_data[modified_patient_data == 2] = 0  # Remove original region
        modified_patient_data[aligned_target_mask == 1] = 2     # Insert aligned region

        # Save updated mask
        self.save_nifti_image(modified_patient_data, patient_path, gt_patient)

    def process_all_patients(self, num_patients, output_dir):
        """Process all patients in the dataset."""
        for patient_id in range(1, num_patients + 1):
            patient_id_str = f"{patient_id:02d}"
            print(f"Processing patient {patient_id_str}")
            self.process_patient(patient_id_str, output_dir)
        print("All patients processed successfully.")


# Main logic to execute the processing for all patients
if __name__ == "__main__":
    reference_file = "data/segthor_train/train/Patient_27/GT2.nii.gz"
    current_file = "data/segthor_train/train/Patient_27/GT.nii.gz"
    output_dir = "data/segthor_train/train"
    num_patients = 40

    processor = NiftiProcessor(reference_file, current_file)
    processor.process_all_patients(num_patients, output_dir)