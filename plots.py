import matplotlib.pyplot as plt
import nrrd  # You might need to install this: pip install pynrrd
import os

def plot_mri_masks(mri_volume_path, breast_mask_path, dv_mask_path, output_dir=None, slice_index=50):
    """
    Loads MRI volume and corresponding mask data, generates a 3-subplot visualization,
    and optionally saves the plot.

    Args:
        mri_volume_path (str): Path to the MRI volume file (e.g., .nrrd).
        breast_mask_path (str): Path to the breast mask file (e.g., .nrrd).
        dv_mask_path (str): Path to the FGT + Blood Vessel mask file (e.g., .nrrd).
        output_dir (str, optional): Directory to save the plots. If None, plots are shown.
        slice_index (int, optional): The slice index to display. Defaults to 50.
    """
    try:
        # Load the data
        image_array, _ = nrrd.read(mri_volume_path)
        nrrd_breast_data, _ = nrrd.read(breast_mask_path)
        nrrd_dv_data, _ = nrrd.read(dv_mask_path)

        # Ensure the slice index is within bounds
        if not (0 <= slice_index < image_array.shape[2]):
            print(f"Warning: Slice index {slice_index} is out of bounds for {mri_volume_path}. "
                  f"Using middle slice instead.")
            slice_index = image_array.shape[2] // 2
        
        if not (0 <= slice_index < nrrd_breast_data.shape[2]):
            print(f"Warning: Slice index {slice_index} is out of bounds for {breast_mask_path}. "
                  f"Using middle slice instead.")
            slice_index = nrrd_breast_data.shape[2] // 2
            
        if not (0 <= slice_index < nrrd_dv_data.shape[2]):
            print(f"Warning: Slice index {slice_index} is out of bounds for {dv_mask_path}. "
                  f"Using middle slice instead.")
            slice_index = nrrd_dv_data.shape[2] // 2


        plt.figure(figsize=(15, 5)) # Adjust figure size as needed

        plt.subplot(1, 3, 1)
        plt.title('MRI Volume')
        plt.imshow(image_array[:, :, slice_index], cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Breast Mask')
        plt.imshow(nrrd_breast_data[:, :, slice_index], cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('FGT + Blood Vessel Mask')
        plt.imshow(nrrd_dv_data[:, :, slice_index], cmap='gray')
        plt.axis('off')

        plt.tight_layout() # Adjust subplot parameters for a tight layout

        if output_dir:
            # Create a unique filename based on the input file
            base_name = os.path.basename(mri_volume_path).replace('.nrrd', '')
            output_filename = os.path.join(output_dir, f"{base_name}_slice_{slice_index}.png")
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            plt.close() # Close the plot to free memory
            print(f"Saved plot for {base_name} to {output_filename}")
        else:
            plt.show()
            plt.close() # Close the plot after showing
            
    except FileNotFoundError as e:
        print(f"Error: One or more files not found: {e}")
    except nrrd.NRRDError as e:
        print(f"Error reading NRRD file: {e}. Check file format for {mri_volume_path}, {breast_mask_path}, or {dv_mask_path}")
    except Exception as e:
        print(f"An unexpected error occurred for {mri_volume_path}: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # Set the directory where your image files are located
    data_directory = '/net/projects/annawoodard/tcia_datasets/duke_breast_cancer_mri/dicom' 
    # Set the directory where you want to save the output plots (create if it doesn't exist)
    output_plots_directory = '/net/projects/annawoodard/tcia_datasets/duke_breast_cancer_mri/plots_output' 
    
    # Create the output directory if it doesn't exist
    if output_plots_directory and not os.path.exists(output_plots_directory):
        os.makedirs(output_plots_directory)
        print(f"Created output directory: {output_plots_directory}")

    # --- File Discovery and Processing ---
    # This part depends heavily on your file naming convention.
    # Here's an example assuming files are named like:
    #   'patientA_MRI.nrrd', 'patientA_breast_mask.nrrd', 'patientA_dv_mask.nrrd'
    #   'patientB_MRI.nrrd', 'patientB_breast_mask.nrrd', 'patientB_dv_mask.nrrd'
    
    # A dictionary to store grouped files (e.g., {'patientA': {'mri': 'path/to/mri.nrrd', ...}})
    grouped_files = {}

    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.nrrd'):
                full_path = os.path.join(root, file)
                # Extract patient ID or common prefix
                # This is a crucial part you'll need to adapt to your actual filenames
                
                # Example: assuming filenames like 'patientID_type.nrrd'
                parts = file.split('_')
                if len(parts) >= 2:
                    patient_id = parts[0]
                    file_type = parts[1].split('.')[0] # e.g., 'MRI', 'breast', 'dv'

                    if patient_id not in grouped_files:
                        grouped_files[patient_id] = {}
                    
                    if 'MRI' in file_type.lower():
                        grouped_files[patient_id]['mri'] = full_path
                    elif 'breast' in file_type.lower():
                        grouped_files[patient_id]['breast'] = full_path
                    elif 'dv' in file_type.lower(): # Or 'fgt', 'vessel', etc.
                        grouped_files[patient_id]['dv'] = full_path
    
    # Process each patient's data
    processed_count = 0
    for patient_id, paths in grouped_files.items():
        if 'mri' in paths and 'breast' in paths and 'dv' in paths:
            print(f"Processing data for patient: {patient_id}")
            plot_mri_masks(
                mri_volume_path=paths['mri'],
                breast_mask_path=paths['breast'],
                dv_mask_path=paths['dv'],
                output_dir=output_plots_directory, # Set to None if you want to display plots
                slice_index=50 # You can make this dynamic if needed
            )
            processed_count += 1
        else:
            print(f"Skipping patient {patient_id}: Missing one or more required files.")

    if processed_count == 0:
        print("No complete sets of MRI, Breast Mask, and DV Mask files found based on the current naming convention.")
        print("Please check 'data_directory' and adapt the file parsing logic in the script.")
    else:
        print(f"\nSuccessfully processed {processed_count} sets of images.")
