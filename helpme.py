import matplotlib.pyplot as plt
import os
from preprocessing import * # Assuming preprocessing.py is in the same directory or accessible via PYTHONPATH


# Set global figure size for matplotlib plots
plt.rcParams["figure.figsize"] = [10, 10]

# Define the base directory for DICOM data
dicom_base_dir = '/net/projects/annawoodard/tcia_datasets/duke_breast_cancer_mri/dicom'
print("checkpoint 2")
# Define the path to the filename mapping CSV
mapping_csv_path = '/net/projects/annawoodard/tcia_datasets/duke_breast_cancer_mri/filename_map.csv'
print("checkpoint 3")
# Define the annotation type
# annotation_type = 'train_annotations'

# Clean the filepath filename mapping once
fpath_mapping_df = clean_filepath_filename_mapping_csv(mapping_csv_path)
print("checkpoint 4")
# Get a list of all subject directories (series)
# We assume each subject's DICOM data is in a subfolder named after the subject ID
subject_dirs = [d for d in os.listdir(dicom_base_dir) if os.path.isdir(os.path.join(dicom_base_dir, d))]
print("checkpoint 5")
# Loop through each subject directory
for sample_subject in subject_dirs: 
    print(f"Processing subject: {sample_subject}")
    nrrd_breast_data = None
    nrrd_dv_data = None
    try:
        image_array, dcm_data = read_precontrast_mri_and_segmentation(
            sample_subject,
            dicom_base_dir,
            fpath_mapping_df,
        )
        nrrd_breast_data = np.zeros_like(image_array)
        nrrd_dv_data = np.zeros_like(image_array)

        image_array = zscore_image(normalize_image(image_array))
        # Determine a slice to display. You might want to make this dynamic
        # For simplicity, we'll keep it at 50, but a robust script might
        # find the middle slice or a slice with content.
        slice_to_display = image_array.shape[2] // 2 # Use the middle slice

        # Create a new figure for each subject
        plt.figure()

        plt.subplot(1, 3, 1)
        plt.title(f'MRI Volume ({sample_subject})')
        plt.imshow(image_array[:, :, slice_to_display], cmap='gray')
        plt.axis('off')

        if nrrd_breast_data is not None:
            plt.subplot(1, 3, 2)
            plt.title('Breast Mask')
            plt.imshow(nrrd_breast_data[:, :, slice_to_display], cmap='gray')
            plt.axis('off')

        if nrrd_dv_data is not None:
            plt.subplot(1, 3, 3)
            plt.title('FGT + Blood Vessel Mask')
            plt.imshow(nrrd_dv_data[:, :, slice_to_display], cmap='gray')
            plt.axis('off')


        

        # Save the plot. You'll want to specify an output directory.
        # For example, create a 'visualizations' folder in your current working directory.
        output_dir = './visualizations'
        os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist
        plt.savefig(os.path.join(output_dir, f'{sample_subject}_mri_segmentation.png'))
        plt.close() # Close the figure to free up memory
    except Exception as e:
        print(f"Error processing subject {sample_subject}: {e}") 
    break
print("Processing complete.")
