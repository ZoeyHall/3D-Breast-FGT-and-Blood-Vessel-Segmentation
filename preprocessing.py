import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
import os
import nrrd

# Images should be downloaded from Duke-Breast-Cancer-MRI Dataset on TCIA
# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903
# Download should be done using descriptive file path. There is an associated 
# file path mapping table that can be downloaded to determine which series is 
# the precontrast sequence. 

def clean_filepath_filename_mapping_csv(filepath_filename_csv_path):
    """
    This function cleans the "Breast-Cancer-MRI-filepath_filename-mapping.csv" 
    file that can be downloaded from TCIA. It is originally an excel file. It
    can be saved as a csv for use in this function. This returns a DataFrame
    that pairs subject_ids to their precontrast dir

    Parameters
    ----------
    fpath_mapping_df: str
        Path that leads to Breast-Cancer-MRI-filepath_filename-mapping.csv

    Returns
    -------
    pd.DataFrame
        Cleaned mapping DataFrame that can be used to find precontrast MRI
        sequences

    """

    # Read in csv as DataFrame
    fpath_mapping_df = pd.read_csv(
        filepath_filename_csv_path
    )
    # We only need the filename and the descriptive path
    fpath_mapping_df = fpath_mapping_df.loc[
        :, ['original_path_and_filename', 'descriptive_path']
    ]

    # We only need the precontrast sequences
    fpath_mapping_df = fpath_mapping_df.loc[
        fpath_mapping_df['original_path_and_filename'].str.contains('pre')
    ]

    # Only need the subject ID in first column
    fpath_mapping_df['original_path_and_filename'] = \
        fpath_mapping_df['original_path_and_filename'].str.split('/').str[1]
    # For second column, only need the name of the dir containing the sequence
    # Need to remove some extra slashes that are in there erroneously
    fpath_mapping_df['descriptive_path'] = \
        fpath_mapping_df['descriptive_path'].str.replace('//', '/')
    fpath_mapping_df['descriptive_path'] = \
        fpath_mapping_df['descriptive_path'].str.split('/').str[-2]

    # Drop duplicates so each subject has on entry
    fpath_mapping_df = fpath_mapping_df.drop_duplicates(
        subset='original_path_and_filename'
    )

    # Rename columns for better clarity
    fpath_mapping_df = fpath_mapping_df.rename(
        columns={
            'original_path_and_filename': 'subject_id',
            'descriptive_path': 'precontrast_dir'
        }
    )

    return fpath_mapping_df

def read_precontrast_mri(
    subject_id, 
    tcia_data_dir, 
    fpath_mapping_df
):
    """
    Reads in the precontrast MRI data given a subject ID. 
    This function also aligns the patient orientation so the patient's body
    is in the lower part of image. The slices from the beginning move inferior
    to superior. 


    Parameters
    ----------
    subject_id: str
        Subject_id (e.g. Breast_MRI_001)
    tcia_data_dir: str 
        Path of downloaded database from TCIA
    fpath_mapping_df: pd.DataFrame
        Cleaned mapping DataFrame that can be used to find precontrast MRI
        sequences

    Returns
    -------
    np.Array
        Raw MRI volume data read from all .dcm files
    pydicom.dataset.FileDataset
        Dicom data from final slice read. This is used for obtaining things
        such as pixel spacing, image orientation, etc. 

    """
    tcia_data_dir = Path(tcia_data_dir)

    # Get the sequence dir from the DataFrame
    sequence_dir = fpath_mapping_df.loc[
        fpath_mapping_df['subject_id'] == subject_id, 'precontrast_dir'
    ].iloc[0]

    # There's also a subdir for every subject that contains the sequences
    # There is only one of these
    sub_dir = os.listdir(tcia_data_dir / subject_id)[0]

    full_sequence_dir = tcia_data_dir / subject_id / sub_dir / sequence_dir

    # Now we can iterate through the files in the sequence dir and reach each
    # of them into a numpy array
    dicom_file_list = sorted(os.listdir(full_sequence_dir))
    dicom_data_list = []
    # Saving the values of first two image positions
    # This is used to orient inferior to superior
    first_image_position = 0
    second_image_position = 0

    for i in range(len(dicom_file_list)):
        dicom_data = pydicom.dcmread(full_sequence_dir / dicom_file_list[i])
        
        if i == 0:
            first_image_position = dicom_data[0x20, 0x32].value[-1]
        elif i == 1:
            second_image_position = dicom_data[0x20, 0x32].value[-1]

        dicom_data_list.append(dicom_data.pixel_array)
        
    # Stack in numpy array
    image_array = np.stack(dicom_data_list, axis=-1)

    # Rotate if inferior and superior are flipped
    if first_image_position > second_image_position:
        image_array = np.rot90(image_array, 2, (1, 2))

    # For patients in a certain orentation, also need to flip in another axis
    # This is the same in all dicom files so we can just use the last
    # dicom file that we have from the iteration. It also needs to be rounded.
    if round(dicom_data[0x20, 0x37].value[0], 0) == -1:
        image_array = np.rot90(image_array, 2)

    return image_array, dicom_data
def read_precontrast_mri_and_segmentation(
    subject_id,
    tcia_data_dir,
    fpath_mapping_df,
):
    """
    Reads and orients the precontrast MRI volume data for a given subject ID.

    Parameters
    ----------
    subject_id: str
        Subject ID (e.g., 'Breast_MRI_001')
    tcia_data_dir: str
        Path to the root TCIA DICOM dataset directory
    fpath_mapping_df: pd.DataFrame
        Mapping from subject_id to precontrast sequence directory

    Returns
    -------
    np.ndarray
        Raw MRI volume
    pydicom.dataset.FileDataset
        Metadata from last DICOM slice
    """
    tcia_data_dir = Path(tcia_data_dir)
    ## print("tcia_data_dir is ", tcia_data_dir)
    # Locate precontrast sequence folder
  
    # this part is still correct
    encounter_dir = tcia_data_dir / subject_id / os.listdir(tcia_data_dir / subject_id)[0]

# new step: get the list of sequence directories inside the encounter
    sequence_dir_list = os.listdir(encounter_dir)
   ## print(sequence_dir_list)
# now, loop through each sequence directory to find the files
    
    for sequence_name in sequence_dir_list:
        full_sequence_path = encounter_dir / sequence_name  # this is the final path to the files

        dicom_file_list = sorted(os.listdir(full_sequence_path))
        ## print(f"--- Found {len(dicom_file_list)} files in sequence: {sequence_name} ---")

        first_image_position = None
        second_image_position = None
        
        dicom_data_list = []
        for i, file_name in enumerate(dicom_file_list):
            dicom_path = full_sequence_path / file_name
            dicom_data = pydicom.dcmread(dicom_path)
            if i == 0:
                first_image_position = dicom_data.ImagePositionPatient[-1]
            elif i == 1:
                second_image_position = dicom_data.ImagePositionPatient[-1]
            dicom_data_list.append(dicom_data.pixel_array)
       
        
        image_array = np.stack(dicom_data_list, axis=-1)

        # Flip if slices are in reverse order
        if first_image_position > second_image_position:
            image_array = np.rot90(image_array, 2, (1, 2))

        # Flip for patient orientation
        if round(dicom_data.ImageOrientationPatient[0], 0) == -1:
            image_array = np.rot90(image_array, 2)

    return image_array, dicom_data


def normalize_image(image_array, min_cutoff = 0.001, max_cutoff = 0.001):
    """
    Normalize the intensity of an image array by cutting off min and max values 
    to a certain percentile and set all values above/below that percentile to 
    the new max/min. 

    Parameters
    ----------
    image_array: np.array
        3D numpy array constructed from dicom files
    min_cutoff: float
        Minimum percentile of image to keep. (0.1% = 0.001)
    max_cutoff: float
        Maximum percentile of image to keep. (0.1% = 0.001)

    Returns
    -------
    np.array
        Normalized image

    """

    # Sort image values
    sorted_array = np.sort(image_array.flatten())

    # Find %ile index and get values
    min_index = int(len(sorted_array) * min_cutoff)
    min_intensity = sorted_array[min_index]

    max_index = int(len(sorted_array) * min_cutoff) * -1
    max_intensity = sorted_array[max_index]

    # Normalize image and cutoff values
    image_array = (image_array - min_intensity) / \
        (max_intensity - min_intensity)
    image_array[image_array < 0.0] = 0.0
    image_array[image_array > 1.0] = 1.0

    return image_array

def zscore_image(image_array):
    """
    Convert intensity values in an image to zscores:
    zscore = (intensity_value - mean) / standard_deviation

    Parameters
    ----------
    image_array: np.array
        3D numpy array constructed from dicom files
    Returns
    -------
    np.array
        Image with zscores for values

    """

    image_array = (image_array - np.mean(image_array)) / np.std(image_array)

    return image_array