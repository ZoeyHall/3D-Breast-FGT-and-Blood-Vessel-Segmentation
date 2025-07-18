import numpy as np
import matplotlib.pyplot as plt

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

    # Corrected: max_index should use max_cutoff, not min_cutoff
    max_index = int(len(sorted_array) * (1 - max_cutoff))
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

# --- FIX: Define image_array before using it ---
# This is a placeholder. You would typically load your DICOM data here.
# For example, if you're using pydicom, it would look something like:
# import pydicom
# from pydicom.pixel_data_handlers.util import apply_voi_lut
#
# # Assuming 'dicom_folder' is the path to your DICOM series
# datasets = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in os.listdir(dicom_folder)]
# datasets.sort(key=lambda x: x.InstanceNumber)
# image_array = np.stack([apply_voi_lut(ds.pixel_array, ds) for ds in datasets])
#
# For now, let's create a dummy 3D array:
image_array = np.random.rand(100, 100, 100) * 1000 # Example 3D array

# Also assuming nrrd_breast_data and nrrd_dv_data are defined,
# let's create dummy ones for demonstration:
nrrd_breast_data = np.random.randint(0, 2, size=(100, 100, 100)) # Binary mask
nrrd_dv_data = np.random.randint(0, 2, size=(100, 100, 100)) # Binary mask
# --- End of FIX ---


# Apply the image processing functions
image_array = zscore_image(normalize_image(image_array))

# Plotting
plt.figure(figsize=(15, 5)) # Adjust figure size as needed

plt.subplot(1, 3, 1)
plt.title('MRI Volume')
plt.imshow(image_array[:, :, 50], cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Breast Mask')
plt.imshow(nrrd_breast_data[:, :, 50], cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('FGT + Blood Vessel Mask')
plt.imshow(nrrd_dv_data[:, :, 50], cmap='gray')
plt.axis('off')

plt.show()
