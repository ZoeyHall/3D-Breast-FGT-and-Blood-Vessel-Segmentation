import argparse
import numpy as np
import torch
import nrrd
from pathlib import Path
import torchio as tio
from preprocessing import read_precontrast_mri_and_segmentation 


# Performs predictions using a trained model.
# Predictions are performed the same method we used and are saved to a
# target directory. 

subject_id = "Breast_MRI_827"
tcia_data_dir = "/net/projects/annawoodard/tcia_datasets/duke_breast_cancer_mri/dicom"

volume, metadata = read_precontrast_mri_and_segmentation(subject_id, tcia_data_dir)
print("Volume shape:", volume.shape)



def get_args():
    parser = argparse.ArgumentParser(
        description='Predict UNet Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-c', '--target-tissue', metavar='C', type=str,
        help='Target tissue, either breast or dv', dest='target_tissue'
    )

    parser.add_argument(
        '-s', '--save-masks-dir', metavar='S', type=str,
        help='Directory to save masks', dest='save_masks_dir'
    )
    parser.add_argument(
        '-p', '--model-save-path', metavar='P', type=str,
        help='Path to saved model', dest='model_save_path'
    )

    return parser.parse_args()


if __name__ == '__main__':
    from dataset_3d import Dataset3DSimple, Dataset3DDivided
    from model_utils import pred_and_save_masks_3d_simple, pred_and_save_masks_3d_divided
    from unet import UNet3D

    args = get_args()

    if args.target_tissue == 'breast':
        n_channels = 1
        n_classes = 1
    elif args.target_tissue == 'dv':
        n_channels = 2
        n_classes = 3
    else:
        raise ValueError('Target tissue must be either "breast" or "dv"')

    unet = UNet3D(
        in_channels=n_channels,
        out_classes=n_classes,
        num_encoding_blocks=3,
        padding=True,
        normalization='batch'
    )

    # 🔧 HARD-CODED PATHS — change these to your test subject
    image_path = Path("/home/zoeyh/3D-Breast-FGT-and-Blood-Vessel-Segmentation/Duke-Breast-Cancer-MRI-Supplement-v3/Segmentation_Masks_NRRD/Breast_MRI_827/Segmentation_Breast_MRI_827_Breast.seg.nrrd")  # MRI volume
    mask_path  = Path("/home/zoeyh/3D-Breast-FGT-and-Blood-Vessel-Segmentation/Duke-Breast-Cancer-MRI-Supplement-v3/Segmentation_Masks_NRRD/Breast_MRI_827/Segmentation_Breast_MRI_827_Dense_and_Vessels.seg.nrrd")  # For dv case only

    if args.target_tissue == 'breast':
        input_dim = (144, 144, 96)

        image_array, _ = nrrd.read(image_path)
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).float()

        transforms = tio.Compose([tio.Resize(input_dim)])
        image_tensor = transforms({'image': image_tensor}, keys=['image'])['image']

        class SingleVolumeDataset(torch.utils.data.Dataset):
            def __init__(self, image_tensor):
                self.image_tensor = image_tensor

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {'image': self.image_tensor}

        dataset = SingleVolumeDataset(image_tensor)

        pred_and_save_masks_3d_simple(
            unet=unet,
            saved_model_path=args.model_save_path,
            dataset=dataset,
            n_classes=n_classes,
            n_channels=n_channels,
            save_masks_dir=args.save_masks_dir
        )

    else:
        x_y_divisions = 8
        z_division = 3

        # Read both image and additional input mask (e.g., breast mask)
        image_array, _ = nrrd.read(image_path)
        additional_input_array, _ = nrrd.read(mask_path)

        class SingleDividedDataset(torch.utils.data.Dataset):
            def __init__(self, image, additional_input, input_dim, x_y_divs, z_divs):
                self.image = image
                self.additional_input = additional_input
                self.input_dim = input_dim
                self.x_y_divs = x_y_divs
                self.z_divs = z_divs

                self.shape = image.shape
                self.indices = []

                x_len, y_len, z_len = self.shape
                x_step = (x_len - input_dim) // (x_y_divs - 1)
                y_step = (y_len - input_dim) // (x_y_divs - 1)
                z_step = (z_len - input_dim) // (z_divs - 1)

                for z in range(z_divs):
                    for y in range(x_y_divs):
                        for x in range(x_y_divs):
                            xi = x * x_step if x < x_y_divs - 1 else x_len - input_dim
                            yi = y * y_step if y < x_y_divs - 1 else y_len - input_dim
                            zi = z * z_step if z < z_divs - 1 else z_len - input_dim
                            self.indices.append((xi, yi, zi))

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                x, y, z = self.indices[idx]
                img_crop = self.image[x:x+self.input_dim, y:y+self.input_dim, z:z+self.input_dim]
                add_crop = self.additional_input[x:x+self.input_dim, y:y+self.input_dim, z:z+self.input_dim]
                img = torch.from_numpy(img_crop).unsqueeze(0).float()
                add = torch.from_numpy(add_crop).unsqueeze(0).float()
                return {'image': torch.cat([img, add], dim=0)}

        dataset = SingleDividedDataset(
            image=image_array,
            additional_input=additional_input_array,
            input_dim=96,
            x_y_divs=x_y_divisions,
            z_divs=z_division
        )

        pred_and_save_masks_3d_divided(
            unet,
            args.model_save_path,
            dataset,
            n_classes,
            args.save_masks_dir
        )
