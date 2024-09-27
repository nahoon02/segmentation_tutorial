import os
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from dataloader.util import get_ct_image_tensor, get_mask_tensor, horizontal_flip, save_image_from_tensor
from typing import Optional, Tuple, List


class TrainDataset(Dataset):
    """ custom dataset for 2d segmentation
    """

    def __init__(self, csv_file_path: str, ct_data_dir: str, mask_data_dir: str,
                 abnormal_ratio: Optional[int] = None, augmentation: bool = False, data_type = 'train') -> None:
        """
        Args:
            csv_file_path: csv file path
            ct_data_dir: ct image data root directory
            mask_data_dir: mask image data root directory
            abnormal_ratio: increase the number of train abnormal slices.
                            the total number of [train abnormal slice]  = number of [normal_slice] X [abnormal_ratio]
                            default = None means abnormal_ratio value does not be used
            augmentation: horizontal flip or not
            data_type: train or valid or test
        """
        super(TrainDataset, self).__init__()

        """  open csv file   """
        df = pd.read_csv(csv_file_path)

        self.ct_data_dir = ct_data_dir
        self.mask_data_dir = mask_data_dir
        self.augmentation = augmentation
        self.resize = None

        """ select dataset """
        if data_type == 'train':
            df = df[df['type'] == 'train']
        elif data_type == 'valid':
            df = df[df['type'] == 'valid']
        elif data_type == 'test':
            df = df[df['type'] == 'test']
        else:
            raise RuntimeError(f'data_type=[{data_type}] does not be supported.')

        abnormal_df = df[df['abnormal'] == 1]
        normal_df = df[df['abnormal'] == 0]

        num_abnormal = len(abnormal_df)
        num_normal = len(normal_df)

        """ augmentation """
        if abnormal_ratio is not None and abnormal_ratio > 0.0 and data_type == 'train':
            num_aug_abnormal = int(num_normal * abnormal_ratio) - num_abnormal
        else:
            num_aug_abnormal = 0

        if num_aug_abnormal > 0:
            new_abnormal_df = abnormal_df.sample(n=num_aug_abnormal, replace=False)

            abnormal_df = pd.concat([abnormal_df, new_abnormal_df], axis=0)
            df = pd.concat([abnormal_df, normal_df])

        """ reset index """
        df = df.reset_index(drop=True)

        """ get a dataframe """
        self.df = df
        self.num_data = len(df)
        self.num_normal_data = num_normal
        self.num_abnormal_data = num_abnormal
        self.num_aug_abnormal_data = num_aug_abnormal

    def __getitem__(self, index):

        row = self.df.loc[index, :]

        filename = row['filename']
        label = row['abnormal']

        ct_png_path = os.path.join(self.ct_data_dir, filename)
        mask_png_path = os.path.join(self.mask_data_dir, filename)

        if not os.path.isfile(ct_png_path):
            raise RuntimeError(f'ct image file=[{ct_png_path}] does not exist!')

        if not os.path.isfile(mask_png_path):
            raise RuntimeError(f'mask image file=[{mask_png_path}] does not exist')

        """
            for ct image
            (1) resize the ct image if needed         
            (2) make 3d tensor : (height, width) --> (1, resize_h, resize_w)
        """
        ct_tensor, orginal_size = get_ct_image_tensor(ct_png_path, self.resize)

        """
            for make image
            (1) resize the make image if needed       
            (2) make 3d tensor : (height, width) --> (1, resized_h, resized_w)
        """
        mask_tensor = get_mask_tensor(mask_png_path, self.resize, orginal_size)

        assert ct_tensor.shape == mask_tensor.shape

        """
            Augmentation
        """
        if self.augmentation:
            if np.random.rand() <= 0.5:
                ct_tensor = horizontal_flip(ct_tensor)
                mask_tensor = horizontal_flip(mask_tensor)

        """
            ct_tensor shape: 3d tensor (1, resized_h, resized_w)
            mask_tensor shape: 3d tensor (1, resized_h, resized_w)
            label : integer
            dcm_png_path: string
            mask_png_path: string
        """
        return ct_tensor, mask_tensor, label, ct_png_path, mask_png_path


    def __len__(self):
        return self.num_data

    def get_info(self):
        normal_slice_info = f'# of normal slices = {self.num_normal_data:,}\n'
        abnormal_slice_info = f'# of abnormal slices = {self.num_abnormal_data:,}\n'
        augmented_abnormal_slice = f'# of augmented abnormal slice = {self.num_aug_abnormal_data:,}\n'
        total_slices = f'# of total slices = {self.num_data:,}\n'
        data_ratio = f'# abnormal:normal = {1.0}:' \
                      f'{self.num_normal_data/(self.num_abnormal_data + self.num_aug_abnormal_data):.2f} \n'

        return normal_slice_info + abnormal_slice_info + augmented_abnormal_slice + total_slices + data_ratio


if __name__ == '__main__':

    import torchvision
    import os

    csv_path = '../dataset/dataset.csv'
    ct_data_dir = '../dataset/dataset_256/ct'
    mask_data_dir = '../dataset/dataset_256/mask'
    image_idx = 0

    train_dataset = TrainDataset(csv_path, ct_data_dir=ct_data_dir, mask_data_dir=mask_data_dir,
                                 abnormal_ratio = None, data_type='train')
    info = train_dataset.get_info()
    print(info)

    ct_tensor, mask_tensor, label, ct_png_path, mask_png_path = train_dataset.__getitem__(image_idx)
    print(f'ct_tensor = {ct_tensor.shape}')
    print(f'mask_tensor = {mask_tensor.shape}')
    print(f'label = {label}')
    print(f'ct png path = {ct_png_path}')
    print(f'mask png path = {mask_png_path}')

    save_image_from_tensor(ct_tensor, f'../dataset/{image_idx}_ct.png')
    save_image_from_tensor(mask_tensor, f'../dataset/{image_idx}_mask.png')
