from typing import Tuple
import os
import numpy as np
import cv2
import torch
import torchvision
from osgeo import gdal
from tqdm import tqdm


gdal.DontUseExceptions()


def read_gdal(filename, bands, view=None):
    ds = gdal.Open(filename, gdal.GA_ReadOnly)

    if view is not None:
        start_h, start_w, end_h, end_w = view
        arr = ds.ReadAsArray(start_w, start_h, end_w - start_w, end_h - start_h)
    else:
        arr = ds.ReadAsArray()

    arr = arr[bands]
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255
    arr = arr.transpose(1, 2, 0).astype(np.uint8)
    return arr


def filter_valid_views(hr_path, sr_path, views):
    """
    go through all the views and check if the views are valid

    invalid: the view contains all zeros elements, e.g. (0, 0, 0) or (0, 0, 0, 0)
    """

    hr_ds = gdal.Open(hr_path, gdal.GA_ReadOnly)
    sr_ds = gdal.Open(sr_path, gdal.GA_ReadOnly)

    def _valid(view):
        """
        check if the view is valid, return True/False

        NOTE: ReadAsArray must be called inside this wrapper function,
            otherwise, the memory will not be released.
            The mechanism of gdal is not clear. Seems like the memory can only
            be released when funcation returns.
        """
        start_h, start_w, end_h, end_w = view
        hr_data = hr_ds.ReadAsArray(start_w, start_h, end_w - start_w, end_h - start_h)
        sr_data = sr_ds.ReadAsArray(start_w, start_h, end_w - start_w, end_h - start_h)

        return not (
            (hr_data == 0).all(axis=0).any() or
            (sr_data == 0).all(axis=0).any()
        )

    valids = [_valid(view) for view in tqdm(views, desc='Filtering valid views')]

    del hr_ds, sr_ds
    return views[valids]


totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img


class LRHRDataset:
    def __init__(self,
                 dataroot,
                 l_resolution=16,
                 r_resolution=128,
                 split='train',
                 split_train_ratio=1.0,
                 seed=0,
                 data_len=-1):
        """"
        Args:
            dataroot (str): the path of the images and bands order
                hr.tif:1234;sr.tif:3214
        """
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.split = split

        if ';' in dataroot:
            hr, sr = dataroot.split(';')
        else:
            hr = dataroot
            sr = hr

        self.hr_path, self.hr_bands = hr.split(':')
        self.sr_path, self.sr_bands = sr.split(':')

        # when the sr is the same as hr, create lr on the fly
        self.create_lr = self.sr_path == self.hr_path

        self.hr_bands = [int(b) - 1 for b in self.hr_bands]
        self.sr_bands = [int(b) - 1 for b in self.sr_bands]

        ds = gdal.Open(self.hr_path, gdal.GA_ReadOnly)
        image_size = (ds.RasterYSize, ds.RasterXSize)
        del ds

        patch_size = (r_resolution, r_resolution)

        # determine if the valid views are stored
        self.valid_views_path = os.path.join(
            os.path.dirname(self.hr_path),
            os.path.basename(self.hr_path) + os.path.basename(self.sr_path) + '_valid_views.npy'
        )

        if os.path.exists(self.valid_views_path):
            # load the valid views
            self.views = np.load(self.valid_views_path)
        else:
            # generate the valid views
            self.views = self.get_views(
                image_size, patch_size, patch_size
            )
            self.views = filter_valid_views(self.hr_path, self.sr_path, self.views)
            np.save(self.valid_views_path, self.views)

        self.views = self.views[:self.data_len]

        # split the dataset
        # if split_train_ratio < 1, split the dataset into train and val
        # otherwise, use the whole dataset for both train and val
        if split_train_ratio < 1:
            # shuffle the views
            np.random.seed(seed)
            np.random.shuffle(self.views)

            if split == 'train':
                self.views = self.views[:int(len(self.views) * split_train_ratio)]
            elif split == 'val':
                self.views = self.views[int(len(self.views) * split_train_ratio):]

    def get_views(self,
                  image_size: Tuple[int, int],
                  patch_size: Tuple[int, int],
                  stride: Tuple[int, int]):
        """
        Generate views from a CHW image.

        :param image_size: image size, (H, W)
        :param patch_size: patch size, (H, W)
        :param stride: stride, (H, W)
        :return: index of the patches, shape (N, start_h, start_w, end_h, end_w)
        """

        H, W = image_size
        patch_h, patch_w = patch_size
        stride_h, stride_w = stride

        # Calculate the number of patches in each dimension
        n_h = (H - patch_h) // stride_h + 1
        n_w = (W - patch_w) // stride_w + 1

        # Generate the views
        views = []
        for i in range(n_h):
            for j in range(n_w):
                start_h = i * stride_h
                start_w = j * stride_w
                end_h = start_h + patch_h
                end_w = start_w + patch_w
                views.append((start_h, start_w, end_h, end_w))

        return np.array(views)

    def __len__(self):
        return len(self.views)

    def __getitem__(self, index):
        view = self.views[index]
        img_HR = read_gdal(self.hr_path, self.hr_bands, view)
        img_SR = read_gdal(self.sr_path, self.sr_bands, view)

        if self.create_lr:
            # create lr on the fly
            # downsample the sr image to lr then upsample to sr
            # use average pooling to downsample, use bicubic to upsample
            img_LR = cv2.resize(img_SR, (self.l_res, self.l_res), interpolation=cv2.INTER_AREA)
            img_SR = cv2.resize(img_LR, (self.r_res, self.r_res), interpolation=cv2.INTER_CUBIC)

        [img_SR, img_HR] = transform_augment(
            [img_SR, img_HR], split=self.split, min_max=(-1, 1)
        )
        return {'HR': img_HR, 'SR': img_SR, 'Index': index}

    def get_geo_ref(self, index):
        """
        Get the georeference information of the image
        """
        ds = gdal.Open(self.hr_path, gdal.GA_ReadOnly)
        transform = list(ds.GetGeoTransform())
        projection = ds.GetProjection()
        del ds

        view = self.views[index]
        # derive the offset
        start_h, start_w, _, _ = view
        transform = list(transform)
        transform[0] += start_w * transform[1]
        transform[3] += start_h * transform[5]

        return transform, projection
