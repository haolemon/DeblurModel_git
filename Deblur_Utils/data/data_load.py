import os
import cv2
import lmdb
import torch
import numpy as np
from PIL import Image as Image
from Deblur_Utils.data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor, PairCenterCrop
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, expend_scale=1):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))
        self.label_list = os.listdir(os.path.join(image_dir, 'sharp/'))

        self._check_image(self.image_list)
        self._check_image(self.label_list)
        self.image_list.sort()
        self.label_list.sort()
        # 扩展数据集
        self.image_list_copy = self.image_list.copy()
        self.label_list_copy = self.label_list.copy()
        for _ in range(expend_scale - 1):
            self.image_list.extend(self.image_list_copy)
            self.label_list.extend(self.label_list_copy)

        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.label_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'PNG']:
                raise ValueError


class Val_DeblurDataset(Dataset):
    def __init__(self, image_dir, crop_size=256, is_crop=True):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.crop = transforms.CenterCrop(crop_size)
        self.is_crop = is_crop

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))

        if self.is_crop:
            image = self.crop(image)
            label = self.crop(label)
        image = F.to_tensor(image)
        label = F.to_tensor(label)
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


class SSID_LMDB_Read(Dataset):
    def __init__(self, img_dir):
        self.image_dir = img_dir
        self.gt_env = lmdb.open(os.path.join(img_dir, 'gt'))
        self.inp_env = lmdb.open(os.path.join(img_dir, 'inp'))

        self.gt_txn = self.gt_env.begin()
        self.inp_txn = self.inp_env.begin()

        self.gt_key, self.inp_key = [], []
        for key, _ in self.gt_txn.cursor():
            self.gt_key.append(key)
        for key, _ in self.inp_txn.cursor():
            self.inp_key.append(key)
        self.gt_key.sort(), self.inp_key.sort()

    def __len__(self):
        return len(self.inp_key)

    def __getitem__(self, idx):
        inp = self.inp_txn.get(self.inp_key[idx])
        gt = self.gt_txn.get(self.gt_key[idx])

        inp_np = np.frombuffer(inp, np.uint8)
        gt_np = np.frombuffer(gt, np.uint8)

        inp_cv = cv2.imdecode(inp_np, cv2.IMREAD_COLOR)
        gt_cv = cv2.imdecode(gt_np, cv2.IMREAD_COLOR)

        inp_tensor = torch.from_numpy(np.transpose(inp_cv, (2, 0, 1)).astype(np.float32) / 255.)
        gt_tensor = torch.from_numpy(np.transpose(gt_cv, (2, 0, 1)).astype(np.float32) / 255.)

        return inp_tensor, gt_tensor


class RealBlurDataset(Dataset):
    def __init__(self, data_dir, txt_path, transform=None, expend_scale=1):
        self.image_list = []
        self.label_list = []

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label_path, image_path = line.strip().split()

                image_path = os.path.join(data_dir, image_path)
                label_path = os.path.join(data_dir, label_path)

                self.image_list.append(image_path)
                self.label_list.append(label_path)

        # 扩展数据集
        self.image_list_copy = self.image_list.copy()
        self.label_list_copy = self.label_list.copy()
        for _ in range(expend_scale - 1):
            self.image_list.extend(self.image_list_copy)
            self.label_list.extend(self.label_list_copy)

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label = Image.open(self.label_list[idx])

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        return image, label
