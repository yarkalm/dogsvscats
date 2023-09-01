import cv2
import copy
import matplotlib.pyplot as plt
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision.transforms as transforms

# Data augmentation and normalisation for training

base_augmentations = A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))





train_dataset = torchvision.datasets.ImageFolder(root='./dataset/train', transform=Transforms(transforms=A.Compose(base_augmentations)))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

val_dataset = torchvision.datasets.ImageFolder(root='./dataset/validation', transform=Transforms(transforms=A.Compose(base_augmentations)))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

