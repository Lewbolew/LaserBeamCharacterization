import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms
import numpy as np
import os
import operator
from skimage import io
from skimage.transform import resize
import random
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    ShiftScaleRotate,
    RandomBrightness,
    Blur
)

random.seed(42)

class LaserBeamData(Dataset):

    def __init__(self, root_dir, partition, augment=True):
        self.root_dir = root_dir
        self.list_IDs = os.listdir(os.path.join(self.root_dir, 'x_{}'.format(partition)))
        self.partition = partition
        self.augment = augment
        self.augmentator = Compose([
                    # Non destructive transformations
                        VerticalFlip(p=0.6),
                        HorizontalFlip(p=0.6),
                        RandomRotate90(),
                        Transpose(p=0.6),
                        ShiftScaleRotate(p=0.45, scale_limit=(0.1, 0.3)),

                    #     # Non-rigid transformations
                        ElasticTransform(p=0.25, alpha=160, sigma=180 * 0.05, alpha_affine=120 * 0.03),

                        Blur(blur_limit=3, p=0.2),

                    #     Color augmentation
                        RandomBrightness(p=0.5),
                        RandomContrast(p=0.5),
                        RandomGamma(p=0.5),
                        CLAHE(p=0.5)
                        ]
                    )

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]


        img_path = os.path.join(self.root_dir, 'x_{}'.format(self.partition), self.list_IDs[index])
        mask_path = os.path.join(self.root_dir, 'y_{}'.format(self.partition), self.list_IDs[index])

        # Reading
        X = io.imread(img_path)
        y = io.imread(mask_path)

        # Augmentation
        if self.augment: 
            augmented = self.augmentator(image=X, mask=y)
            X = augmented['image']
            y = augmented['mask']

        # Preprocessing
        X = resize(X, (512,512))

        y = 255 - y
        y[y==255] = 1
        y = (resize(y, (512,512)) * 255).astype(np.uint8)

        # To Tensor
        to_tensor = transforms.ToTensor()
        X = to_tensor(X).float()
        y = torch.from_numpy(y).long()


        return X, y

class LaserBeamOnlyX(LaserBeamData):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_IDs = os.listdir(os.path.join(self.root_dir))

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        img_path = os.path.join(self.root_dir, self.list_IDs[index])

        X = io.imread(img_path)

        # Preprocessing
        X = self.cropND(X, (2048, 2048))
        X = resize(X, (1024,1024))

        to_tensor = transforms.ToTensor()
        X = to_tensor(X).float()

        return X

    def cropND(self, img, bounding):
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]