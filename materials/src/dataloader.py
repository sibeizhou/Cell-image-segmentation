import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random

from PIL import Image, ImageOps

#import any other libraries you need below this line
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2


def histogram_equalization(img):
    img = np.array(img)
    equalized_img = cv2.equalizeHist(img)
    return Image.fromarray(equalized_img)

class Cell_data(Dataset):
    def __init__(self, data_dir, size, train='True', train_test_split=0.8, augment_data=True):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        # initialize the data class
        self.data_dir = data_dir
        self.size = size
        self.train = train
        self.train_test_split = train_test_split
        self.augment_data = augment_data

        self.image_paths = sorted(
            [os.path.join(data_dir, 'scans', file_name) for file_name in os.listdir(os.path.join(data_dir, 'scans'))])
        self.mask_paths = sorted(
            [os.path.join(data_dir, 'labels', file_name) for file_name in os.listdir(os.path.join(data_dir, 'labels'))])

        split_idx = int(len(self.image_paths) * train_test_split)

        if self.train:
            self.image_paths = self.image_paths[:split_idx]
            self.mask_paths = self.mask_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
            self.mask_paths = self.mask_paths[split_idx:]

        # Define basic transformations (resize and normalize)
        self.image_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

        self.label_transform = transforms.Compose([
            transforms.Resize((size, size)),
            lambda img: torch.from_numpy(np.array(img)).unsqueeze(0)
        ])

    def __getitem__(self, idx):
        # load image and mask from index idx of your data
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('L')  # Grayscale image (1 channel)
        mask = Image.open(mask_path)

        # apply histogram equalization for data balancing
        image = histogram_equalization(image)

        # Apply basic transformations
        image = self.image_transform(image)
        mask = self.label_transform(mask)

        # data augmentation part
        if self.augment_data and self.train:
            augment_mode = np.random.randint(0, 6)
            if augment_mode == 0:
                # flip image vertically
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            elif augment_mode == 1:
                # flip image horizontally
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            elif augment_mode == 2:
                # zoom image
                crop_transform = transforms.Compose([
                    transforms.CenterCrop(int(self.size * 0.8)),
                    transforms.Resize((self.size, self.size))
                ])
                image = crop_transform(image)
                mask = crop_transform(mask)
            elif augment_mode == 3:
                # rotate image
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            elif augment_mode == 4:
                # Gamma correction
                gamma = random.uniform(0.8, 1.2)
                image = TF.adjust_gamma(image, gamma=gamma)
            else:
                # Non-rigid transformation
                elastic_transformer = transforms.ElasticTransform(alpha=10.0, sigma=10.0)
                image = elastic_transformer(image)
                mask = elastic_transformer(mask)

        # return image and mask in tensors
        return image.squeeze(0), mask.squeeze(0)

    def __len__(self):
        return len(self.image_paths)

