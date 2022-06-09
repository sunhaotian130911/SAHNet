import os
import cv2
import scipy.io as scio
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import glob
from skimage import transform

from config import DefaultConfig
opt = DefaultConfig()
from matplotlib import pyplot as plt


LABELS = ['Benign','Malignant']


class MCBenMal(data.Dataset):

    def __init__(self, root, transforms=None, train=False, validation=False, test=False):
        # ./dataset/train/Benign(Malignant)/filename.mat
        if train:
            imgs = {name: index for index in range(len(LABELS)) for name in
                    glob.glob(root + LABELS[index] + '/*.mat')}
        elif validation:
            imgs = {name: index for index in range(len(LABELS)) for name in
                    glob.glob(root + LABELS[index] + '/*.mat')}
        elif test:
            imgs = {name: index for index in range(len(LABELS)) for name in
                    glob.glob(root + LABELS[index] + '/*.mat')}
        else:
            print('Error:Not input data set')

        imgs_num = len(imgs)
        self.root = root
        self.train = train
        self.validation = validation
        self.test = test
        self.imgs = imgs
        self.names = list(sorted(imgs.keys()))
        self.img_num = imgs_num

    def __getitem__(self, index):
        label = self.imgs[self.names[index]]
        img_path = self.names[index]
        temp = scio.loadmat(img_path)
        if self.train:
            img = np.array(temp.get('cropped_image'))  # Patch cropped_image('cropped_image'))  #
        else:
            img = np.array(temp.get('cropped_image'))  # cropped_image
        patch = img
        patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
        # crop to 256*256*8
        crop_size = 256
        patch = patch[patch.shape[0]//2-crop_size//2:patch.shape[0]//2+crop_size//2,
                      patch.shape[1] // 2 - crop_size // 2:patch.shape[1] // 2 + crop_size // 2,:]
        if opt.debug:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(img[...,3],cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(patch[...,3],cmap='gray')
            plt.show()

        patch = T.ToTensor()(patch)
        if self.test:
            return patch,label,img_path
        else:
            return patch, label
    def __len__(self):
        return len(self.imgs)