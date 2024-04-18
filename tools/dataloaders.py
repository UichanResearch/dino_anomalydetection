import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
import random
import copy

class CheXpert(Dataset):
    def __init__(self, root = "data", mode = 'train', img_size=(224, 224), normalize=False, enable_transform=True, full=True):

        self.data = []
        self.mode = mode
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full

        if self.mode == 'train':
            if enable_transform:
                self.transforms = transforms.Compose([
                    transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                    transforms.ToTensor()
                ])
            else:
                self.transforms = transforms.ToTensor()

        else:
            self.transforms = transforms.ToTensor()

        self.load_data()

    def load_data(self):
        #train
        if self.mode == 'train':
            items = os.listdir(os.path.join(self.root, 'chexpert/train_256/normal_256'))
            for item in items:
                self.data.append((os.path.join(self.root, 'chexpert/train_256/normal_256', item), 0))

        #val
        elif self.mode == 'val':
            items = os.listdir(os.path.join(self.root, 'chexpert/val_256/normal_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'chexpert/val_256/normal_256', item), 0))

            items = os.listdir(os.path.join(self.root, 'chexpert/val_256/abnormal_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'chexpert/val_256/abnormal_256', item), 1))
        
        #normal
        elif self.mode == 'val_normal':
            items = os.listdir(os.path.join(self.root, 'chexpert/val_256/normal_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'chexpert/val_256/normal_256', item), 0))
        
        #abnormal
        elif self.mode == 'val_abnormal':
            items = os.listdir(os.path.join(self.root, 'chexpert/val_256/abnormal_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'chexpert/val_256/abnormal_256', item), 1))
        #test
        elif self.mode == 'test_normal':
            items = os.listdir(os.path.join(self.root, 'chexpert/test_256/normal_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'chexpert/test_256/normal_256', item), 0))

        elif self.mode == 'test_abnormal':
            items = os.listdir(os.path.join(self.root, 'chexpert/test_256/abnormal_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'chexpert/test_256/abnormal_256', item), 1))
        
        print('Data:', "CheXpert")
        print('Data len: ', len(self.data))

    def __getitem__(self, index):
        img, label = copy.deepcopy(self.data[index])
        img = Image.open(img).resize(self.img_size)
        img = img.convert("L")
        img = self.transforms(img)
        img = img.repeat(3, 1, 1)
        if self.normalize:
            img -= self.mean
            img /= self.std
        return img, (torch.zeros((1,)) + label).long()

    def __len__(self):
        return len(self.data)

class zhanglab(Dataset):
    def __init__(self, root = "data", mode = 'train', img_size=(224, 224), normalize=False, enable_transform=True, full=True):

        self.data = []
        self.mode = mode
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full

        if self.mode == 'train':
            if enable_transform:
                self.transforms = transforms.Compose([
                    transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95,1.05)),
                    transforms.ToTensor()
                ])
            else:
                self.transforms = transforms.ToTensor()
        else:
            self.transforms = transforms.ToTensor()

        self.load_data()

    def load_data(self):
        #train
        if self.mode == 'train':
            items = os.listdir(os.path.join(self.root, 'zhanglab/train/normal_256'))
            for item in items:
                self.data.append((os.path.join(self.root, 'zhanglab/train/normal_256', item), 0))

        #val
        elif self.mode == 'val':
            items = os.listdir(os.path.join(self.root, 'zhanglab/val/normal_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'zhanglab/val/normal_256', item), 0))

            items = os.listdir(os.path.join(self.root, 'zhanglab/val/pneumonia_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'zhanglab/val/pneumonia_256', item), 1))

        elif self.mode == 'val_normal':
            items = os.listdir(os.path.join(self.root, 'zhanglab/val/normal_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'zhanglab/val/normal_256', item), 0))

        elif self.mode == 'val_abnormal':
            items = os.listdir(os.path.join(self.root, 'zhanglab/val/pneumonia_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'zhanglab/val/pneumonia_256', item), 1))

        #test
        elif self.mode == 'test_normal':
            items = os.listdir(os.path.join(self.root, 'zhanglab/test/normal_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'zhanglab/test/normal_256', item), 0))

        elif self.mode == 'test_abnormal':
            items = os.listdir(os.path.join(self.root, 'zhanglab/test/pneumonia_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'zhanglab/test/pneumonia_256', item), 1))
        
        print('Data:', "zhanglab")
        print('Data len: ', len(self.data))


    def __getitem__(self, index):
        img, label = copy.deepcopy(self.data[index])
        img = Image.open(img)
        img = img.convert("L")
        img = img.resize(self.img_size)
        img = self.transforms(img)
        img = img.repeat(3, 1, 1)
        if self.normalize:
            img -= self.mean
            img /= self.std
        return img, (torch.zeros((1,)) + label).long()

    def __len__(self):
        return len(self.data)

class digit_local(Dataset):
    def __init__(self, root = "data", mode = 'train', img_size=(224, 224), normalize=True, enable_transform=True, full=True):

        self.data = []
        self.mode = mode
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.full = full

        if self.mode == 'train':
            if enable_transform:
                self.transforms = transforms.Compose([
                    transforms.ToTensor()
                ])
            else:
                self.transforms = transforms.ToTensor()
        else:
            self.transforms = transforms.ToTensor()

        self.load_data()

    def load_data(self):
        #train
        if self.mode == 'train':
            items = os.listdir(os.path.join(self.root, 'digit_local/train/normal/img'))
            for item in items:
                self.data.append((os.path.join(self.root, 'digit_local/train/normal/img', item), 0))

        #val
        elif self.mode == 'val':
            items = os.listdir(os.path.join(self.root, 'digit_local/val/normal/img'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'digit_local/val/normal/img', item), 0))

            items = os.listdir(os.path.join(self.root, 'digit_local/val/abnormal/img'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'digit_local/val/abnormal/img', item), 1))

        elif self.mode == 'val_normal':
            items = os.listdir(os.path.join(self.root, 'digit_local/val/normal/img'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'digit_local/val/normal/img', item), 0))

        elif self.mode == 'val_abnormal':
            items = os.listdir(os.path.join(self.root, 'digit_local/val/abnormal/img'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'digit_local/val/abnormal/img', item), 1))

        #test
        elif self.mode == 'test_normal':
            items = os.listdir(os.path.join(self.root, 'digit_local/test/normal/img'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'digit_local/test/normal/img', item), 0))

        elif self.mode == 'test_abnormal':
            items = os.listdir(os.path.join(self.root, 'digit_local/test/abnormal/img'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((os.path.join(self.root, 'digit_local/test/abnormal/img', item), 1))
        
        print('Data:', "zhanglab")
        print('Data len: ', len(self.data))


    def __getitem__(self, index):
        img, label = copy.deepcopy(self.data[index])
        img = Image.open(img)
        img = img.convert("L")
        img = img.resize(self.img_size)
        img = self.transforms(img)
        img = img.repeat(3, 1, 1)
        if self.normalize:
            img /= img.max()
        return img, (torch.zeros((1,)) + label).long()

    def __len__(self):
        return len(self.data)
    

def load_data(dataset = "chexpert", type = "test_normal"):
    if dataset == "chexpert":
        return CheXpert(mode = type)
    elif dataset == "zhang":
        return zhanglab(mode = type)
    elif dataset == "digit_local":
        return digit_local(mode = type)


if __name__ == "__main__":
    data = zhanglab(mode = "train")
    data[0]