import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2

class RealPastedGlasses(Dataset):
    def __init__(self, real_root, pasted_root, transform = None):
        self.real_root = real_root
        self.pasted_root = pasted_root

        self.real = os.listdir(real_root)
        self.pasted = os.listdir(pasted_root)
        self.len_real = len(self.real)
        self.len_pasted = len(self.pasted)
        self.transform = transform

    def __len__(self):
        return self.len_real + self.len_pasted

    def __getitem__(self, i):
        if i < self.len_real:
            img = np.array(Image.open(os.path.join(self.real_root, self.real[i])))
            mask = np.ones((img.shape[0], img.shape[1]))
            cls = torch.FloatTensor([1, 0])
        else:
            img = np.array(Image.open(os.path.join(self.pasted_root, self.pasted[i - self.len_real])))
            mask = np.mean(img[:,144:,:],axis=2)
            img = img[:,:144,:]
            cls = torch.FloatTensor([0, 1])

        mask[mask == 255.0] = 1.0
        album = self.transform(image=img, mask=mask)
        return album["image"], album["mask"].unsqueeze(0), cls

class RealGlasses(Dataset):
    def __init__(self, real_root, transform):
        self.root = real_root
        self.real = os.listdir(real_root)
        self.transform = transform

    def __len__(self):
        return len(self.real)

    def __getitem__(self, i):
        img = np.array(Image.open(os.path.join(self.root, self.real[i])))
        album = self.transform(image=img)
        target_cls = torch.FloatTensor([0, 1])
        # target_cls = torch.FloatTensor([1, 0])
        return album["image"], target_cls


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from utils import *
    from torch.utils.data import WeightedRandomSampler
    class_weight = [1 / len(os.listdir("E:/finalyrs_project/real_imgs")), 1 / len(os.listdir("E:/finalyrs_project/DatasetAugmentation/pasted"))]
    dataset = RealPastedGlasses("E:/finalyrs_project/real_imgs","E:/finalyrs_project/DatasetAugmentation/pasted", transform=transform)
    real_weight = [1 / len(os.listdir("E:/finalyrs_project/real_imgs"))] *len(os.listdir("E:/finalyrs_project/real_imgs"))
    fake_weight = [1 / len(os.listdir("E:/finalyrs_project/DatasetAugmentation/pasted"))] *len(os.listdir("E:/finalyrs_project/DatasetAugmentation/pasted"))
    sample_weight = real_weight + fake_weight
    sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
    loader = DataLoader(dataset, batch_size=8,  pin_memory=True, drop_last=True, sampler=sampler)

    for idx, (image,mask, cls) in enumerate(loader):
        print(image.shape,mask.shape, torch.abs(cls - 1))




