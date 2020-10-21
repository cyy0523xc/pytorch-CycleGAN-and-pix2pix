"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data


class ImageDataset(data.Dataset):
    """传入图像列表的数据集，通常用于预测阶段"""

    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        return self.transform(img)

    def __len__(self):
        return len(self.imgs)
