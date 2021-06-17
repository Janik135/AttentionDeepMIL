"""Pytorch dataset object that loads barley dataset."""

import numpy as np
import torch.utils.data as data_utils
import torch
from barley.barley_data import BarleyDataset
from barley.barley_transformations import *
from torchvision import transforms


class BarleyBatches(data_utils.Dataset):
    def __init__(self, train=True, dai="5"):
        self.train = train
        self.dai = dai
        self.data_path = "/Users/janik/Downloads/UV_Gerste/"
        self.transformations = [ToRGB(), CenterCrop((25, 750)), ToBatches((5, 10), 0.5), RescaleBatches((28, 28)),
                                BatchesToTensors()]
        #self.transformations = [ToDynamicBatches(10, 0.5), RescaleBatches((56, 56)), BatchesToTensors()]
        self.train_split = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                            28, 29, 30, 31, 32, 33, 34, 35]
        self.test_split = [7, 8, 9, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_batches()
        else:
            self.test_bags_list, self.test_labels_list = self._create_batches()

    def _create_batches(self):
        if self.train:
            loader = data_utils.DataLoader(BarleyDataset(data_path=self.data_path,
                                                         dai=self.dai,
                                                         split=self.train_split,
                                                         downloaded=True,
                                                         transform=transforms.Compose(self.transformations)),
                                           batch_size=1,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(BarleyDataset(data_path=self.data_path,
                                                         dai=self.dai,
                                                         split=self.test_split,
                                                         downloaded=True,
                                                         transform=transforms.Compose(self.transformations)),
                                           batch_size=1,
                                           shuffle=False)

        bags_list = []
        labels_list = []

        for i, sample_batched in enumerate(loader):
            if i == 1:
                break
            bags_list.append(torch.squeeze(sample_batched['images'], 0))
            labels_list.append(torch.squeeze(sample_batched['labels'], 0))

        bags_list = torch.cat(bags_list)
        labels_list = torch.cat(labels_list)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = torch.Tensor([self.train_labels_list[index]]).type(torch.LongTensor)
        else:
            bag = self.test_bags_list[index]
            label = torch.Tensor([self.test_labels_list[index]]).type(torch.LongTensor)
        return bag, label


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(BarleyBatches(train=True, dai="5"), batch_size=1, shuffle=True)

    test_loader = data_utils.DataLoader(BarleyBatches(train=False, dai="5"), batch_size=1, shuffle=False)
