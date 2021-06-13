"""Pytorch dataset object that loads barley dataset as bags."""

import numpy as np
import torch.utils.data as data_utils
import torch
from barley.barley_data import BarleyDataset
from barley.barley_transformations import *
from torchvision import transforms


class BarleyBags(data_utils.Dataset):
    def __init__(self, train=True, dai="5"):
        self.train = train
        self.dai = dai
        self.data_path = "/Users/janik/Downloads/UV_Gerste/"
        self.transformations = [ToRGB(), ToDynamicBatches(10, 0), RescaleBatches((56, 56)), BatchesToTensors()]
        train_split = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                       29, 30, 31, 32, 33, 34, 35]
        test_split = [7, 8, 9, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

        if self.train:
            self.dataset = BarleyDataset(data_path=self.data_path,
                                         dai=self.dai,
                                         split=train_split,
                                         downloaded=True,
                                         transform=transforms.Compose(self.transformations))
        else:
            self.dataset = BarleyDataset(data_path=self.data_path,
                                         dai=self.dai,
                                         split=test_split,
                                         downloaded=True,
                                         transform=transforms.Compose(self.transformations))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample_batched = self.dataset[index]
        bag = torch.squeeze(sample_batched['images'], 0)
        labels = torch.squeeze(sample_batched['labels'], 0)
        label = max(labels), labels

        return bag, label


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(BarleyBags(train=True, dai="5"), batch_size=1, shuffle=True)

    test_loader = data_utils.DataLoader(BarleyBags(train=False, dai="5"), batch_size=1, shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(mnist_bags_train, len(train_loader),
                                                                            np.mean(len_bag_list_train),
                                                                            np.max(len_bag_list_train),
                                                                            np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(mnist_bags_test, len(test_loader),
                                                                            np.mean(len_bag_list_test),
                                                                            np.max(len_bag_list_test),
                                                                            np.min(len_bag_list_test)))
