"""Pytorch dataset object that loads barley dataset as bags."""

import numpy as np
import torch.utils.data as data_utils
import torch
from barley_data import BarleyDataset
from barley_transformations import *
from torchvision import transforms


class BarleyBags(data_utils.Dataset):
    def __init__(self, train=True, dai="5"):
        self.train = train
        self.dai = dai
        self.data_path = "/Users/janik/Downloads/UV_Gerste/"
        self.transformations = [ToDynamicBatches(10, 0), RescaleBatches((56, 56)), BatchesToTensors()]

        if self.train:
            self.loader = data_utils.DataLoader(BarleyDataset(data_path=self.data_path,
                                                              dai=self.dai,
                                                              transform=transforms.Compose(self.transformations),
                                                              downloaded=True),
                                                batch_size=1,
                                                shuffle=False)
        else:
            self.loader = data_utils.DataLoader(BarleyDataset(data_path=self.data_path,
                                                              dai=self.dai,
                                                              transform=transforms.Compose(self.transformations),
                                                              downloaded=True),
                                                batch_size=1,
                                                shuffle=False)

        self.dataloader_iterator = iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index):
        print(index)
        try:
            sample_batched = next(self.dataloader_iterator)
            bag = torch.squeeze(sample_batched['images'], 0)
            labels = torch.squeeze(sample_batched['labels'], 0)
            label = max(labels), labels
        except:
            self.dataloader_iterator = iter(self.loader)
            sample_batched = next(self.dataloader_iterator)
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
