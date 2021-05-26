"""Pytorch dataset object that loads grape vine dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from transformations import *
from torchvision import transforms
from data import GrapeVineDataset


class VineBags(data_utils.Dataset):
    def __init__(self, train=True, seed=1):
        self.train = train
        self.r = np.random.RandomState(seed)
        self.train_split = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
        self.test_split = [3, 4, 12, 13, 14]
        # self.transformations = [ToRGB(), ToDynamicBatches(10, 0.5), AugmentBatches()]
        self.transformations = [ToRGB(), CentralCrop((416, 369)), ToBatches((52, 41), 0.5), AugmentBatches()]
        # self.transformations = [RandomCrop((416, 369), self.r), ToBatches((52, 41), 0), BatchesToTensors()]
        self.annotation_dir = '/Users/janik/Downloads/UVVorversuch_cropped/cropped_annotation'
        self.image_dir = '/Users/janik/Downloads/UVVorversuch_cropped/cropped_norm'

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        train_dataset = self._create_train_dataset()
        test_dataset = self._create_test_dataset()

        if self.train:
            loader = data_utils.DataLoader(train_dataset,
                                           batch_size=1,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(test_dataset,
                                           batch_size=1,
                                           shuffle=False)

        bags_list = []
        labels_list = []

        for i, sample_batched in enumerate(loader):
            bags_list.append(torch.squeeze(sample_batched['images'], 0))
            labels_list.append(torch.squeeze(sample_batched['labels'], 0))

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label

    def _create_train_dataset(self):
        train_dataset = GrapeVineDataset(annotation_dir=self.annotation_dir,
                                         image_dir=self.image_dir,
                                         split=self.train_split,
                                         transform=transforms.Compose(self.transformations))

        return train_dataset

    def _create_test_dataset(self):
        test_dataset = GrapeVineDataset(annotation_dir=self.annotation_dir,
                                        image_dir=self.image_dir,
                                        split=self.test_split,
                                        transform=transforms.Compose([ToRGB(), CentralCrop((416, 369)),
                                                                      ToBatches((52, 41), 0.5),
                                                                      PrepareValidationBatches()]))
                                        # transform=transforms.Compose([RandomCrop((416, 369), self.r),
                                        #                               ToBatches((52, 41), 0), BatchesToTensors()]))
        return test_dataset


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(VineBags(train=True), batch_size=1, shuffle=True)

    test_loader = data_utils.DataLoader(VineBags(train=False), batch_size=1, shuffle=False)

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
