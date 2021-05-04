import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class GrapeVineDataset(Dataset):
    """Grape Vine dataset."""

    def __init__(self, annotation_dir, image_dir, split, transform=None):
        """
        Args:
            annotation_dir: Directory with all annotations.
            image_dir = Directory with the regarding images.
        """
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.split = split

        _, _, self.annotation_filenames = next(os.walk(annotation_dir))
        self.annotation_filenames.sort()
        self.annotation_filenames = [self.annotation_filenames[i] for i, _ in enumerate(self.annotation_filenames) if
                                     i in split]

        _, _, self.image_filenames = next(os.walk(image_dir))
        self.image_filenames.sort()
        self.image_filenames = [self.image_filenames[i] for i, _ in enumerate(self.image_filenames) if i in split]

        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        annotation_name = os.path.join(self.annotation_dir,
                                       self.annotation_filenames[idx])
        annotation = pd.read_pickle(annotation_name)

        img_shape = annotation['mm_shape']
        image_name = os.path.join(self.image_dir,
                                  self.image_filenames[idx])
        image = np.memmap(image_name, dtype='float32', mode='r', shape=(img_shape[0], img_shape[1], img_shape[2]))

        sample = {'image': image, 'annotation': annotation}

        if self.transform:
            sample = self.transform(sample)

        return sample
