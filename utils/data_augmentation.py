import numpy as np
import torch

from torchvision import transforms


class Augmentation(object):
    """Augment gerste patches from uv_dataloader.
    """

    def __call__(self, sample):
        images, label = sample
        random_fliplr = transforms.RandomApply([np.fliplr], p=0.5)
        random_flipud = transforms.RandomApply([np.flipud], p=0.5)
        random_rot180 = transforms.RandomApply([np.rot90, np.rot90], p=0.5)
        transformations = transforms.Compose([
            random_fliplr,
            random_flipud,
            random_rot180
        ])

        augmented_batches = []
        for image in images:
            img = transformations(image)
            augmented_batches.append(torch.from_numpy(img.copy().transpose((2, 0, 1))))

        return torch.stack(augmented_batches), label


class AugmentationSingle(object):
    """Augment gerste image from uv_dataloader.
    """

    def __call__(self, sample):
        image, label = sample
        random_fliplr = transforms.RandomApply([np.fliplr], p=0.5)
        random_flipud = transforms.RandomApply([np.flipud], p=0.5)
        random_rot180 = transforms.RandomApply([np.rot90, np.rot90], p=0.5)
        transformations = transforms.Compose([
            random_fliplr,
            random_flipud,
            random_rot180
        ])

        img = transformations(image)

        return torch.from_numpy(img.copy().transpose((2, 0, 1))), label
