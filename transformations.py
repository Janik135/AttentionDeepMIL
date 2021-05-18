import numpy as np
import torch

from skimage import transform, util
from spectral import get_rgb


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']['mask']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for annotations because for images,
        # x and y axes are axis 1 and 0 respectively
        ann = transform.resize(annotation, (new_h, new_w))

        return {'image': img, 'annotation': {'mask': ann, 'control': sample['annotation']['control'],
                                             'dat_path': sample['annotation']['dat_path'],
                                             'wine_type': sample['annotation']['wine_type'],
                                             'img_num': sample['annotation']['img_num'],
                                             'date_id': sample['annotation']['date_id'],
                                             'plantid': sample['annotation']['plantid'],
                                             'mm_shape': sample['annotation']['mm_shape'],
                                             'img_min': sample['annotation']['img_min'],
                                             'img_max': sample['annotation']['img_max']}}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, random_state):
        self.r = random_state
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = self.r.randint(0, h - new_h)
        left = self.r.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        annotation = annotation[top: top + new_h, left: left + new_w]

        return {'image': image, 'annotation': {'mask': annotation, 'control': sample['annotation']['control'],
                                               'dat_path': sample['annotation']['dat_path'],
                                               'wine_type': sample['annotation']['wine_type'],
                                               'img_num': sample['annotation']['img_num'],
                                               'date_id': sample['annotation']['date_id'],
                                               'plantid': sample['annotation']['plantid'],
                                               'mm_shape': sample['annotation']['mm_shape'],
                                               'img_min': sample['annotation']['img_min'],
                                               'img_max': sample['annotation']['img_max']}}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        annotation = annotation[:, :, np.newaxis]

        return {'image': torch.from_numpy(image), 'annotation': {
            'mask': torch.from_numpy(annotation.transpose((2, 0, 1))),
            'control': sample['annotation']['control'], 'dat_path': sample['annotation']['dat_path'],
            'wine_type': sample['annotation']['wine_type'], 'img_num': sample['annotation']['img_num'],
            'date_id': sample['annotation']['date_id'], 'plantid': sample['annotation']['plantid'],
            'mm_shape': sample['annotation']['mm_shape'], 'img_min': sample['annotation']['img_min'],
            'img_max': sample['annotation']['img_max']
        }}


class ToRGB(object):
    """Convert hyperspectral images to RGB"""

    def __call__(self, sample):
        image = sample['image']

        image = get_rgb(image, bands=[55, 41, 12])

        return {'image': image, 'annotation': sample['annotation'].copy()}


class Rotate(object):
    """Rotate and wrap image and mask"""

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']['mask']

        rotated_image = transform.rotate(image, angle=45, mode="wrap")
        rotated_mask = transform.rotate(annotation, angle=45, mode="wrap")
        return {'image': rotated_image, 'annotation': {'mask': rotated_mask, 'control': sample['annotation']['control'],
                                                       'dat_path': sample['annotation']['dat_path'],
                                                       'wine_type': sample['annotation']['wine_type'],
                                                       'img_num': sample['annotation']['img_num'],
                                                       'date_id': sample['annotation']['date_id'],
                                                       'plantid': sample['annotation']['plantid'],
                                                       'mm_shape': sample['annotation']['mm_shape'],
                                                       'img_min': sample['annotation']['img_min'],
                                                       'img_max': sample['annotation']['img_max']}}


class Shift(object):
    """Shift and wrap image and mask"""

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']['mask']

        trnsfrm = transform.AffineTransform(translation=(25, 25))
        shifted_image = transform.warp(image, trnsfrm, mode="wrap")
        shifted_mask = transform.warp(image, trnsfrm, mode="wrap")
        return {'image': shifted_image, 'annotation': {'mask': shifted_mask, 'control': sample['annotation']['control'],
                                                       'dat_path': sample['annotation']['dat_path'],
                                                       'wine_type': sample['annotation']['wine_type'],
                                                       'img_num': sample['annotation']['img_num'],
                                                       'date_id': sample['annotation']['date_id'],
                                                       'plantid': sample['annotation']['plantid'],
                                                       'mm_shape': sample['annotation']['mm_shape'],
                                                       'img_min': sample['annotation']['img_min'],
                                                       'img_max': sample['annotation']['img_max']}}


class FlipLeftRight(object):
    """Flip image and mask left to right"""

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']['mask']

        flipped_image = np.fliplr(image)
        flipped_mask = np.fliplr(annotation)
        return {'image': flipped_image, 'annotation': {'mask': flipped_mask, 'control': sample['annotation']['control'],
                                                       'dat_path': sample['annotation']['dat_path'],
                                                       'wine_type': sample['annotation']['wine_type'],
                                                       'img_num': sample['annotation']['img_num'],
                                                       'date_id': sample['annotation']['date_id'],
                                                       'plantid': sample['annotation']['plantid'],
                                                       'mm_shape': sample['annotation']['mm_shape'],
                                                       'img_min': sample['annotation']['img_min'],
                                                       'img_max': sample['annotation']['img_max']}}


class FlipUpDown(object):
    """Flip image and mask up to down"""

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']['mask']

        flipped_image = np.flipud(image)
        flipped_mask = np.flipud(annotation)
        return {'image': flipped_image, 'annotation': {'mask': flipped_mask, 'control': sample['annotation']['control'],
                                                       'dat_path': sample['annotation']['dat_path'],
                                                       'wine_type': sample['annotation']['wine_type'],
                                                       'img_num': sample['annotation']['img_num'],
                                                       'date_id': sample['annotation']['date_id'],
                                                       'plantid': sample['annotation']['plantid'],
                                                       'mm_shape': sample['annotation']['mm_shape'],
                                                       'img_min': sample['annotation']['img_min'],
                                                       'img_max': sample['annotation']['img_max']}}


class AddRandomNoise(object):
    """Flip image and mask up to down"""

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']['mask']

        sigma = 0.155
        noisy_image = util.random_noise(image, var=sigma**2, seed=1)
        noisy_mask = util.random_noise(annotation, var=sigma**2, seed=1)

        return {'image': noisy_image, 'annotation': {'mask': noisy_mask, 'control': sample['annotation']['control'],
                                                       'dat_path': sample['annotation']['dat_path'],
                                                       'wine_type': sample['annotation']['wine_type'],
                                                       'img_num': sample['annotation']['img_num'],
                                                       'date_id': sample['annotation']['date_id'],
                                                       'plantid': sample['annotation']['plantid'],
                                                       'mm_shape': sample['annotation']['mm_shape'],
                                                       'img_min': sample['annotation']['img_min'],
                                                       'img_max': sample['annotation']['img_max']}}



class ToBatches(object):
    """Convert images to bags"""

    def __init__(self, output_size, threshold):
        self.output_size = output_size
        self.threshold = threshold

    def __call__(self, sample):
        image, mask, control = sample['image'], sample['annotation']['mask'], sample['annotation']['control']

        row_pixels, column_pixels = self.output_size
        threshold = int(row_pixels * column_pixels * self.threshold)

        image_batches, batch_labels = [], []

        for r in range(0, image.shape[0], row_pixels):
            for c in range(0, image.shape[1], column_pixels):
                label = int(np.sum(mask[r:r+row_pixels, c:c+column_pixels]) >= 1 and control is False)
                if np.sum(mask[r:r+row_pixels, c:c+column_pixels]) >= threshold:
                    batch_labels.append(label)
                    image_batches.append((image[r:r+row_pixels, c:c+column_pixels, :]))

        return {'images': image_batches, 'labels': batch_labels}


class ToDynamicBatches(object):
    """Convert images to bags"""

    def __init__(self, amount_tiles, threshold):
        self.amount_tiles = amount_tiles
        self.threshold = threshold

    def __call__(self, sample):
        image, mask, control = sample['image'], sample['annotation']['mask'], sample['annotation']['control']

        row_pixels = int(image.shape[0] / self.amount_tiles)
        column_pixels = int(image.shape[1] / self.amount_tiles)
        threshold = int(row_pixels * column_pixels * self.threshold)

        image_batches, batch_labels = [], []

        for r in range(0, image.shape[0], row_pixels):
            for c in range(0, image.shape[1], column_pixels):
                label = int(np.sum(mask[r:r+row_pixels, c:c+column_pixels]) >= 1 and control is False)
                if np.sum(mask[r:r+row_pixels, c:c+column_pixels]) >= threshold:
                    batch_labels.append(label)
                    image_batches.append((image[r:r+row_pixels, c:c+column_pixels, :]))

        return {'images': image_batches, 'labels': batch_labels}


class BatchesToTensors(object):
    """Convert every entry in bag to tensor"""

    def __call__(self, sample):
        images, labels = sample['images'], sample["labels"]

        image_tensors = []
        for image in images:
            img = image.transpose((2, 0, 1))
            image_tensors.append(torch.from_numpy(img.copy()).to(torch.float32))

        label_tensors = []
        for label in labels:
            label_tensors.append((torch.tensor(label)))

        return {'images': torch.stack(image_tensors),
                'labels': torch.stack(label_tensors)}


class RescaleBatches(object):
    """Rescale the image batches in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        images, labels = sample['images'], sample["labels"]

        resized_images = []
        for image in images:

            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image, (new_h, new_w))
            resized_images.append(img)

        return {'images': resized_images, 'labels': labels}
