import numpy as np
import torch

from skimage import transform
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

    def __init__(self, output_size):
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

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

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


class ToBag(object):
    """Convert images to bags"""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask, control = sample['image'], sample['annotation']['mask'], sample['annotation']['control']

        row_pixels, column_pixels = self.output_size

        images, image_index_labels = [], []

        for r in range(0, image.shape[0], row_pixels):
            for c in range(0, image.shape[1], column_pixels):
                label = int(np.sum(mask[r:r+row_pixels, c:c+column_pixels]) >= 1 and control is False)
                image_index_labels.append(label)
                images.append((image[r:r+row_pixels, c:c+column_pixels, :]))

        return {'images': images, 'labels': image_index_labels}


class BagToTensors(object):
    """Convert every entry in bag to tensor"""

    def __call__(self, sample):
        images, labels = sample['images'], sample["labels"]

        image_tensors = []
        for image in images:
            img = image.transpose((2, 0, 1))
            image_tensors.append(torch.from_numpy(img).to(torch.float32))

        label_tensors = []
        for label in labels:
            label_tensors.append((torch.tensor(label)))

        im_shape = images[0].shape
        im_out_dim = torch.Tensor(len(images), im_shape[0], im_shape[1], im_shape[2])

        return {'images': torch.stack(image_tensors),
                'labels': torch.stack(label_tensors)}
