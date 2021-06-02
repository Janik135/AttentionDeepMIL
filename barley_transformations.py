import numpy as np
import torch

from skimage import transform


class ToDynamicBatches(object):
    """Convert images to bags"""

    def __init__(self, amount_tiles, threshold):
        self.amount_tiles = amount_tiles
        self.threshold = threshold

    def __call__(self, sample):
        image, mask, inoculated = sample['image'], sample['annotation']['mask'], \
                                  sample['annotation']['label_inoculated']

        row_pixels = int(image.shape[0] / self.amount_tiles)
        column_pixels = int(image.shape[1] / self.amount_tiles)
        threshold = int(row_pixels * column_pixels * self.threshold)

        image_batches, batch_labels = [], []

        for r in range(0, image.shape[0], row_pixels):
            for c in range(0, image.shape[1], column_pixels):
                label = int(np.sum(mask[r:r+row_pixels, c:c+column_pixels]) >= 1 and bool(inoculated) is True)
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
