import glob
import os
import pickle
import spectral
import torch
from torch.utils.data import Dataset


class BarleyDataset(Dataset):
    """Barley dataset."""

    def __init__(self, data_path, dai, transform=None):
        """
        Args:
            data_path: Directory with all data.
            dai: days after inoculation.
        """
        self.data_path = data_path
        self.dai = dai
        self.parsed_data_path = os.path.join(data_path, "parsed_data")
        current_path = os.path.join(self.parsed_data_path, "*.p")

        self.filenames = sorted(list(set(glob.glob(current_path))))
        self.filenames = self._get_filenames()

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.filenames[idx]

        bbox_obj_dict = pickle.load(open(filename, "rb"))
        [(min_y, min_x), (max_y, max_x)] = bbox_obj_dict["bbox"]
        hs_img_path = os.path.join(os.path.join(self.data_path, "{}dai".format(bbox_obj_dict["label_dai"])),
                                   bbox_obj_dict["filename"] + "/data.hdr")
        img = spectral.open_image(hs_img_path)
        img_cropped = img[min_x:max_x, min_y:max_y, :]

        sample = {'image': img_cropped, 'annotation': bbox_obj_dict}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_filenames(self):
        filtered_filenames = []
        for filename in self.filenames:
            bbox_obj_dict = pickle.load(open(filename, "rb"))
            if bbox_obj_dict['label_dai'] == self.dai:
                filtered_filenames.append(filename)

        return filtered_filenames
