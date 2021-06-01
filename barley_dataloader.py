"""Pytorch dataset object that loads barley dataset as bags."""

import os
from barley_data import BarleyDataset

data_path = "/Users/janik/Downloads/UV_Gerste/"
parsed_data_path = os.path.join(data_path, "parsed_data")
current_path = os.path.join(parsed_data_path, "*.p")

dataset = BarleyDataset(data_path, "5")
for i in range(len(dataset)):
    sample = dataset[i]
    print(sample['image'].shape)
