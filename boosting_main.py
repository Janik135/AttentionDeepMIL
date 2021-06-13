from __future__ import print_function

import argparse
import numpy as np
import os
import torch

from barley.barley_single_dataloader import BarleyBatches
from sklearn.ensemble import GradientBoostingClassifier

def set_seed(seed=42):
    """
    Set random seeds for all possible random processes.

    Args:
        seed (int)
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    print('Start Training')
    set_seed(1)
    print('Load Train and Test Set')

    train_data = BarleyBatches(train=True)
    x_train, y_train = [], []
    for i in range(len(train_data)):
        x, y = train_data[i]
        x_train.append([x.cpu().numpy().transpose((1, 2, 0)).reshape(-1)])
        y_train.append(y.cpu().numpy())
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    test_data = BarleyBatches(train=False)

    print('Init Model')
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).\
        fit(x_train, y_train)
