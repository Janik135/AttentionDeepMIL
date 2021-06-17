from __future__ import print_function

import argparse
import numpy as np
import os
import torch

from barley.barley_single_dataloader import BarleyBatches
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from uv_dataloader import LeafDataset as DatasetGerste

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


parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--n_splits', default=5, type=int,
                    help='')
parser.add_argument('--split', default=0, type=int,
                    help='')
parser.add_argument('-dp', '--dataset_path', default="/Users/janik/Downloads/UV_Gerste/", type=str,
                    help='')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=57865)
args = parser.parse_args()

hyperparams = {
    'genotype': ["WT"],
    'inoculated': [0, 1],  # [0, 1],
    'dai': ["5"],
    'signature_pre_clip': 0,
    'signature_post_clip': 1,
    'test_size': 0.5,
    'max_num_balanced_inoculated': 1300,
    'num_classes': 2,
    'num_heads': 2,
    'hidden_layer_size': 64,
    'savgol_filter_params': [7, 3]
}


if __name__ == "__main__":
    print('Start Training')
    set_seed(1)
    print('Load Train and Test Set')

    train_data = DatasetGerste(data_path=args.dataset_path,
                                          genotype=hyperparams['genotype'], inoculated=hyperparams['inoculated'],
                                          dai=hyperparams['dai'],
                                          test_size=0.2,
                                          signature_pre_clip=hyperparams['signature_pre_clip'],
                                          signature_post_clip=hyperparams['signature_post_clip'],
                                          max_num_balanced_inoculated=hyperparams['max_num_balanced_inoculated'],
                                          num_samples_file=130,
                                          mode='train',
                                          n_splits=args.n_splits,
                                          split=args.split,
                                          superpixel=True,
                                          bags=True)
    test_data = DatasetGerste(data_path=args.dataset_path,
                              genotype=hyperparams['genotype'], inoculated=hyperparams['inoculated'],
                              dai=hyperparams['dai'],
                              test_size=0.2,
                              signature_pre_clip=hyperparams['signature_pre_clip'],
                              signature_post_clip=hyperparams['signature_post_clip'],
                              max_num_balanced_inoculated=hyperparams['max_num_balanced_inoculated'],
                              num_samples_file=130,
                              mode='test',
                              n_splits=args.n_splits,
                              split=args.split,
                              superpixel=True,
                              bags=True)
    all_x_train, all_y_train = [], []
    for i in range(len(train_data)):
        x_train, y_train = train_data[i]
        all_x_train.append(x_train)
        all_y_train.append(y_train)
    all_x_train = np.concatenate(all_x_train)
    all_y_train = np.concatenate(all_y_train)

    print('Init Model')
    clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, max_depth=1, random_state=0).\
        fit(all_x_train, all_y_train)
    all_y_test, all_y_pred= [], []
    for i in range(len(test_data)):
        x_test, y_test = test_data[i]
        y_pred = clf.predict(x_test)
        all_y_test.append(np.max(y_test))
        #all_y_pred.append(np.max(y_pred))
        all_y_pred.append(np.mean(y_pred))
    print(balanced_accuracy_score(all_y_test, all_y_pred))
