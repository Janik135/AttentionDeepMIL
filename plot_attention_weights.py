from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import spectral
import torch
import uuid

from hyperparams import get_param_class
from torch.utils.data import DataLoader
from uv_dataloader import LeafDataset

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('-d', '--data', required=True, type=str,
                    help='What do you want to do? Select either train, test, full_image_test, attention')
parser.add_argument('-dp', '--dataset_path',
                    default="/Users/janik/Downloads/UV_Gerste/", type=str,
                    help='')
args = parser.parse_args()


def plot_attention():
    param_class = get_param_class(args.data)
    run_id = args.data + '_' + str(uuid.uuid1())

    dataset_train = LeafDataset(data_path=args.dataset_path,
                                genotype=param_class.genotype, inoculated=param_class.inoculated, dai=param_class.dai,
                                test_size=param_class.test_size,
                                signature_pre_clip=param_class.signature_pre_clip,
                                signature_post_clip=param_class.signature_post_clip,
                                max_num_balanced_inoculated=param_class.max_num_balanced_inoculated,
                                num_samples_file=param_class.num_samples_file,
                                mode='train',
                                superpixel=True,
                                bags=True)  # 50000
    dataset_test = LeafDataset(data_path=args.dataset_path,
                               genotype=param_class.genotype, inoculated=param_class.inoculated, dai=param_class.dai,
                               test_size=param_class.test_size,
                               signature_pre_clip=param_class.signature_pre_clip,
                               signature_post_clip=param_class.signature_post_clip,
                               max_num_balanced_inoculated=param_class.max_num_balanced_inoculated,  # 50000
                               num_samples_file=param_class.num_samples_file,
                               mode="test",
                               superpixel=True,
                               bags=True)

    print("Number of samples train", len(dataset_train))
    print("Number of samples test", len(dataset_test))
    dataloader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,
                                 drop_last=False)

    # sample = dataset_test[0]
    attention_weights = np.load('mains/attention_weights.npy', allow_pickle=True)
    min_att = 2.489354e-18
    max_att = 0.99967647
    for i, (patches, labels) in enumerate(dataset_test):
        weighted_batches = []
        for j, weights in enumerate(attention_weights[i]):
            weighted_batches.append(patches[j] * ((weights - min(attention_weights[i])) /
                                                  (max(attention_weights[i]) - min(attention_weights[i]))))
        whole_img = np.concatenate(patches, axis=1)
        weighted_img = np.concatenate(weighted_batches, axis=1)

        spectral.imshow(whole_img)
        plt.title('Original image #{}'.format(i+1))
        plt.savefig('images/{}_original_image.pdf'.format(i+1))

        spectral.imshow(weighted_img)
        plt.title('Attention weights image #{}'.format(i+1))
        plt.savefig('images/{}_attention_image.pdf'.format(i+1))
        plt.show()


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
    print('Plot attention weights')
    set_seed(1)
    plot_attention()
