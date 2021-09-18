import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from captum.attr import IntegratedGradients
from cnn_model import CNNModel
from hyperparams import get_param_class
from torch.utils.data import DataLoader
from uv_dataloader import LeafDataset

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('-d', '--data', required=True, type=str,
                    help='What do you want to do? Select either train, test, full_image_test, attention')
parser.add_argument('--split', required=True, type=int,
                    help='')
parser.add_argument('--model_path', required=True, type=str,
                    help='')
parser.add_argument('-dp', '--dataset_path',
                    default="/Users/janik/Downloads/UV_Gerste/", type=str,
                    help='')
args = parser.parse_args()

param_class = get_param_class(args.data)


def get_int_gradients():
    dataset_test = LeafDataset(data_path=args.dataset_path,
                               genotype=param_class.genotype, inoculated=param_class.inoculated, dai=param_class.dai,
                               test_size=param_class.test_size,
                               signature_pre_clip=param_class.signature_pre_clip,
                               signature_post_clip=param_class.signature_post_clip,
                               max_num_balanced_inoculated=param_class.max_num_balanced_inoculated,  # 50000
                               num_samples_file=param_class.num_samples_file,
                               split=args.split,
                               mode="test",
                               superpixel=True,
                               bags=False,
                               validation=False)

    net = CNNModel(2)
    net = net.eval()
    net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,
                                 drop_last=False)

    attributions_inoc = []
    attributions_ok = []
    for (features, labels) in dataloader_test:
        features = features.unsqueeze(1)
        labels = labels[2]
        output = net(features)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)

        pred_label_idx.squeeze_()

        integrated_gradients = IntegratedGradients(net)
        attributions_ig = integrated_gradients.attribute(features, target=pred_label_idx, n_steps=200)

        attr_np = np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))

        max_x = np.max(attr_np, axis=0)
        max_y = np.max(max_x, axis=0)
        if labels == 0:
            attributions_ok.append(max_y)
        elif labels == 1:
            attributions_inoc.append(max_y)

    attr_df_ok = pd.DataFrame(attributions_ok)
    attr_df_ok.to_pickle('cnn_bad_split_ok.pkl')
    attr_df_inoc = pd.DataFrame(attributions_inoc)
    attr_df_inoc.to_pickle('cnn_bad_split_inoc.pkl')


if __name__ == "__main__":
    get_int_gradients()
