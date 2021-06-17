from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import torch.utils.data as data_utils
import torch.optim as optim

from model import Attention, GatedAttention
from sklearn.metrics import balanced_accuracy_score
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from uv_dataloader import LeafDataset as DatasetGerste

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
parser.add_argument('--n_splits', default=5, type=int,
                    help='')
parser.add_argument('--split', default=0, type=int,
                    help='')
parser.add_argument('-dp', '--dataset_path', default="/Users/janik/Downloads/UV_Gerste/", type=str,
                    help='')
parser.add_argument("--port", default=52720)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

hyperparams = {
    'genotype': ["WT"],
    'inoculated': [0, 1],  # [0, 1],
    'dai': ["5"],
    'signature_pre_clip': 0,
    'signature_post_clip': 1,
    'test_size': 0.5,
    'max_num_balanced_inoculated': 5000,
    'num_classes': 2,
    'num_heads': 2,
    'hidden_layer_size': 64,
    'savgol_filter_params': [7, 3]
}

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = data_utils.DataLoader(DatasetGerste(data_path=args.dataset_path,
                                                   genotype=hyperparams['genotype'], inoculated=hyperparams['inoculated'],
                                                   dai=hyperparams['dai'],
                                                   test_size=0.2,
                                                   signature_pre_clip=hyperparams['signature_pre_clip'],
                                                   signature_post_clip=hyperparams['signature_post_clip'],
                                                   max_num_balanced_inoculated=hyperparams['max_num_balanced_inoculated'],
                                                   num_samples_file=500,
                                                   mode='train',
                                                   n_splits=args.n_splits,
                                                   split=args.split,
                                                   superpixel=True),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(DatasetGerste(data_path=args.dataset_path,
                                                  genotype=hyperparams['genotype'], inoculated=hyperparams['inoculated'],
                                                  dai=hyperparams['dai'],
                                                  test_size=0.2,
                                                  signature_pre_clip=hyperparams['signature_pre_clip'],
                                                  signature_post_clip=hyperparams['signature_post_clip'],
                                                  max_num_balanced_inoculated=hyperparams['max_num_balanced_inoculated'],
                                                  num_samples_file=500,
                                                  mode='test',
                                                  n_splits=args.n_splits,
                                                  split=args.split,
                                                  superpixel=True),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)

print('Init Model')
if args.model == 'attention':
    model = Attention()
elif args.model == 'gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
writer = SummaryWriter()


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    y_hat = []
    y = []
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[2]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss = model.calculate_nll(data, bag_label)
        train_loss += loss.data
        _, Y_hat = model.forward(data)
        y_hat.append(Y_hat.numpy().item())
        y.append(bag_label.numpy().item())

        # backward pass
        loss.backward()
        # step
        optimizer.step()

    balanced_acc = balanced_accuracy_score(y, y_hat)
    train_error /= len(train_loader)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Acc/train", train_error, epoch)

    print('Epoch: {}, Loss: {:.4f}, Train acc(balanc.): {:.4f}'.format(epoch, train_loss.cpu().numpy(), balanced_acc))


def test():
    model.eval()
    test_loss = 0.
    y_hat = []
    y = []
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[2]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_nll(data, bag_label)
        test_loss += loss.data
        _, Y_hat = model.forward(data)
        y_hat.append(Y_hat.numpy().item())
        y.append(bag_label.numpy().item())


    balanced_acc = balanced_accuracy_score(y, y_hat)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test acc(balanc.): {:.4f}'.format(test_loss.cpu().numpy(), balanced_acc))


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
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test()
    writer.flush()
    writer.close()
