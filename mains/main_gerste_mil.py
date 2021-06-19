from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import uuid

from hyperparams import get_param_class
from setproctitle import setproctitle
from train_self_attention import SANNetwork
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from uv_dataloader import LeafDataset

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('-m', '--mode', default="train", type=str,
                    help='What do you want to do? Select either train, test, full_image_test, attention')
parser.add_argument('-d', '--data', required=True, type=str,
                    help='What do you want to do? Select either train, test, full_image_test, attention')
parser.add_argument('--lr', required=True, type=float,
                    help='')
parser.add_argument('-e', '--num_epochs', required=True, type=int,
                    help='')
parser.add_argument('--lr_scheduler_steps', required=True, type=int,
                    help='')
parser.add_argument('--test_epoch', required=True, type=int,
                    help='')

parser.add_argument('--n_splits', required=True, type=int,
                    help='')
parser.add_argument('--split', required=True, type=int,
                    help='')

parser.add_argument('-dp', '--dataset_path',
                    default="/Users/janik/Downloads/UV_Gerste/", type=str,
                    help='')
args = parser.parse_args()


def train_attention():
    param_class = get_param_class(args.data)
    run_id = args.data + '_' + str(uuid.uuid1())

    dataset_train = LeafDataset(data_path=args.dataset_path,
                                genotype=param_class.genotype, inoculated=param_class.inoculated, dai=param_class.dai,
                                test_size=param_class.test_size,
                                signature_pre_clip=param_class.signature_pre_clip,
                                signature_post_clip=param_class.signature_post_clip,
                                max_num_balanced_inoculated=param_class.max_num_balanced_inoculated,
                                num_samples_file=param_class.num_samples_file,
                                mode='train')  # 50000
    dataset_test = LeafDataset(data_path=args.dataset_path,
                               genotype=param_class.genotype, inoculated=param_class.inoculated, dai=param_class.dai,
                               test_size=param_class.test_size,
                               signature_pre_clip=param_class.signature_pre_clip,
                               signature_post_clip=param_class.signature_post_clip,
                               max_num_balanced_inoculated=param_class.max_num_balanced_inoculated,  # 50000
                               num_samples_file=param_class.num_samples_file,
                               mode="test")

    print("Number of samples train", len(dataset_train))
    print("Number of samples test", len(dataset_test))
    dataloader = DataLoader(dataset_train, batch_size=param_class.batch_size, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=param_class.batch_size, shuffle=False, num_workers=0,
                                 drop_last=False)

    hyperparams = dataset_train.hyperparams
    print("Number of batches train", len(dataloader))
    print("Number of batches test", len(dataloader_test))

    # Original class counts train: 67578 264112
    # Original class counts test: 68093 263597
    hyperparams['num_classes'] = param_class.num_classes
    hyperparams['hidden_layer_size'] = param_class.hidden_layer_size
    hyperparams['num_heads'] = param_class.num_heads
    hyperparams['lr'] = args.lr
    hyperparams['num_epochs'] = args.num_epochs
    hyperparams['lr_scheduler_steps'] = args.lr_scheduler_steps

    model = SANNetwork(input_size=dataset_train.input_size,
                       num_classes=hyperparams['num_classes'],
                       hidden_layer_size=hyperparams['hidden_layer_size'],
                       dropout=0.02,
                       num_heads=hyperparams['num_heads'],
                       device="cuda")

    num_epochs = hyperparams['num_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hyperparams['lr_scheduler_steps'], gamma=0.5, last_epoch=-1)
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters {}".format(num_params))
    print("Starting training for {} epochs".format(num_epochs))
    save_dir = "./uv_dataset/results_cv/"
    writer = SummaryWriter(log_dir=save_dir + run_id, comment="_" + "_id_{}".format(run_id))

    device = "cuda"
    model.to(device)

    balanced_loss_weight = torch.tensor([1., 1.], device=device)  # torch.tensor([0.75, 0.25], device=device)
    crit = torch.nn.CrossEntropyLoss(weight=balanced_loss_weight)
    best_acc = 0
    for epoch in tqdm(range(num_epochs)):
        setproctitle("Gerste_MIL" + args.mode + " | epoch {} of {}".format(epoch + 1, num_epochs))
        losses_per_batch = []
        correct = 0
        total = 0
        for i, (features, labels) in enumerate(dataloader):
            labels = labels[2]

            features = features.float().to(device)
            labels = labels.long().to(device)
            model.train()
            outputs = model.forward(features)
            outputs = outputs.view(labels.shape[0], -1)
            loss = crit(outputs, labels)
            optimizer.zero_grad()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            losses_per_batch.append(float(loss))
        mean_loss = np.mean(losses_per_batch)

        writer.add_scalar('Loss/train', mean_loss, epoch)
        writer.add_scalar('Accuracy/train', 100 * correct / total, epoch)
        print("Epoch {}, mean loss per batch {}, train acc {}".format(epoch, mean_loss, 100 * correct / total))

        if (epoch + 1) % args.test_epoch == 0 or epoch + 1 == num_epochs:
            correct = 0
            total = 0
            model.eval()
            losses_per_batch = []
            with torch.no_grad():
                for i, (features, labels) in enumerate(dataloader_test):
                    labels = labels[2]
                    features = features.float().to(device)
                    labels = labels.long().to(device)
                    outputs = model.forward(features)
                    loss = crit(outputs, labels)
                    losses_per_batch.append(float(loss))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                mean_loss = np.mean(losses_per_batch)
                writer.add_scalar('Loss/test', mean_loss, epoch)

            print('Accuracy of the network on the test samples: %d %%' % (
                    100 * correct / total))
            writer.add_scalar('Accuracy/test', 100 * correct / total, epoch)

            if (correct / total) >= best_acc:
                best_acc = (correct / total)
            model.train()

        scheduler.step()


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
    train_attention()
