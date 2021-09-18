from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
import uuid

import uv_dataloader
from cnn_model import CNNModel
from cnn3d_model import ConvNetBarley
from hyperparams import get_param_class
from matplotlib.ticker import MaxNLocator
from setproctitle import setproctitle
from sklearn.metrics import balanced_accuracy_score
from train_self_attention import SANNetwork
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.data_augmentation import AugmentationSingle
from uv_dataloader import LeafDataset
from uv_dataloader import wavelength as wv

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
parser.add_argument('--fp_ckpt', type=str, help='')
parser.add_argument('-dp', '--dataset_path',
                    default="/Users/janik/Downloads/UV_Gerste/", type=str,
                    help='')
args = parser.parse_args()


def balanced_accuracy(target, pred):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    res = balanced_accuracy_score(target, pred)
    return res


def getPredAndTarget(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        return pred.view(-1).detach().cpu().numpy().tolist(), target.view(-1).detach().cpu().numpy().tolist()


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
                                split=args.split,
                                mode='train',
                                superpixel=True,
                                bags=False,
                                validation=False)
                                #transform=AugmentationSingle())  # 50000
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

    print("Number of samples train", len(dataset_train))
    print("Number of samples test", len(dataset_test))
    dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=0,
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

    #model = ConvNetBarley(elu=False, avgpool=False, nll=False, num_classes=param_class.num_classes)
    #model = CNNModel(num_classes=param_class.num_classes)

    num_epochs = hyperparams['num_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hyperparams['lr_scheduler_steps'], gamma=0.5, last_epoch=-1)
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters {}".format(num_params))
    print("Starting training for {} epochs".format(num_epochs))
    save_dir = "./uv_dataset/results_cv/"
    writer = SummaryWriter(log_dir=save_dir + run_id, comment="_" + "_id_{}".format(run_id))

    device = "cpu"
    model.to(device)

    balanced_loss_weight = torch.tensor([0.75, 0.25], device=device)  # torch.tensor([0.75, 0.25], device=device)
    #balanced_loss_weight = torch.tensor([0.75, 0.25])
    crit = torch.nn.CrossEntropyLoss(weight=balanced_loss_weight)
    best_acc = 0
    for epoch in tqdm(range(num_epochs)):
        setproctitle("Gerste_MIL" + args.mode + " | epoch {} of {}".format(epoch + 1, num_epochs))
        losses_per_batch = []
        correct = 0
        target, pred = [], []
        total = 0
        for i, (features, labels) in enumerate(dataloader):
            labels = labels[2]
            features = features.float().to(device)
            features = features.unsqueeze(1)
            labels = labels.long().to(device)
            model.train()
            outputs = model.forward(features)
            outputs = outputs.view(labels.shape[0], -1)
            #labels = labels.view(-1)
            loss = crit(outputs, labels)
            optimizer.zero_grad()
            _, predicted = torch.max(outputs.data, 1)
            batch_pred, batch_target = getPredAndTarget(outputs, labels)
            target.append(batch_target)
            pred.append(batch_pred)
            correct += balanced_accuracy(batch_target, batch_pred) * labels.size(0)  # mean
            # correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            losses_per_batch.append(float(loss))
        mean_loss = np.mean(losses_per_batch)
        #correct = balanced_accuracy(target, pred)
        writer.add_scalar('Loss/train', mean_loss, epoch)
        writer.add_scalar('Accuracy/train', 100 * correct / total, epoch)
        print("Epoch {}, mean loss per batch {}, train acc {}".format(epoch, mean_loss, 100 * correct / total))

        if (epoch + 1) % args.test_epoch == 0 or epoch + 1 == num_epochs:
            correct_test = 0
            target, pred = [], []
            total = 0
            model.eval()
            losses_per_batch = []
            attention_weights = []
            with torch.no_grad():
                for i, (features, labels) in enumerate(dataloader_test):
                    labels = labels[2]
                    features = features.float().to(device)
                    features = features.unsqueeze(1)
                    labels = labels.long().to(device)
                    outputs = model.forward(features)
                    outputs = outputs.view(labels.shape[0], -1)
                    #labels = labels.view(-1)
                    loss = crit(outputs, labels)
                    losses_per_batch.append(float(loss))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    batch_pred, batch_target = getPredAndTarget(outputs, labels)
                    target.append(batch_target)
                    pred.append(batch_pred)
                    correct_test += balanced_accuracy(batch_target, batch_pred) * labels.size(0)
                    # correct += (predicted == labels).sum().item()
                mean_loss = np.mean(losses_per_batch)
                print(target, pred)
                #correct_test = balanced_accuracy(target, pred)
                writer.add_scalar('Loss/test', mean_loss, epoch)
            print('Accuracy, mean loss per batch of the network on the test samples: {} %, {}'.format(
                    100 * correct_test / total, mean_loss))
            writer.add_scalar('Accuracy/test', 100 * correct_test / total, epoch)

            if (correct_test) >= best_acc:
                best_acc = (correct_test / total)
            save_checkpoint(save_dir, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hyper_params': hyperparams,
                'eval_bal_acc': correct_test / total,
            }, epoch + 1, best=correct_test / total >= best_acc, run_id=run_id)
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


def save_checkpoint(save_dir, state, epoch, best, run_id, filename='last_model.pth.tar', proceed=False):
    if proceed:
        save_path_checkpoint = os.path.join(save_dir, filename)
    else:
        save_path_checkpoint = os.path.join(os.path.join(save_dir, run_id), filename)
    os.makedirs(os.path.dirname(save_path_checkpoint), exist_ok=True)
    if epoch % 10 == 0:
        torch.save(state, save_path_checkpoint)
    if best:
        torch.save(state, save_path_checkpoint.replace('last_model.pth.tar', 'best_model.pth.tar'))


def test_attention_weights_image():
    param_class = get_param_class(args.data)

    checkpoint = torch.load(args.fp_ckpt)
    if 'hyper_params' in list(checkpoint.keys()):
        hyperparams = checkpoint['hyper_params']
        if hyperparams['n_splits'] not in list(hyperparams.keys()):
            hyperparams['n_splits'] = args.n_splits
        if hyperparams['split'] not in list(hyperparams.keys()):
            hyperparams['split'] = args.split
    else:
        raise ValueError("Should not happen")

    print(args.fp_ckpt)
    print("\nModel with params:\n", hyperparams)
    epoch = checkpoint['epoch']
    eval_bal_acc = checkpoint['eval_bal_acc']
    print(f"\nEpoch of best val acc: {epoch} with {np.round(eval_bal_acc*100,2)}% accuracy")

    print("\nLoading Data ...")

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

    print("Data loaded...\n")
    print("Number of samples test", len(dataset_test))
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
                                 drop_last=False)

    model = SANNetwork(input_size=dataset_test.input_size,
                       num_classes=hyperparams['num_classes'],
                       hidden_layer_size=hyperparams['hidden_layer_size'],
                       dropout=0.02,
                       num_heads=hyperparams['num_heads'],
                       device="cuda")
    model.load_state_dict(checkpoint['state_dict'])

    device = "cuda"
    model.to(device)

    wavelength = wv[20:408]

    palette = sns.color_palette(palette='Set2', n_colors=None, desat=None)

    save_dir = f"{args.fp_ckpt.split('best_model.pth.tar')[0]}figures/"
    print("Saving to ", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    threshold = 0.01

    mean_spectra = []
    std_spectra = []

    with torch.no_grad():
        (features, labels) = next(iter(dataloader_test))
        labels = labels[2]
        for i in np.unique(labels).astype("int"):
            print(f"\nCreating plots for {dataset_test.class_names[i]}")
            # only features per class
            subset_features = features[labels == i]
            subset_features = subset_features.float().to(device)

            # get attention of the samples
            attn = model.get_attention(subset_features)

            # compute the class prediction
            outputs = model.forward(subset_features)
            outputs = outputs.view(subset_features.shape[0], -1)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()

            # look only at attn of correctly predicted
            attn = attn[predicted == i]

            attn_mean = torch.mean(attn, dim=(0,)).detach().cpu().numpy()
            attn_std = torch.std(attn, dim=(0,)).detach().cpu().numpy()

            print(f"Important features (thresholded at {threshold}): {np.array(wavelength)[attn_mean > threshold]}\n")

            plt_title = f"Attention {dataset_test.class_names[i]}"

            fig, ax = _create_fig(hyperparams, plt_title + " Mean Feature Importance", figsize=(10, 5))

            plt.xlabel("Wavelength [nm]", fontsize=10)
            plt.ylabel("Feature Importance [0-1]", fontsize=10)

            # ax.plot(wavelength, attn_mean, linewidth=4)
            attn_mean_plt = attn_mean.copy()
            attn_mean_plt_inv = attn_mean.copy()
            attn_std_plt = attn_std.copy()
            attn_mean_plt[attn_mean <= threshold] = 0.
            attn_mean_plt_inv[attn_mean > threshold] = 0.

            attn_std_plt[attn_mean <= threshold] = None

            ax.bar(wavelength, attn_mean_plt, align="center",
                   width=3.,
                   color=palette[0],
                   capsize=4.)
            ax.bar(wavelength, attn_mean_plt_inv, align="center", width=1.8, color=palette[0])

            # ax.set_ylim(0, 0.19)
            # xticks = np.array(wavelength)[attn_mean > threshold]
            # ax.set_xticks(xticks)
            # ax.set_xticklabels(xticks, rotation='vertical')

            fig.savefig(f"{save_dir}{dataset_test.class_names[i]}_mean.png",
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

            fig, ax = _create_fig(hyperparams, plt_title + " \nSelected Feature Importance (thresholded)", figsize=(5, 5))

            plt.xlabel("Wavelength [nm]", fontsize=10)
            plt.ylabel("Feature Importance [0-1]", fontsize=10)

            xticks = np.array(wavelength)[attn_mean > threshold].round(decimals=2)
            # xticks = np.array(wavelength)
            y_data = attn_mean[attn_mean > threshold]
            y_err = attn_std[attn_mean > threshold]
            plt.xticks(np.arange(len(y_data)), np.round(xticks, 2), rotation=35)
            ax.bar(np.arange(len(y_data)), y_data,
                   yerr=[np.zeros(len(y_err)), y_err], align="center", width=.8,
                   color=palette[0],
                   capsize=4.)
            ax.set_ylim(0, 0.35)

            fig.savefig(f"{save_dir}/{dataset_test.class_names[i]}_feature_importance.png",
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

            fig, ax = _create_fig(hyperparams, plt_title + " Std")

            plt.xlabel("Wavelength [nm]", fontsize=10)
            plt.ylabel("Feature Importance [0-1]", fontsize=10)

            ax.bar(wavelength, attn_std, width=2.)

            fig.savefig(f"{save_dir}/{dataset_test.class_names[i]}_std.png",
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

            # show for 100 individual samples
            num_samples = dataset_test.__len__()
            features_nsamples = subset_features[0:num_samples]
            attn_times_sample = model.forward_attention(features_nsamples, return_softmax=False)
            attn_times_sample = torch.mean(attn_times_sample, dim=(0,)).detach().cpu().numpy()

            attn = model.get_attention(features_nsamples)
            attn_mean = torch.mean(attn, dim=(0,)).detach().cpu().numpy()

            std_spectra.append(torch.std(features_nsamples, dim=(0,)).detach().cpu().numpy())
            features_nsamples = torch.mean(features_nsamples, dim=(0,)).detach().cpu().numpy()
            mean_spectra.append(features_nsamples)

            fig, ax = _create_fig(hyperparams, f"{plt_title} Test Samples Feature Imp.")

            plt.xlabel("Wavelength [nm]", fontsize=10)
            plt.ylabel("Feature Importance [0-1]", fontsize=10)

            attn_mean_plt = attn_mean.copy()
            attn_mean_plt_inv = attn_mean.copy()
            attn_std_plt = attn_std.copy()
            attn_mean_plt[attn_mean <= threshold] = 0.
            attn_mean_plt_inv[attn_mean > threshold] = 0.

            attn_std_plt[attn_mean <= threshold] = None

            ax.bar(wavelength, attn_mean_plt, align="center",
                   width=3.,
                   color=palette[0],
                   capsize=4.)
            ax.bar(wavelength, attn_mean_plt_inv, align="center", width=1.)

            xticks = np.array(wavelength)[attn_mean > threshold]
            ax.set_xticks(xticks)
            ax.set_xticklabels(np.round(xticks, 2), rotation=35)
            # ax.set_ylim(0, 0.15)

            fig.savefig(f"{save_dir}/{dataset_test.class_names[i]}_test_samples_attn.png",
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

            fig, ax = _create_fig(hyperparams, f"{plt_title} Test Samples Mean Signature")

            plt.xlabel("Wavelength [nm]", fontsize=10)
            # plt.ylabel("Feature Importance [0-1]", fontsize=10)
            ax.plot(wavelength, features_nsamples, linewidth=2, alpha=0.6)
            # ax.set_ylim(0.0, 0.6)
            plt.ylabel("Reflectance", fontsize=10)
            fig.savefig(f"{save_dir}/{dataset_test.class_names[i]}_test_samples.png",
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

            fig, ax = _create_fig(hyperparams, f"{plt_title} Test Samples mult. by Feature Imp.")
            ax.plot(wavelength, attn_times_sample, linewidth=2, alpha=0.6)
            ax.set_ylim(0, 0.25)
            plt.ylabel("Reflectance", fontsize=10)
            fig.savefig(f"{save_dir}/{dataset_test.class_names[i]}_test_samples_after_attn.png",
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

        fig, ax = _create_fig(hyperparams, f"All Classes Test Samples Mean Signature")
        plt.xlabel("Wavelength [nm]", fontsize=10)
        # plt.ylabel("Feature Importance [0-1]", fontsize=10)
        for i in np.unique(labels).astype("int"):
            ax.plot(wavelength, mean_spectra[i], linewidth=2, alpha=0.6)
            ax.fill_between(wavelength, mean_spectra[i]+std_spectra[i], mean_spectra[i]-std_spectra[i], alpha=0.5)
        ax.set_ylim(0.0, 1.1)
        ax.legend(dataset_test.class_names)
        plt.ylabel("Reflectance", fontsize=10)
        fig.savefig(f"{save_dir}/all_classes_test_samples.png",
                    bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()


def _create_fig(hyperparams, title="", figsize=(10, 5)):
    fig = plt.figure(figsize=figsize)
    plt.title(title, fontdict={"fontsize": 12})
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    ax.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=18))
    return fig, ax


if __name__ == "__main__":
    print('Start Training')
    set_seed(1)
    train_attention()
