import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from uv_dataloader import LeafDataset as DatasetGerste
import uuid
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
from torch.utils.tensorboard import SummaryWriter
import spectral

mpl.rcParams['savefig.pad_inches'] = 0

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('-m', '--mode', default="train", type=str,
                    help='What do you want to do? Select either train, test, full_image_test, attention')
parser.add_argument('--lr', default=0.01, type=float,
                    help='')
parser.add_argument('-e', '--num_epochs', default=1000, type=int,
                    help='')
parser.add_argument('--lr_scheduler_steps', default=1000, type=int,
                    help='')
parser.add_argument('--lr_scheduler_laststep', default=1000000, type=int,
                    help='epoch of last lr decay')

parser.add_argument('--test_epoch', default=10, type=int,
                    help='')

parser.add_argument('--n_splits', default=5, type=int,
                    help='')
parser.add_argument('--split', default=0, type=int,
                    help='')
parser.add_argument('-ds', '--dataset',
                    default="gerste", type=str,
                    help='')
parser.add_argument('-dp', '--dataset_path', default="/Users/janik/Downloads/UV_Gerste/", type=str,
                    help='')
parser.add_argument('--device', default="cuda", type=str, help='')
parser.add_argument('--save_dir', default="results_cv_new", type=str, help='')
parser.add_argument("--port", default=57865)



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


def getDataset(mode, param_class, test_size=None):
    if isinstance(param_class, dict):
        _param_class = Dict(param_class)
    else:
        _param_class = param_class

    if test_size is None:
        test_size = _param_class.test_size

    hyperparams = {
        'genotype': ["WT"],
        'inoculated': [0, 1],  # [0, 1],
        'dai': ["5"],
        'signature_pre_clip': 0,
        'signature_post_clip': 250,
        'test_size': 0.5,
        'max_num_balanced_inoculated': 5000,
        'num_classes': 2,
        'num_heads': 2,
        'hidden_layer_size': 64,
        'savgol_filter_params': [7, 3]
    }
    if args.dataset == 'gerste':
        dataset_ = DatasetGerste(data_path=args.dataset_path,
                                 genotype=hyperparams['genotype'], inoculated=hyperparams['inoculated'],
                                 dai=hyperparams['dai'],
                                 test_size=hyperparams['test_size'],
                                 signature_pre_clip=hyperparams['signature_pre_clip'],
                                 signature_post_clip=hyperparams['signature_post_clip'],
                                 max_num_balanced_inoculated=hyperparams['max_num_balanced_inoculated'],
                                 num_samples_file=5,
                                 mode='train',
                                 n_splits=args.n_splits,
                                 split=args.split,
                                 superpixel=True)
    elif args.dataset == 'ruebe':
        dataset_ = DatasetRuebe(data_path=args.dataset_path,
                                dai=_param_class.dai,
                                test_size=_param_class.test_size,
                                signature_pre_clip=_param_class.signature_pre_clip,
                                signature_post_clip=_param_class.signature_post_clip,
                                max_num_balanced_leafs=_param_class.max_num_balanced_leafs,
                                num_samples_file=_param_class.num_samples_file,
                                mode=mode,
                                n_splits=args.n_splits,
                                split=args.split,
                                superpixel=_param_class.superpixel)
    else:
        raise ValueError('dataset unknown')

    return dataset_


def train_gradientboosting():
    global proctitle
    from uv_dataset.gradientboosting import gradientboosting
    import pickle
    setproctitle(proctitle + args.mode + " GB")
    param_class = get_param_class(args.data)
    run_id = args.data + '_' + str(uuid.uuid1())

    save_dir = "./uv_dataset/data/{}/gb_superpixel/cv_{}/".format(args.dataset, args.n_splits)
    data_file_name = os.path.join(save_dir, "data_{}.p".format(args.data))

    dataset_train = getDataset('train', param_class, test_size=0)

    print("Number of samples", len(dataset_train))
    wavelength = dataset_train.wavelength
    if not os.path.isfile(data_file_name):
        print("Pre-processing data")
        dataloader = DataLoader(dataset_train, batch_size=16, shuffle=False, num_workers=0)  # len(dataset_train)

        """
         it = iter(dataloader)
        X_train, y_train = next(it)
    
        it = iter(dataloader_test)
        X_test, y_test = next(it)
    
        print(len(X_train), len(y_train))
        print(len(X_test), len(y_test))
        """
        X_, y_ = [], []
        for features, labels in tqdm(dataloader):
            X_ += features.tolist()
            labels = labels[2]
            y_ += labels.tolist()

        X_, y_ = np.array(X_), np.array(y_)

        os.makedirs(save_dir, exist_ok=True)

        if not os.path.isfile(data_file_name):
            pickle.dump({'X_': X_, 'y_': y_},
                        open(data_file_name, "wb"))
    else:
        print("Loading pre-processed data")
        data_dict = pickle.load(open(data_file_name, "rb"))
        X_, y_ = data_dict['X_'], data_dict['y_']

    save_dir = save_dir.replace('data/', 'results_gb_superpixel/{}/'.format(args.data))
    os.makedirs(save_dir, exist_ok=True)
    gradientboosting(X_, y_, param_class, save_dir, args.n_splits)


def test_gradientboosting():
    global proctitle
    from uv_dataset.gradientboosting import gradientboosting
    import pickle
    setproctitle(proctitle + args.mode + " GB")

    # path = 'uv_dataset/results/2786fc22-515a-11ea-b1c3-1c1b0d97d8cf/best_model.pth.tar'
    path = 'uv_dataset/{}'.format(args.save_dir)  # /{}/best_model.pth.tar'
    # models_to_test_all = [x[0] for x in os.walk(path)][1:]
    results = dict()
    used_dict_classes_keys = [x for x in dict_classes_keys if "_260" in x]
    for class_key in sorted(used_dict_classes_keys):
        # models_to_test = [x for x in models_to_test_all if class_key in x]

        param_class = get_param_class(class_key)
        save_dir = "./uv_dataset/data/{}/gb{}/cv_{}/".format(args.dataset, args.save_dir, args.n_splits)
        data_file_name = os.path.join(save_dir, "data_{}.p".format(class_key))
        if os.path.isfile(data_file_name):
            print("Loading pre-processed data")
            data_dict = pickle.load(open(data_file_name, "rb"))
            X_, y_ = data_dict['X_'], data_dict['y_']

            save_dir = save_dir.replace('data/', 'results_gb{}/{}/'.format(args.save_dir, class_key))
            print("--" * 42)
            print("Testing", class_key)
            results[class_key] = gradientboosting(X_, y_, param_class, save_dir, args.n_splits, mode="test")

    for k in sorted(list(results.keys())):
        print(k, results[k]['mean'], "%, +-", results[k]['std'])


def train_attention():
    global proctitle

    param_class = get_param_class(args.data)
    run_id = args.data + '_' + '{}of{}cv'.format(args.split, args.n_splits) + '_' + str(uuid.uuid1())
    dataset_train = getDataset('train', param_class)
    dataset_test = getDataset('test', param_class)

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
    hyperparams['lr_scheduler_laststep'] = args.lr_scheduler_laststep
    # input("{} press key".format(args.data))
    # exit()

    device = args.device

    model = SANNetwork(input_size=dataset_train.input_size,
                       num_classes=hyperparams['num_classes'],
                       hidden_layer_size=hyperparams['hidden_layer_size'],
                       dropout=0.02,
                       num_heads=hyperparams['num_heads'],
                       device=device)

    # path = 'uv_dataset/results/0_first_good_resultWT_2786fc22-515a-11ea-b1c3-1c1b0d97d8cf/best_model.pth.tar'
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint['state_dict'])

    num_epochs = hyperparams['num_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams['lr'], momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hyperparams['lr_scheduler_steps'], gamma=0.5, last_epoch=-1)
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters {}".format(num_params))
    print("Starting training for {} epochs".format(num_epochs))
    # os.makedirs('./uv_dataset/results', exist_ok=True)
    save_dir = "./uv_dataset/{}/".format(args.save_dir)
    writer = SummaryWriter(log_dir=save_dir + run_id, comment="_" + "_id_{}".format(run_id))

    model.to(device)

    balanced_loss_weight = torch.tensor([1.] * hyperparams['num_classes'],
                                        device=device)  # torch.tensor([0.75, 0.25], device=device)
    crit = torch.nn.CrossEntropyLoss(weight=balanced_loss_weight)
    best_acc = 0
    for epoch in tqdm(range(num_epochs)):
        setproctitle(proctitle + args.mode + " {}|{}-{}cv|e {} of {}".format(args.data,
                                                                             hyperparams["split"],
                                                                             hyperparams["n_splits"],
                                                                             epoch + 1,
                                                                             num_epochs))
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
                    # batch_pred, batch_target = getPredAndTarget(outputs, labels)
                    # correct += balanced_accuracy(batch_target, batch_pred) * labels.size(0)  # mean
                    # correct += (predicted == labels).sum().item()
                mean_loss = np.mean(losses_per_batch)
                writer.add_scalar('Loss/test', mean_loss, epoch)

            save_checkpoint(save_dir, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hyper_params': hyperparams,
                'eval_acc': 100 * correct / total,
            }, epoch + 1, best=(correct / total) >= best_acc, run_id=run_id)
            print('Accuracy of the network on the test samples: %f %%' % (
                    100 * correct / total))
            writer.add_scalar('Accuracy/test', 100 * correct / total, epoch)

            if (correct / total) >= best_acc:
                best_acc = (correct / total)
            model.train()
        if epoch < args.lr_scheduler_laststep:
            scheduler.step()


def test_model_accuracy():
    # path = 'uv_dataset/results/2786fc22-515a-11ea-b1c3-1c1b0d97d8cf/best_model.pth.tar'
    path = 'uv_dataset/{}'.format(args.save_dir)  # /{}/best_model.pth.tar'
    checkpoint_template = 'best_model.pth.tar'
    models_to_test_all = [x[0] for x in os.walk(path)][1:]
    models_to_test_all = [x for x in models_to_test_all if "_260_" in x]
    print(models_to_test_all)
    # 1 / 0
    device = args.device
    results = dict()
    used_dict_classes_keys = [x for x in dict_classes_keys if "_260" in x]
    for class_key in used_dict_classes_keys:
        models_to_test = [x for x in models_to_test_all if class_key in x]
        print(models_to_test)
        # if len(models_to_test) != 5:
        #    continue
        cv_acc = [0] * len(models_to_test)
        for checkpoint_dir in models_to_test:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_template)
            print(checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            hyperparams = checkpoint['hyper_params']
            # print("Model with params:", hyperparams)
            param_class = get_param_class(class_key)
            epoch = checkpoint['epoch']
            eval_acc = checkpoint['eval_acc']
            # print("\n")
            # print(class_key, epoch, eval_acc)
            # print("\n")
            cv_acc[int(hyperparams['split'])] = eval_acc
            continue
            """
            dataset_test = LeafDataset(data_path=args.dataset_path,
                                       genotype=param_class.genotype, inoculated=param_class.inoculated,
                                       dai=param_class.dai,
                                       test_size=param_class.test_size,
                                       signature_pre_clip=param_class.signature_pre_clip,
                                       signature_post_clip=param_class.signature_post_clip,
                                       max_num_balanced_inoculated=param_class.max_num_balanced_inoculated,
                                       num_samples_file=param_class.num_samples_file,
                                       mode="test",
                                       n_splits=hyperparams['n_splits'],
                                       split=hyperparams['split'],
                                       superpixel=param_class.superpixel)
            dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=0)
            print("Number of batches test", len(dataloader_test))

            model = SANNetwork(input_size=dataset_test.input_size,
                               num_classes=hyperparams['num_classes'],
                               hidden_layer_size=hyperparams['hidden_layer_size'],
                               dropout=0.02,
                               num_heads=hyperparams['num_heads'],
                               device=device)

            epoch = checkpoint['epoch']
            eval_acc = checkpoint['eval_acc']
            print("Loaded model with Acc of {} trained for {} epochs".format(eval_acc, epoch))
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            model.eval()

            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for i, (features, labels) in enumerate(tqdm(dataloader_test)):
                    labels = labels[2]
                    features = features.float().to(device)
                    labels = labels.long().to(device)
                    outputs = model.forward(features)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    batch_pred, batch_target = getPredAndTarget(outputs, labels)
                    correct += balanced_accuracy(batch_target, batch_pred) * labels.size(0)  # mean
                    # correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the test samples: %d %%' % (
                    100 * correct / total))
            """
        results[class_key] = {'all': cv_acc,
                              'mean': round(np.mean(cv_acc).item(), 2),
                              'std': round(np.std(cv_acc).item(), 2)}

    for k in sorted(list(results.keys())):
        print(k, results[k]['mean'], "+-", results[k]['std'])
    # print(results)


def test_attention_acc_on_full_image():
    mpl.rcParams['figure.figsize'] = (20.0, 8.0)

    models_to_test = ["run_dgx2/P22_3_dcf23826-54be-11ea-9a44-0242ac150002",
                      "run_dgx2/WT_3_dcf9eddc-54be-11ea-a0ce-0242ac150002"]
    path = 'uv_dataset/results/{}/best_model.pth.tar'

    checkpoint = torch.load(path.format(models_to_test[0]))
    if 'hyper_params' in list(checkpoint.keys()):
        hyperparams = checkpoint['hyper_params']
    else:
        hyperparams = {
            'genotype': ["WT"],
            'inoculated': [0],  # [0, 1],
            'dai': ["5"],
            'signature_pre_clip': 0,
            'signature_post_clip': 250,
            'test_size': 0.5,
            'max_num_balanced_inoculated': 5000,
            'num_classes': 2,
            'num_heads': 2,
            'hidden_layer_size': 64,
            'savgol_filter_params': [7, 3]
        }
    print(hyperparams)
    # exit()
    if args.dataset == 'gerste':
        dataset = DatasetGerste(genotype=hyperparams['genotype'],
                                inoculated=[0, 1],
                                dai=hyperparams['dai'],
                                test_size=hyperparams['test_size'],
                                signature_pre_clip=hyperparams['signature_pre_clip'],
                                signature_post_clip=hyperparams['signature_post_clip'],
                                max_num_balanced_inoculated=hyperparams['max_num_balanced_inoculated'], mode='test')
    elif args.dataset == 'ruebe':
        dataset = DatasetRuebe(leaf_type=hyperparams['leaf_type'],
                               dai=hyperparams['dai'],
                               test_size=hyperparams['test_size'],
                               signature_pre_clip=hyperparams['signature_pre_clip'],
                               signature_post_clip=hyperparams['signature_post_clip'],
                               max_num_balanced_leafs=hyperparams['max_num_balanced_leafs'], mode='test')
    else:
        raise ValueError('unknown dataset')

    device = "cuda"
    model = SANNetwork(input_size=dataset.input_size,
                       num_classes=hyperparams['num_classes'],
                       hidden_layer_size=hyperparams['hidden_layer_size'],
                       dropout=0.02,
                       num_heads=hyperparams['num_heads'],
                       device="cuda")

    epoch = checkpoint['epoch']
    eval_acc = "Something"  # checkpoint['eval_acc']
    print("Loaded model with Acc of {} trained for {} epochs".format(eval_acc, epoch))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    def forward(x):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)
        return model.forward(x)

    dataset.test_full_image(forward, num_images_per_class=10)


def _create_fig(hyperparams, title="", figsize=(10, 5)):
    fig = plt.figure(figsize=figsize)
    # plt.title(str(hyperparams['genotype'][0]) + " " + title, fontdict={"fontsize": 24})
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    ax.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=26))
    return fig, ax


def test_attention_weights():
    path_template = 'uv_dataset/{}/best_model.pth.tar'
    if args.dataset == 'gerste':
        result_dir = 'cv_superpixel_2run'
        labels = [0, 1]
        label_dict_name = 'genotype'

    elif args.dataset == 'ruebe':
        result_dir = 'results_ruebe/cv_superpixel'
        labels = [0, 1, 2]
        label_dict_name = 'leaftype'

    else:
        raise ValueError('not implemented')
    for checkpoint_dirs_cv in superpixel_trained_models:
        # attn_cv_mean = []
        # attn_cv_std = []

        checkpoint_dirs = ["{}/".format(result_dir) + x for x in checkpoint_dirs_cv]
        i_label = "both"
        #for i_label in labels:
        attn_cv = []
        for checkpoint_dir in checkpoint_dirs:
            path = path_template.format(checkpoint_dir)
            checkpoint = torch.load(path)
            if 'hyper_params' in list(checkpoint.keys()):
                hyperparams = checkpoint['hyper_params']
                if args.dataset == 'ruebe' and 'max_num_balanced_inoculated' in hyperparams.keys():
                    hyperparams['max_num_balanced_leafs'] = hyperparams['max_num_balanced_inoculated']
            else:
                raise ValueError("Should not happen")

            dataset = getDataset('test', hyperparams)

            if hyperparams[label_dict_name] is None:
                hyperparams[label_dict_name] = 'all'

            device = "cuda"
            model = SANNetwork(input_size=dataset.input_size,
                               num_classes=hyperparams['num_classes'],
                               hidden_layer_size=hyperparams['hidden_layer_size'],
                               dropout=0.02,
                               num_heads=hyperparams['num_heads'],
                               device="cuda")

            epoch = checkpoint['epoch']
            eval_acc = checkpoint['eval_acc']
            print("Loaded model with Acc of {} trained for {} epochs".format(eval_acc, epoch))
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            model.eval()

            dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

            wavelength = dataset.wavelength
            plt.close()
            plt.clf()

            palette = sns.color_palette(palette='Set2', n_colors=None, desat=None)

            threshold = 0.05

            # for i, (features, _) in enumerate(dataloader):
            it = iter(dataloader)
            features, labels = next(it)
            features = features.float().to(device)
            labels = labels[2]
            labels = labels.long().to(device)

            #features = features[labels == i_label]
            #labels = labels[labels == i_label]

            attn = model.get_attention(features)

            outputs = model.forward(features)
            outputs = outputs.view(labels.shape[0], -1)

            _, predicted = torch.max(outputs.data, 1)

            print(hyperparams)
            print("Running " + str(hyperparams[label_dict_name]))

            attn_mean = torch.mean(attn, dim=(0,)).detach().cpu().numpy()
            attn_std = torch.std(attn, dim=(0,)).detach().cpu().numpy()
            attn = attn[predicted == labels]
            print("Test num correct", len(attn), "/", len(features))
            attn_cv += attn.detach().cpu().numpy().tolist()
            # attn_cv_mean.append(attn_mean)
            # attn_cv_std.append(attn_std)

        attn_cv = np.array(attn_cv)
        attn_mean = np.mean(attn_cv, axis=(0,))
        attn_std = np.std(attn_cv, axis=(0,))
        fig, ax = _create_fig(hyperparams, "", figsize=(5, 5))

        plt.xlabel("Wavelength [nm]", fontsize=22)
        plt.ylabel("Feature Importance [0-1]", fontsize=22)

        xticks = np.array(wavelength)[attn_mean > threshold]
        y_data = attn_mean[attn_mean > threshold]
        y_err = attn_std[attn_mean > threshold]
        #print(y_err)
        y_err = np.array([[np.minimum(y_data[i], err), err] for i, err in enumerate(y_err)]).T
        #print(y_err)
        # input("press key")
        ax.bar(np.arange(len(y_data)), y_data,
               yerr=y_err, align="center", width=.8,
               ecolor=palette[7],
               capsize=4., )
        # ax.set_xticks(xticks)
        plt.xticks(np.arange(len(y_data)), xticks, rotation='vertical')  #
        plt.ylim(top=0.35)
        save_dir = "uv_dataset/{}/plots_".format(args.dataset) + result_dir + "/{}".format(
            str(hyperparams[label_dict_name]))
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir)
        fig.savefig(save_dir + "/{}_feature_importance.png".format(i_label),
                   bbox_inches='tight', dpi=300)
        #plt.show()
        plt.clf()
        plt.close()
    exit()


def test_attention_weights_image():
    path_template = 'uv_dataset/results/{}/best_model.pth.tar'
    checkpoint_dirs = ["cv_superpixel_2run/WT_superpixel_dai5_260_5789cf2c-6108-11ea-8cea-0242ac150002"]

    for checkpoint_dir in checkpoint_dirs:
        path = path_template.format(checkpoint_dir)
        checkpoint = torch.load(path)
        if 'hyper_params' in list(checkpoint.keys()):
            hyperparams = checkpoint['hyper_params']
            # print(hyperparams)
            # continue
            if hyperparams['n_splits'] not in list(hyperparams.keys()):
                hyperparams['n_splits'] = args.n_splits
            if hyperparams['split'] not in list(hyperparams.keys()):
                hyperparams['split'] = args.split

        else:
            raise ValueError("Should not happen")
        dataset = getDataset('test', hyperparams)

        device = "cuda"
        model = SANNetwork(input_size=dataset.input_size,
                           num_classes=hyperparams['num_classes'],
                           hidden_layer_size=hyperparams['hidden_layer_size'],
                           dropout=0.02,
                           num_heads=hyperparams['num_heads'],
                           device="cuda")

        epoch = checkpoint['epoch']
        eval_acc = checkpoint['eval_acc']
        print("Loaded model with Acc of {} trained for {} epochs".format(eval_acc, epoch))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()

        wavelength = dataset.wavelength

        def forward(x):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)
            return model.forward(x)

        num_images_per_class = 2
        res_samples = dataset.test_full_image(forward, num_images_per_class=num_images_per_class)
        print("Number Images", len(res_samples))
        cnt_selected_inoculated = [0, 0]
        for res_sample in res_samples:
            i_label = res_sample['label']
            save_dir = "uv_dataset/plots_cv/{}_{}/{}_{}".format(hyperparams['n_splits'], hyperparams['split'],
                                                                str(hyperparams['genotype'][0]),
                                                                cnt_selected_inoculated[i_label])
            cnt_selected_inoculated[i_label] += 1
            plt_title = "control" if res_sample['label'] == 0 else "inoculated"

            res_sample_mask = res_sample['mask'].astype(float)
            palette = sns.color_palette(palette='Set2', n_colors=None, desat=None)

            print("Saving to ", save_dir)
            os.makedirs(save_dir, exist_ok=True)

            view = spectral.imshow(res_sample['img'],
                                   classes=res_sample['pred'])

            hs_img_view = view.data_rgb
            plt.clf()
            plt.close()

            f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True, sharex=True)
            ax1.imshow(hs_img_view)

            ax2.imshow(hs_img_view)
            ax2.imshow(res_sample_mask, alpha=1., vmin=0, vmax=2. if res_sample['label'] == 0 else 1.)

            ax3.imshow(hs_img_view)
            ax3.imshow(res_sample['pred'], alpha=1.)
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            ax1.set_aspect('equal')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax2.set_aspect('equal')
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax3.set_aspect('equal')
            ax3.set_xticklabels([])
            ax3.set_yticklabels([])
            # plt.show()
            plt.subplots_adjust(top=0.25, wspace=0, hspace=0.2)

            plt.savefig(save_dir + "/{}_sample_viz.png".format(i_label),
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

            features = res_sample['img'][res_sample['mask'] == 1]
            assert features.shape[0] == np.sum(res_sample['mask'])

            features = np.array([dataset.normalize(f) for f in features])
            features = torch.tensor(features, dtype=torch.float32)
            features = features.to(device)
            attn = model.get_attention(features)

            attn_mean = torch.mean(attn, dim=(0,)).detach().cpu().numpy()
            attn_std = torch.std(attn, dim=(0,)).detach().cpu().numpy()

            fig, ax = _create_fig(hyperparams, plt_title, figsize=(10, 5))

            plt.xlabel("Wavelength [nm]", fontsize=22)
            plt.ylabel("Feature Importance [0-1]", fontsize=22)

            # ax.plot(wavelength, attn_mean, linewidth=4)
            attn_mean_plt = attn_mean.copy()
            attn_mean_plt_inv = attn_mean.copy()
            attn_std_plt = attn_std.copy()
            attn_mean_plt[attn_mean <= 0.05] = 0.
            attn_mean_plt_inv[attn_mean > 0.05] = 0.

            attn_std_plt[attn_mean <= 0.05] = None

            ax.bar(wavelength, attn_mean_plt, align="center",
                   width=3.,
                   ecolor=palette[7],
                   capsize=4.)
            ax.bar(wavelength, attn_mean_plt_inv, align="center", width=1.)

            xticks = np.array(wavelength)[attn_mean > 0.05]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, rotation='vertical')

            fig.savefig(save_dir + "/{}_mean.png".format(i_label),
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

            fig, ax = _create_fig(hyperparams, plt_title, figsize=(5, 5))

            plt.xlabel("Wavelength [nm]", fontsize=22)
            plt.ylabel("Feature Importance [0-1]", fontsize=22)

            xticks = np.array(wavelength)[attn_mean > 0.05]
            y_data = attn_mean[attn_mean > 0.05]
            y_err = attn_std[attn_mean > 0.05]
            ax.bar(np.arange(len(y_data)), y_data,
                   yerr=y_err, align="center", width=.8,
                   ecolor=palette[7],
                   capsize=4.)
            # ax.set_xticks(xticks)
            plt.xticks(np.arange(len(y_data)), xticks, rotation='vertical')
            fig.savefig(save_dir + "/{}_feature_importance.png".format(i_label),
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

            fig, ax = _create_fig(hyperparams, plt_title)

            plt.xlabel("Wavelength [nm]", fontsize=22)
            plt.ylabel("Feature Importance [0-1]", fontsize=22)

            ax.bar(wavelength, attn_std, width=2.)

            # xticks = np.array(wavelength)[attn_mean > 0.05]
            # ax.set_xticks(xticks)
            # ax.set_xticklabels(xticks, rotation='vertical')

            # ax.set_yticks(xticks)

            fig.savefig(save_dir + "/{}_std.png".format(i_label),
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

            feature = features[0:100]
            attn_times_sample = model.forward_attention(feature, return_softmax=False)
            attn_times_sample = torch.mean(attn_times_sample, dim=(0,)).detach().cpu().numpy()

            attn = model.get_attention(features)
            attn_mean = torch.mean(attn, dim=(0,)).detach().cpu().numpy()

            feature = torch.mean(feature, dim=(0,)).detach().cpu().numpy()

            fig, ax = _create_fig(hyperparams, plt_title)

            plt.xlabel("Wavelength [nm]", fontsize=22)
            plt.ylabel("Feature Importance [0-1]", fontsize=22)

            attn_mean_plt = attn.copy()
            attn_mean_plt_inv = attn.copy()
            attn_mean_plt[attn <= 0.05] = 0.
            attn_mean_plt_inv[attn > 0.05] = 0.

            ax.bar(wavelength, attn_mean_plt, align="center",
                   width=3.,
                   ecolor=palette[7],
                   capsize=4.)
            ax.bar(wavelength, attn_mean_plt_inv, align="center", width=1.)

            xticks = np.array(wavelength)[attn > 0.05]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, rotation='vertical')

            fig.savefig(save_dir + "/{}_sample_attn.png".format(i_label),
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

            fig, ax = _create_fig(hyperparams, plt_title)

            plt.xlabel("Wavelength [nm]", fontsize=22)
            # plt.ylabel("Feature Importance [0-1]", fontsize=22)

            ax.plot(wavelength, feature, linewidth=2, alpha=0.6)
            fig.savefig(save_dir + "/{}_sample.png".format(i_label),
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()

            fig, ax = _create_fig(hyperparams, plt_title)
            ax.plot(wavelength, attn_times_sample, linewidth=2, alpha=0.6)
            fig.savefig(save_dir + "/{}_sample_after_attn.png".format(i_label),
                        bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close()


def save_checkpoint(save_dir, state, epoch, best, run_id, filename='checkpoint.pth.tar'):
    save_path_checkpoint = os.path.join(os.path.join(save_dir, run_id), filename)

    os.makedirs(os.path.dirname(save_path_checkpoint), exist_ok=True)
    if epoch % 10 == 0:
        torch.save(state, save_path_checkpoint)
    if best:
        torch.save(state, save_path_checkpoint.replace('checkpoint.pth.tar', 'best_model.pth.tar'))


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    args = parser.parse_args()

    if args.dataset == 'gerste':
        from uv_dataset.hyperparams.hyperparams import get_param_class, dict_classes_keys, superpixel_trained_models
    elif args.dataset == 'ruebe':
        from uv_dataset.hyperparams.hyperparams_ruebe import get_param_class, dict_classes_keys, \
            superpixel_trained_models

    if args.device == "cpu":
        torch.set_num_threads(2)

    global proctitle
    proctitle = "PS43 "
    setproctitle(proctitle + args.mode + " | warming up")

    if args.mode == 'train':
        train_attention()
    elif args.mode == 'test':
        test_model_accuracy()
    elif args.mode == 'train_gb':
        train_gradientboosting()
    elif args.mode == 'test_gb':
        test_gradientboosting()
    elif args.mode == 'full_image_test':
        test_attention_acc_on_full_image()
    elif args.mode == 'attention':
        test_attention_weights()
    elif args.mode == 'attention_image':
        test_attention_weights_image()
    else:
        print("Nothing to do here")
        exit()
