from glob import glob
from tensorboard.backend.event_processing import event_accumulator as ea
import matplotlib.pyplot as plt
import os
import pandas as pd

dir_path = ["/Users/janik/Documents/TU Darmstadt/21 SS/MA/Experimente/Gerste/5_1000_dropout0_02_7_3/",
            "/Users/janik/Documents/TU Darmstadt/21 SS/MA/Experimente/Gerste/5_1000_dropout0_9_7_3/",
            "/Users/janik/Documents/TU Darmstadt/21 SS/MA/Experimente/Gerste/5_1000_dropout0_9_7_3_overlap/",
            "/Users/janik/Documents/TU Darmstadt/21 SS/MA/Experimente/Gerste/5_1000_dropout0_9_gated_7_3/",
            "/Users/janik/Documents/TU Darmstadt/21 SS/MA/Experimente/Gerste/5_1000_dropout0_9_gated_7_3_overlap/"]


def create_plot(dir_path, label):
    subfolders = glob(os.path.join(dir_path, "*/"))
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for folder in subfolders:
        acc = ea.EventAccumulator(folder)
        acc.Reload()

        # Print tags of contained entities, use these names to retrieve entities as below
        # print(acc.Tags())

        # train_loss = [(s.step, s.value) for s in acc.Scalars('Loss/train')]
        train_loss = [s.value for s in acc.Scalars('Loss/train')]
        train_losses.append(train_loss)
        train_bal_accuracy = [s.value for s in acc.Scalars('Accuracy/train')]
        train_accuracies.append(train_bal_accuracy)
        test_loss = [s.value for s in acc.Scalars('Loss/test')]
        test_losses.append(test_loss)
        test_bal_accuracy = [s.value for s in acc.Scalars('Accuracy/test')]
        test_accuracies.append(test_bal_accuracy)

    df_train_acc = pd.DataFrame(train_accuracies)
    df_train_loss = pd.DataFrame(train_losses)
    df_test_acc = pd.DataFrame(test_accuracies)
    df_test_loss = pd.DataFrame(test_losses)

    df_train_acc_mean = df_train_acc.mean()
    df_train_acc_std = df_train_acc.std()
    df_train_loss_mean = df_train_loss.mean()
    df_train_loss_std = df_train_loss.std()
    df_test_acc_mean = df_test_acc.mean()
    df_test_acc_std = df_test_acc.std()
    df_test_loss_mean = df_test_loss.mean()
    df_test_loss_std = df_test_loss.std()
    '''
    alpha = [0.2, 0.2, 0.2, 0.2, 0.2]
    color = ['#CC4F1B', '#1B2ACC', '#3F7F4C', '#e377c2', '#7f7f7f']
    edgecolor = ['#CC4F1B', '#1B2ACC', '#3F7F4C', '#e377c2', '#7f7f7f']
    facecolor = ['#CC4F1B', '#1B2ACC', '#3F7F4C', '#e377c2', '#7f7f7f']
    ax1.plot(range(len(df_train_acc_mean)), df_train_acc_mean, label=label, color=color[label])
    ax1.fill_between(range(len(df_train_acc_mean)), df_train_acc_mean-df_train_acc_std,
                     df_train_acc_mean+df_train_acc_std, facecolor=facecolor[label], edgecolor=edgecolor[label],
                     alpha=alpha[label])
    ax2.plot(range(len(df_train_loss_mean)), df_train_loss_mean, label=label, color=color[label])
    ax2.fill_between(range(len(df_train_loss_mean)), df_train_loss_mean-df_train_loss_std,
                     df_train_loss_mean+df_train_loss_std, facecolor=facecolor[label], edgecolor=edgecolor[label],
                     alpha=alpha[label])
    ax3.plot(range(len(df_test_acc_mean)), df_test_acc_mean, label=label, color=color[label])
    ax3.fill_between(range(len(df_test_acc_mean)), df_test_acc_mean-df_test_acc_std, df_test_acc_mean+df_test_acc_std,
                     facecolor=facecolor[label], edgecolor=edgecolor[label], alpha=alpha[label])
    ax4.plot(range(len(df_test_loss_mean)), df_test_loss_mean, label=label, color=color[label])
    ax4.fill_between(range(len(df_test_loss_mean)), df_test_loss_mean-df_test_loss_std,
                     df_test_loss_mean+df_test_loss_std, facecolor=facecolor[label], edgecolor=edgecolor[label],
                     alpha=alpha[label])
    '''

    fig = plt.figure()
    plt.suptitle('{}'.format(dir.split('/')[-2]))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.set_title('Bal. train acc')
    ax2.set_title('Train loss')
    ax3.set_title('Bal. test acc')
    ax4.set_title('Test loss')
    ax1.plot(range(len(df_train_acc_mean)), df_train_acc_mean, 'k-', label=label)
    ax1.fill_between(range(len(df_train_acc_mean)), df_train_acc_mean-df_train_acc_std,
                     df_train_acc_mean+df_train_acc_std)
    ax2.plot(range(len(df_train_loss_mean)), df_train_loss_mean, 'k-', label=label)
    ax2.fill_between(range(len(df_train_loss_mean)), df_train_loss_mean-df_train_loss_std,
                     df_train_loss_mean+df_train_loss_std)
    ax3.plot(range(len(df_test_acc_mean)), df_test_acc_mean, 'k-', label=label)
    ax3.fill_between(range(len(df_test_acc_mean)), df_test_acc_mean-df_test_acc_std, df_test_acc_mean+df_test_acc_std)
    ax4.plot(range(len(df_test_loss_mean)), df_test_loss_mean, 'k-', label=label)
    ax4.fill_between(range(len(df_test_loss_mean)), df_test_loss_mean-df_test_loss_std,
                     df_test_loss_mean+df_test_loss_std)
    plt.tight_layout()
    plt.savefig('5-fold-cv_{}.pdf'.format(label))
    plt.show()

'''
fig = plt.figure()
fig.suptitle("5-fold cross validation SAN", fontsize=16)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.set_title('Bal. train acc')
ax2.set_title('Train loss')
ax3.set_title('Bal. test acc')
ax4.set_title('Test loss')
for i, dir in enumerate(dir_path):
    create_plot(dir, ax1, ax2, ax3, ax4, i)
    print('{}: {}'.format(i, dir.split('/')[-2]))
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('5_fold_SAN.pdf')
plt.show()
'''

for i, dir in enumerate(dir_path):
    create_plot(dir, i)
    print('{}: {}'.format(i, dir.split('/')[-2]))
