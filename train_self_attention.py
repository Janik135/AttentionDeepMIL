## pure implementation of SANs
## Skrlj, Dzeroski, Lavrac and Petkovic.

"""
The code containing neural network part, Skrlj 2019
"""

import torch
from torch.utils.data import DataLoader

torch.manual_seed(123321)
import tqdm
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np

np.random.seed(123321)

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


class E2EDatasetLoader(Dataset):
    def __init__(self, features, targets=None, transform=None):
        self.features = features.tocsr()

        if not targets is None:
            self.targets = targets  # .tocsr()
        else:
            self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        instance = torch.from_numpy(self.features[index, :].todense())
        if self.targets is not None:
            target = torch.from_numpy(np.array(self.targets[index]))
        else:
            target = None
        return instance, target


def to_one_hot(lbx):
    enc = OneHotEncoder(handle_unknown='ignore')
    return enc.fit_transform(lbx.reshape(-1, 1))


class SANNetwork(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer_size, dropout=0.02, num_heads=2, device="cuda"):
        super(SANNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=0)
        self.activation = nn.SELU()
        self.num_heads = num_heads
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, num_classes)
        self.embedding_pooling = nn.AdaptiveAvgPool2d((1, hidden_layer_size))
        self.instance_pooling = nn.AdaptiveAvgPool2d((1, num_classes))
        self.multi_head = torch.nn.ModuleList([torch.nn.Linear(input_size, input_size) for i in [1] * num_heads])
        self.attention = nn.Sequential(
            nn.Linear(hidden_layer_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        #self.multi_head.to(device)

    def forward_attention(self, input_space, return_softmax=False):
        attention_output_space = []
        for head in self.multi_head:
            if return_softmax:
                attention_output_space.append(self.softmax(head(input_space)))
            else:
                ## this is critical for maintaining a connection to the input space!
                attention_output_space.append(self.softmax(head(input_space)) * input_space)

        ## initialize a placeholder
        placeholder = torch.zeros(input_space.shape)#.to(self.device)

        ## traverse the heads and construct the attention matrix
        for element in attention_output_space:
            placeholder = torch.max(placeholder, element)

        ## normalize by the number of heads
        out = placeholder # / self.num_heads
        return out

    def get_mean_attention_weights(self):
        activated_weight_matrices = []
        for head in self.multi_head:
            wm = head.weight.data
            diagonal_els = torch.diag(wm)
            activated_diagonal = self.softmax2(diagonal_els)
            activated_weight_matrices.append(activated_diagonal)
        output_mean = torch.mean(torch.stack(activated_weight_matrices, axis=0), axis=0)

        return output_mean

    def forward(self, x):

        ## attend and aggregate
        out = self.forward_attention(x)

        ## dense hidden (l1 in the paper)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.activation(out)
        '''
        ## attention pooling
        out = out.squeeze(0)
        A = self.attention(out)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = nn.Softmax(dim=1)(A)  # softmax over N

        M = torch.mm(A, out)
        '''

        out = self.embedding_pooling(out)
        ## dense hidden (l2 in the paper, output)
        out = self.fc3(out)
        #out = self.instance_pooling(out)
        return out

    def get_attention(self, x):
        return self.forward_attention(x, return_softmax=True) / self.num_heads

    def get_softmax_hadamand_layer(self):
        return self.get_mean_attention_weights()

