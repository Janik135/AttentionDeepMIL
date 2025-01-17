import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(2**3*64*95, 256)
        #self.fc1 = nn.Linear(2**3*64*100, 256)
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=0.02)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)

        A = self.attention(out)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = nn.Softmax(dim=1)(A)  # softmax over N

        M = torch.mm(A, out)

        out = self.fc2(M)

        return out
