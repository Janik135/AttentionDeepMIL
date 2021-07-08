import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.ReLU(inplace=True)
        #return nn.PReLU(nchan)


# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(nchan)  # ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(16)  # ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels

        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)

        out = self.relu1(torch.add(out, x16))

        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, outChans=0, dropout=False):
        super(DownTransition, self).__init__()
        if outChans == 0:
            outChans = 4 * inChans

        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans)  # ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans // 2)  # ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(2)  # ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class ConvNetBarley(nn.Module):
    def __init__(self, elu=True, avgpool=False, nll=False, num_classes=8):
        super(ConvNetBarley, self).__init__()
        if avgpool:
            self.last_wv = 1
            self.features = nn.Sequential(
                InputTransition(16, elu),
                DownTransition(16, 1, elu),
                DownTransition(32, 2, elu),
                DownTransition(64, 3, elu, dropout=True),
                DownTransition(128, 2, elu, dropout=True),
                nn.AvgPool3d(kernel_size=(4, 1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 * 1 * 7 * 7, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 2)
            )
        else:
            self.last_wv = 11
            self.features = nn.Sequential(
                InputTransition(16, elu),
                DownTransition(16, 1, elu),
                DownTransition(64, 2, elu, dropout=True)
            )
            self.attention = nn.Sequential(
                nn.Linear(256 * 102 * 6 * 3, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 * 102 * 6 * 3, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        features = self.features(x)
        features = features.view(-1, 256 * 102 * 6 * 3)
        A = self.attention(features)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = nn.Softmax(dim=1)(A)  # softmax over N

        M = torch.mm(A, features)
        out = self.classifier(M)
        return out
