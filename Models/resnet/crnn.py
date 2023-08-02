import torch
from torch import nn


# Recall: output_size = (input_size - kernel_size + 2*padding) // stride + 1

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=125, stride=stride, padding=62, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=61, padding=30, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class CRNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CRNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=64, out_channels=8, kernel_size=7, stride=2, padding=3)        # output preserves temporal size
        self.bn = nn.BatchNorm1d(num_features=8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_residual_block(in_channels=8, out_channels=4, stride=1)
        self.rnn = nn.LSTM(input_size=4, hidden_size=16, num_layers=2, batch_first=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(4 + 16, num_classes)  # input size is sum of layer1 and rnn output sizes

    def make_residual_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            # ResidualBlock(out_channels, out_channels, stride=1)
        )

    def forward(self, x):
        out = self.conv(x)  # out: (batch, 64, 251)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out_layer1 = self.avg_pool(out)  # output of layer1, will be concatenated with output of rnn
        out_layer1 = out_layer1.view(out_layer1.size(0), -1)        # this should be (batch, 4)

        out = out.permute(0, 2, 1)  # RNN expects batch_size, seq_len, input_size
        out, _ = self.rnn(out)  # out: (batch, seq_len, hidden_size)
        out_rnn = out[:, -1, :]  # get output of last RNN unit (batch, 16)

        out = torch.cat((out_layer1, out_rnn), dim=1)  # concatenate along the channel dimension (batch, 4 + 16)
        out = self.fc(out)
        return out
