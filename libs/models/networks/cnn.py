import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride
                    , padding, bias=False):
        super(ResBlock, self).__init__()

        self.conv0 = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=1, stride=stride,
                                    padding=0, bias=bias)
        
        self.conv1 = nn.Conv1d(out_channels, out_channels,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=0.9)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x   = self.conv0(x)
        out = self.conv1(x)
        # out = self.bn1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out + x


class CNN(nn.Module):
    def __init__(self, input_dim, num_classes, conv_layers, **kwargs):
        super(CNN, self).__init__()
        kernel_size = kwargs["kernel_size"]
        fc1_out_features = kwargs["latent_feat"]
        dropout_rate = kwargs["dropout_rate"]
        pool_size = kwargs["pool_size"]

        self.conv_layers = nn.ModuleList()
        in_channels = 1  
        for out_channels in conv_layers:

            conv_block = ResBlock(in_channels, 
                                  out_channels, 
                                  kernel_size, 
                                  stride=1,
                                  padding=kernel_size // 2)

            self.conv_layers.append(conv_block)
            in_channels = out_channels  # Actualizar in_channels para la próxima capa
        
        self.pool = nn.MaxPool1d(pool_size)
        # self.pool = nn.AvgPool1d(pool_size)
        self.fc1_size = conv_layers[-1] * (input_dim // pool_size ** len(conv_layers))
        
        self.fc1 = nn.Linear(self.fc1_size, fc1_out_features)
        self.fc2 = nn.Linear(fc1_out_features, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # match the input shape to the conv1d input shape
        
        for conv in self.conv_layers:
            x = conv(x)
            x = self.pool(x)
            
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))

        if self.training:
            x = self.dropout(x)

        x = self.fc2(x)
        return x
