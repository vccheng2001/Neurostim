import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CNN(nn.Module):

    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # conv1d: (in_channels, out_channels, kernel_size)
        # kernel has same width as time series (in this case, 1 feature)
        self.conv1 = nn.Conv1d(self.input_size, 64, kernel_size=1)
        # normalize batches
        self.bn1 = BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 16, kernel_size=1)
        self.bn2 = BatchNorm1d(16)

        self.conv3 = nn.Conv1d(16, 1, kernel_size=1)
        self.dropout1 = nn.Dropout(p=0.1, inplace=False)

        # fully-connected layer 
        self.fc1 = nn.Linear(104, self.output_size) # B x out x 104 
        self.softmax = nn.Softmax(-1) # B x out x 104

    def forward(self, inputs):
        # T: sequence length (timesteps)
        # B: Batch size
        T, B, _ = inputs.shape            # (T, B, In)
        inputs = inputs.permute(1, 2, 0)  # (B, In, T) 
        
        # Input: (batch size, feature_dim, sequence_length)

        x = self.conv1(inputs)            # (B, 16, T)
        x = self.bn1(x)

        x = self.conv2(x)                 # (B, 4 , T)
        x = self.bn2(x)

        x = self.conv3(x)                 # (B, 1 , T)
        x = self.dropout1(x)

        x = x.view(B, -1)                 # (B, T)
        x = self.fc1(x)                   # (B, Out)
        return x