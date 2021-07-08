import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d


class GaussianNoise():
    
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def add_noise(self, x):
        return x + torch.randn(x.size()) * self.std + self.mean


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.gn = GaussianNoise(mean=0, std=0.01)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.PReLU()
        
    def forward(self, x):
        x = self.gn.add_noise(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
        


class CNN(nn.Module):

    def __init__(self, input_size, 
                       output_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        print('-------block 1---------------')
        self.block1 = nn.Sequential(

                # input: NumSamples, In_Channels, SeqLen
                nn.Conv1d(in_channels=input_size,
                out_channels=128,
                kernel_size=64, # num steps to slide across at each pass
                stride=32,
                bias=True),
             
                nn.BatchNorm1d(128),

                # nn.MaxPool1d(kernel_size=4, stride=2),

                nn.Flatten()
        )

        print('-------block 2---------------')

        self.block2 = nn.Sequential(

                nn.Conv1d(in_channels=input_size,
                out_channels=64,
                kernel_size=32,
                stride=2,
                bias=True),
             
                nn.BatchNorm1d(64),

                nn.MaxPool1d(kernel_size=4, stride=2),

                nn.Flatten()
        )
        print('-------block 3---------------')

        self.block3 = nn.Sequential(

                nn.Conv1d(in_channels=input_size,
                out_channels=8,
                kernel_size=4,
                stride=2,
                bias=True),
             
                nn.BatchNorm1d(8),

                nn.MaxPool1d(kernel_size=4, stride=2),

                nn.Flatten()
        )
        
        print('-------block 4---------------')
        self.block4 = nn.Sequential(

                nn.Conv1d(in_channels=input_size,
                out_channels=4,
                kernel_size=2,
                stride=1,
                bias=True),
             
                nn.BatchNorm1d(4),

                nn.MaxPool1d(kernel_size=4, stride=2),

                nn.Flatten()
        )
        

    
        self.flatten = nn.Flatten()



        print('-------classification layer-------------')

        self.fc1 = nn.Linear(1208, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # self.dropout = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(128, self.output_size) 
        self.softmax = nn.Softmax(dim=-1)


    
    def forward(self, inp):
        
        inp = inp.permute(0,2,1) # permute to become (N, C, T)
        
        B, C, T = inp.shape

        x = self.block1(inp)  
        xx = self.block2(inp)
        xxx = self.block3(inp)  
        xxxx = self.block4(inp)
        
        flat_inp = self.flatten(inp)
        x = torch.cat([x, xx, xxx, xxxx, flat_inp], -1)

        # classification layer 
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.dropout(x)

        out = self.fc3(x)
        out = self.softmax(out)
        return out
