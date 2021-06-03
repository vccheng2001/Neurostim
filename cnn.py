import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CNN(nn.Module):

    def __init__(self,input_dim,hidden_dim,num_layers,timesteps,output_dim=1):
        super(CNN, self).__init__()

          # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 7 * 7, 10) 

    def forward(self, x):
        # Convolution 1
        output = self.cnn1(x)
        output = self.relu1(out)

        # Max pool 1
        output = self.maxpool1(out)

        # Convolution 2 
        output = self.cnn2(out)
        output = self.relu2(out)

        # Max pool 2 
        output = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # output.size(0): 100
        # New output size: (100, 32*7*7)
        output = output.view(out.size(0), -1)

        # Linear function (readout)
        output = self.fc1(out)
        return output