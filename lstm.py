'''LSTM model to classify time series signals as apnea/non-apnea events'''
import torch
from torch import nn

class LSTM(nn.Module):
    
    def __init__(self,input_dim, 
                    hidden_dim,
                    num_layers,
                    output_dim,
                    dropout=0):
        super(LSTM,self).__init__()

        self.lstm = nn.LSTM(input_dim,hidden_dim,num_layers,dropout=dropout,batch_first=True)
        self.relu = nn.ReLU(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim,)
        self.softmax = nn.Softmax(-1)
        
    def forward(self,inp):
        x, _ = self.lstm(inp)
        x = self.fc(x)
        x = x[:, -1, :].squeeze()
        x = self.softmax(x)
        return x