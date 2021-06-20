import torch
import torch.nn as nn


'''LSTM model to classify time series signals as apnea/non-apnea events'''

class LSTM(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,num_layers,timesteps,output_dim=1):
        super(LSTM,self).__init__()
        self.input_dim  = input_dim     # timesteps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers 
        self.output_dim = output_dim
        self.timesteps = timesteps
        self.lstm = nn.LSTM(input_dim,hidden_dim,num_layers,dropout=0.2)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.softmax = nn.Softmax(dim=-1)
        # hidden_dim -> output_dim
        # self.bn = nn.BatchNorm1d(self.timesteps)
        
    def forward(self,inputs):
        # x = self.bn(inputs)
        output, _ = self.lstm(inputs)
        #print("Output of LSTM: ", output.shape)
        output = self.fc(output)
        # print("after fc", output.shape)
        output = self.softmax(output)
        # print("after sm", output)
        output = output.permute(1,0,2) # 120, 64, 1
        output = output[:,-1,0]
        return output

# batch_size = 10
# n_timesteps = 128
# n_outputs = 1
# input = torch.randn(batch_size, n_timesteps, n_outputs)
# n_layers = 3
# n_hidden = 64
# lstm = LSTM(1,n_hidden,n_layers,n_outputs).to(device)
# # batch_size, seq_len,hidden_dim -> batch_size, seq_len, 1
# lstm(input)