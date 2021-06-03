import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,timesteps,output_dim=1):