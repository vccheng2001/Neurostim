import time
import math
from lstm import LSTM
from dataloader import ApneaDataset
import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"device: {device}")

timesteps = {'dreams': 120,
                'mit': 150,
                'dublin': 160}

def main():
    # hyper-parameters
    num_epochs = 10
    batch_size = 128
    lr = 0.001
    # loss balancing factor 
    alpha = 0.5

    print(f"Params: epochs: {num_epochs}, batch: {batch_size}, lr: {lr}, alpha: {alpha}\n")

    # dataset/excerpt parameters 
    root = "data/"
    dataset = "dreams"
    apnea_type="osa"
    excerpt=1
    train_data = ApneaDataset(root,dataset,apnea_type,excerpt)
    train_loader = DataLoader(dataset=train_data, \
                                 batch_size=batch_size,\
                                 shuffle=False)



    num_train = len(train_data)
    # num_test = len(test_data)
    print('Train dataset size: ', num_train)
    # print('Test dataset size: ', num_test)



    # model parameters + hyperparameters 
    
    n_timesteps = timesteps[dataset]    

    n_outputs = 1
    n_layers = 3

    batch_size = 10
    n_hidden = 64
    model = LSTM(1,n_hidden,n_layers,n_timesteps,n_outputs).double()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optim = Adam(model.parameters(), lr=lr)

    # begin train 
    model.train()
    for epoch in range(num_epochs):
        print(f"epoch #{epoch}")
        loss_epoch = []
        running_loss = 0.0

        for n_batch, (seq, label, file) in enumerate(train_loader):
            optim.zero_grad()
            
            print('seq',seq.shape)
            pred = model(seq)
            print('pred', pred)
            loss = criterion(pred, label)
            loss.backward()
            optim.step() 
            #scheduler.step()
            
            running_loss += loss.item()
            print(f'batch #{n_batch}')
            print(f'running_loss:{running_loss}')

        loss_epoch += [running_loss]
    
    # save model
    print("Finished Training")
    torch.save(model.state_dict(), './model.ckpt')
    
    


if __name__ == "__main__":
    main()
