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
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"device: {device}")

timesteps = {'dreams': 120,
                'mit': 150,
                'dublin': 160}
def main():
    # hyper-parameters
    num_epochs = 50
    batch_size = 64
    lr = 0.001


    # dataset/excerpt parameters 
    root = "data/"
    dataset = "dreams"
    apnea_type="osa"
    excerpt=1
    train_data = ApneaDataset(root,dataset,apnea_type,excerpt)
    train_loader = DataLoader(dataset=train_data, \
                                 batch_size=batch_size,\
                                 shuffle=True)



    num_train = len(train_data)
    # num_test = len(test_data)
    print('Train dataset size: ', num_train)
    # print('Test dataset size: ', num_test)



    # model parameters
    
    n_timesteps = timesteps[dataset]    

    n_outputs = 2
    n_layers = 3
    n_hidden = 64
    model = LSTM(1,n_hidden,n_layers,n_timesteps,n_outputs).double()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optim = Adam(model.parameters(), lr=lr)

    # begin train 
    model.train()
    training_losses = []
    for epoch in range(num_epochs):
        print(f"epoch #{epoch}")
        train_loss = 0.0

        for n_batch, (seq, label, file) in enumerate(train_loader):

            optim.zero_grad()
            
            #print('seq',seq.shape)
            #ts, bs, is
            seq = seq.permute(1,0,2)
            pred = model(seq)
            #print(pred)
            #print('pred/label', pred.shape, label.shape)
            #print(label, 'ggg')
            
            # print(pred.shape, 'pred')
            # print(label.shape, 'lab')
            loss = criterion(pred, label)
            train_loss += loss.item()

            loss.backward()
            optim.step() 
            #scheduler.step()

            if (n_batch + 1) % 2 == 0:
                print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))
            
            # print(f'batch #{n_batch}')
            # print(f'running_loss:{train_loss}')

        # append training loss for each epoch 
        training_losses.append(train_loss/n_batch)       
        print(f"Loss for epoch {epoch}: {train_loss/n_batch}")
    

    # save model
    print("Finished Training")
    # Visualize loss history
    plt.plot(range(num_epochs), training_losses, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    torch.save(model.state_dict(), './model.ckpt')
    
    


if __name__ == "__main__":
    main()
