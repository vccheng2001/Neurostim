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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"device: {device}")

timesteps = {'dreams': 120,
                'mit': 150,
                'dublin': 160}
def main():
    # hyper-parameters
    num_epochs = 25
    batch_size = 32
    
    init_lr = 0.01
    decay_factor = 0.7


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

    criterion = nn.BCELoss()
    # Define the optimizer
    optim = Adam(model.parameters(), lr=init_lr)
    scheduler = ReduceLROnPlateau(optim, 'min',  factor=decay_factor, patience=2)

    # begin train 
    model.train()
    training_losses = []
    training_errors = []
    for epoch in range(num_epochs):
        print(f"epoch #{epoch}")
        train_loss = 0.0
        train_errors = 0.0

        for n_batch, (seq, label, file) in enumerate(train_loader):

            optim.zero_grad()
         
            seq = seq.permute(1,0,2)
            pred = model(seq).unsqueeze(-1).double() # bs x 1 
            label = label.unsqueeze(-1).double()
            # print(pred, 'pppp')
            # print(label, 'llll')
            loss = criterion(pred.double(), label.double())

            train_loss += loss.item()
            pred_bin = torch.where(pred > 0.5, 1, 0)
            N = len(pred)
            # print(pred_bin, label)
            err_rate = torch.count_nonzero(pred_bin - label) / N
            train_errors += err_rate

            loss.backward()
            optim.step() 
            scheduler.step(loss)
            if (n_batch + 1) % 5 == 0:
                print("Epoch: [{}/{}], Batch: {}, Loss: {}, Acc: {}".format(
                    epoch, num_epochs, n_batch, loss.item(), 1-err_rate))
            

        # append training loss for each epoch 
        training_losses.append(train_loss/n_batch) 
        training_errors.append(train_errors/n_batch)      
        print(f"Loss for epoch {epoch}: {train_loss/n_batch}")
    

    # save model
    print("Finished Training")
    # Visualize loss history
    plt.plot(range(num_epochs), training_losses, 'r--')
    plt.plot(range(num_epochs), training_errors, 'b-')

    plt.legend(['Training Loss', 'Training error'])
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.show()
    torch.save(model.state_dict(), './model.ckpt')
    
    


if __name__ == "__main__":
    main()
