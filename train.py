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
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"device: {device}")

timesteps = {'dreams': 120,
                'mit': 150,
                'dublin': 160}
def main():
    # hyper-parameters
    num_epochs = 30
    batch_size = 32
    
    init_lr = 0.01
    decay_factor = 0.7
    test_frac = 0.3


    # dataset/excerpt parameters 
    root = "data/"
    dataset = "dreams"
    apnea_type="osa"
    excerpt=1
    data = ApneaDataset(root,dataset,apnea_type,excerpt)
    train_data, test_data = data.get_splits(test_frac)
    # prepare data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    



    num_train = len(train_data)
    num_test = len(test_data)
    print('Train dataset size: ', num_train)
    print('Test dataset size: ', num_test)



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
        
            loss = criterion(pred.double(), label.double())

            train_loss += loss.item()
            pred_bin = torch.where(pred > 0.5, 1, 0)
            N = len(pred)
            # print(pred_bin, label)

            errs = torch.count_nonzero(pred_bin - label)
            err_rate = errs/N
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

    # begin test 
    model.eval()
    test_losses = []
    test_errors = []
    print("Testing")
    with torch.no_grad():
        
        for n_batch, (seq, label, file) in enumerate(test_loader):
            seq = seq.permute(1,0,2)
            pred = model(seq).unsqueeze(-1).double() # bs x 1 
            label = label.unsqueeze(-1).double()

            loss = criterion(pred.double(), label.double())

            test_losses += [loss.item()]
            pred_bin = torch.where(pred > 0.5, 1, 0)
            N = len(pred)
            # print(pred_bin, label)

            errs = torch.count_nonzero(pred_bin - label)
            err_rate = errs/N
            test_errors += [err_rate]

            print(f"batch #{n_batch} loss: {loss.item()}, acc: {1-err_rate}")
    
    
    with open("test_loss.txt", "wb") as fp_test:   #Pickling
        pickle.dump(test_losses, fp_test)

    # Visualize loss history
    plt.plot(range(num_epochs), training_losses, 'r--')
    plt.plot(range(num_epochs), training_errors, 'b-')

    plt.legend(['Training Loss', 'Training error'])
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.show()

    save_model_path = './final_model.ckpt'
    print("Saving to... ", save_model_path)
    torch.save(model.state_dict(), save_model_path)
    
    


if __name__ == "__main__":
    main()
