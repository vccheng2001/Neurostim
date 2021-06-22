
import os
import numpy as np

from dataloader import ApneaDataloader

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import argparse

np.set_printoptions(suppress=True) # don't use scientific notation


'''LSTM model to classify time series signals as apnea/non-apnea events'''

class LSTM_Module(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,num_layers,timesteps,output_dim=1):
        super(LSTM_Module,self).__init__()
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




    
class LSTM:
    def __init__(self, dataset, apnea_type, excerpt, batch_size, epochs):
        # hyper-parameters
        self.init_lr = 0.01
        self.decay_factor = 0.7
        self.pos_pred_threshold = 0.7

        self.dataset = dataset
        self.apnea_type = apnea_type
        self.excerpt = excerpt
        self.batch_size = batch_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.epochs = epochs

        # dataset/excerpt parameters 
        self.save_model_root = "saved_models/"
        predictions_root = "predictions/"
        self.data_root = "data/"


        # dataset 
        self.data = ApneaDataloader(self.data_root,self.dataset,self.apnea_type,self.excerpt, self.batch_size)
        self.train_loader = self.data.get_train()
        self.test_loader = self.data.get_test()

        num_train = len(self.data.train_data)
        num_test = len(self.data.test_data)
        print('Train dataset size: ', num_train)
        print('Test dataset size: ', num_test)

        
        # Model 
        n_timesteps = self.data.dataset.timesteps 
        n_outputs = 2
        n_layers = 3
        n_hidden = 64
        self.model = LSTM_Module(1,n_hidden,n_layers,n_timesteps,n_outputs).double()
        self.model.to(self.device)

        # Loss 
        self.criterion = nn.BCELoss()
        # Optimizer
        self.optim = Adam(self.model.parameters(), lr=self.init_lr)
        # LR scheduler 
        self.scheduler = ReduceLROnPlateau(self.optim, 'min',  factor=self.decay_factor, patience=2)
        
        self.save_base_path = f"{self.save_model_root}{self.dataset}/excerpt{self.excerpt}/{self.dataset}_{self.apnea_type}_ex{self.excerpt}_ep{self.epochs}_b{self.batch_size}_lr{self.init_lr}" 

        if not os.path.isdir(self.save_base_path):
            os.makedirs(self.save_base_path)
        self.save_model_path = self.save_base_path + ".ckpt"

    def train(self):
        self.model.train()
        training_losses = []
        training_errors = []
        for epoch in range(self.epochs):
            print(f"epoch #{epoch}")
            train_loss = 0.0
            train_errors = 0.0

            for n_batch, (seq, label, file) in enumerate(self.train_loader):
                self.optim.zero_grad()
            
                seq = seq.permute(1,0,2)
                
                pred = self.model(seq).unsqueeze(-1).double() # bs x 1 
                label = label.unsqueeze(-1).double()
            
                loss = self.criterion(pred.double(), label.double())

                train_loss += loss.item()
                # writer.add_scalar("Loss/train", train_loss, epoch)
                pred_bin = torch.where(pred > self.pos_pred_threshold, 1, 0)
                N = len(pred)
                errs = torch.count_nonzero(pred_bin - label)
                err_rate = errs/N
                train_errors += err_rate

                
                loss.backward()
                self.optim.step() 
                self.scheduler.step(loss)
                if (n_batch) % 5 == 0:
                    print("Epoch: [{}/{}], Batch: {}, Loss: {}, Acc: {}".format(
                        epoch, self.epochs, n_batch, loss.item(), 1-err_rate))

            # writer.flush()
            # append training loss for each epoch 
            training_losses.append(train_loss/n_batch) 
            training_errors.append(train_errors/n_batch)      
            print(f"Loss for epoch {epoch}: {train_loss/n_batch}")
        
        # Visualize loss history
        plt.plot(range(self.epochs), training_losses, 'r--')
        plt.plot(range(self.epochs), training_errors, 'b-')

        plt.legend(['Training Loss', 'Training error'])
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        # save model
        plt.savefig(self.save_base_path + ".png")
        plt.show()

        print("Saving to... ", self.save_model_path)
        torch.save(self.model.state_dict(), self.save_model_path)

        print('Finished training')

        ############################################################################
    def test(self):
    
        # load trained model
        self.model.load_state_dict(torch.load(self.save_model_path))
        # begin test 
        self.model.eval()
        test_losses = []
        test_errors = []
        print("Testing")
        with torch.no_grad():
            
            for n_batch, (seq, label, file) in enumerate(self.test_loader):
                seq = seq.permute(1,0,2)
                pred = self.model(seq).unsqueeze(-1).double() # bs x 1 
                label = label.unsqueeze(-1).double()

                loss = self.criterion(pred.double(), label.double())

                test_losses += [loss.item()]
                pred_bin = torch.where(pred > 0.5, 1, 0)
                N = len(pred)

                errs = torch.count_nonzero(pred_bin - label)
                err_rate = errs/N
                test_errors += [err_rate]
                # if n_batch % 5 == 0:
                #     np.savetxt(f"{self.save_base_path}test_batch{n_batch}.csv", np.hstack((pred.detach().numpy(), pred_bin.detach().numpy(), label.detach().numpy())), delimiter=",")

                print(f"batch #{n_batch} loss: {loss.item()}, acc: {1-err_rate}")
            avg_test_error = sum(test_errors)/n_batch
            print(f"Average test accuracy: {1-avg_test_error}")
        # Save test errors 
        np.savetxt(self.save_base_path + "_test_errors.csv", np.array([avg_test_error]),  delimiter=",")



l = LSTM('dreams','osa',1,64, 10)
# l.train()
l.test()