
import os
import numpy as np
import csv
from dataloader import ApneaDataloader

from cnn import CNN
from lstm import LSTM 

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from scipy.stats import zscore
np.set_printoptions(suppress=True) # don't use scientific notation

# logger 
import wandb
from wandb import init, log, join 

class Model:
    def __init__(self, root_dir=".",
                       dataset="dreams",
                       apnea_type="osa",
                       excerpt=1, 
                       batch_size=16, 
                       epochs=10,
                       config=None):

        self.config=config
        # hyper-parameters
        self.init_lr = 0.0015
        self.decay_factor = 0.9

        self.dataset = dataset
        self.apnea_type = apnea_type
        self.excerpt = excerpt
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # directories 
        self.root_dir = root_dir
        self.data_root = os.path.join(self.root_dir, "data/")
        self.results_root = os.path.join(self.root_dir, "results/")
        self.save_model_root = os.path.join(self.root_dir, "saved_models/")

        # dataset 
        self.data = ApneaDataloader(self.data_root, 
                                    self.dataset,
                                    self.apnea_type,
                                    self.excerpt,
                                    self.batch_size)
        self.train_loader, self.val_loader = self.data.get_data()

        self.num_train = len(self.data.train_data)
        self.num_val = len(self.data.val_data)
        print('Train dataset size: ', self.num_train)
        print('Validation dataset size: ', self.num_val)

        self.base_model_path = "base_model.ckpt"

        
        # Model 
        input_size = 1
        output_size = 2 # binary
    

        # Model
        # self.model = LSTM(input_size=input_size,
                        #   output_size=output_size).double()
        self.model = CNN(input_size=input_size, \
                         output_size=output_size).double()
        
        # self.model.load_state_dict(torch.load(self.save_model_path))
        self.model.to(self.device)

        if self.config is not None:
            wandb.watch(self.model, log_freq=10)

        # Loss
        self.criterion = nn.CrossEntropyLoss() # WithLogitsLoss()

      
        # Optimizer
        self.optim = SGD(self.model.parameters(), lr=self.init_lr, momentum=0.9)


        self.optim = Adam(self.model.parameters(), lr=self.init_lr)
        # LR scheduler 
        self.scheduler = ReduceLROnPlateau(self.optim, 'min',  factor=self.decay_factor, patience=3)
        
        # Save paths 
        self.save_base_path = f"{self.save_model_root}{self.dataset}/excerpt{self.excerpt}/{self.dataset}_{self.apnea_type}_ex{self.excerpt}_ep{self.epochs}_b{self.batch_size}_lr{self.init_lr}" 
        if not os.path.isdir(self.save_base_path):
            os.makedirs(self.save_base_path)
        self.save_model_path = self.save_base_path + ".ckpt"

    def train(self, save_model=False, plot_loss=False, retrain=False):
        self.model.train()
        self.train_losses = []
        self.train_errors = []

        self.val_losses = []
        self.val_errors = []

        if retrain:
            print('Retraining, loading params')
            self.model.load_state_dict(torch.load(self.base_model_path))

        for epoch in range(self.epochs):
            print(f"Epoch #{epoch}")
            batch_losses = []
            batch_errors = []

            # seq: (B, T, C)
            ''' -------------Train-------------'''
            for n_batch, (seq, label, file) in enumerate(self.train_loader):

                self.optim.zero_grad()
            
                pred = self.model(seq)

                loss = self.criterion(pred, label)

                batch_losses += [loss.item()]
                pred_bin = torch.argmax(pred, dim=1)
                errs = torch.count_nonzero(pred_bin - label)
                err_rate = errs/len(pred_bin)
                batch_errors += [err_rate]
                
                loss.backward()
                self.optim.step() 

                # self.scheduler.step(loss)
                if (n_batch) % 5 == 0:
                    print("Epoch: [{}/{}], Batch: {}, Loss: {}, Acc: {}".format(
                        epoch, self.epochs, n_batch, loss.item(), 1-err_rate))


            epoch_loss = np.mean(batch_losses)
            epoch_errs = np.mean(batch_errors) 
            if self.config is not None:
                wandb.log({"train_loss": epoch_loss})
                wandb.log({"train_errors": epoch_errs})

            # append training loss, errors for each epoch 
            self.train_losses.append(epoch_loss)
            self.train_errors.append(epoch_errs)

            print(f"Train Loss for epoch {epoch}: {epoch_loss}")
        

            '''------------------Validation--------------'''
            with torch.no_grad():
                self.model.eval()
                batch_val_losses = []
                batch_val_errors = []
                for n_batch, (seq, label, file) in enumerate(self.val_loader):

                    pred = self.model(seq)
                    loss = self.criterion(pred, label)
                    batch_val_losses += [loss.item()]
                    pred_bin = torch.argmax(pred, dim=1)
                    # print('pred bin', pred_bin)
                    # print('label', label)
                    errs = torch.count_nonzero(pred_bin - label)
                    err_rate = errs/len(pred_bin)
                    batch_val_errors += [err_rate]

        
                epoch_val_loss = np.mean(batch_val_losses)
                epoch_val_errs = np.mean(batch_val_errors) 
                if self.config is not None:
                    wandb.log({"val_loss": epoch_val_loss})
                    wandb.log({"val_errors": epoch_val_errs})

                # append val loss, errors for each epoch 
                self.val_losses.append(epoch_val_loss)
                self.val_errors.append(epoch_val_errs)
                print(f"Validation loss for epoch {epoch}: {epoch_val_loss}")
        
        '--------------Done training------------'
        final_val_acc = 1-self.val_errors[-1]
        print('Final validation accuracy', final_val_acc)

        ''' ---------- Plot losses ----------- '''
        if plot_loss:
            plt.plot(range(self.epochs), self.train_losses, 'r--')
            plt.plot(range(self.epochs), self.val_losses, 'b-')

            plt.legend(['Training Loss', 'Validation loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # save model
            # plt.savefig(self.save_base_path + ".png")
            plt.show()

        if save_model:

            if retrain:
                print("Retrain, saving to base model... ", self.base_model_path)
                torch.save(self.model.state_dict(), self.base_model_path)
            else:
                print("Saving to... ", self.save_model_path)
                torch.save(self.model.state_dict(), self.save_model_path)

        print('Finished training')
        



    
        
        # # load trained model
        # # self.model.load_state_dict(torch.load(self.save_model_path))

        # # begin test 
        # self.model.eval()
        # test_losses = []
        # test_errors = []
        # print("Testing")
        # with torch.no_grad():
            
        #     for n_batch, (seq, label, file) in enumerate(self.test_loader):
            
        #         pred = self.model(seq) 

        #         loss = self.criterion(pred, label)

        #         test_losses += [loss.item()]                
        #         wandb.log({"test_loss": loss.item()})

        #         pred_bin = torch.argmax(pred, dim=1)

        #         # print('Test Label: ', label.shape, label)
        #         # print('Test Pred: ', pred_bin.shape, pred_bin)

        #         errs = torch.count_nonzero(pred_bin - label)

        #         err_rate = errs/len(pred_bin)

        #         test_errors.append(err_rate)
        #         wandb.log({"test_error": err_rate})

        #         print(f"batch #{n_batch} loss: {loss.item()}, acc: {1-err_rate}")

        #     self.avg_test_error = np.mean(test_errors)

        #     print(f"Average test accuracy: {1-self.avg_test_error}")

        #     # plt.plot(range(n_batch), test_losses, 'g--')
        #     # plt.legend(['Test Loss'])
        #     # plt.xlabel('Batch #')
        #     # plt.ylabel('Test loss')
        #     # # save model
        #     # plt.show()


        # write new row to log.txt 
        results_file = self.results_root + "results.csv"
        file_relpath = os.path.relpath(results_file, self.root_dir)
        with open(results_file, 'a', newline='\n') as results:
            fieldnames = ['time','dataset','apnea_type','excerpt', \
                      'file','test_acc','n_train','n_test','epochs']
            writer = csv.DictWriter(results, fieldnames=fieldnames)
            print('Writing row....\n')
            time_format = '%m/%d/%Y %H:%M %p'
            writer.writerow({'time': datetime.now().strftime(time_format),
                            'dataset': self.dataset,
                            'apnea_type': self.apnea_type,
                            'excerpt': self.excerpt,
                            # 'sample_rate': self.sample_rate,
                            # 'scale_factor': self.scale_factor,

                            # val error of last epoch
                            'test_acc': final_val_acc,
                            'file': file_relpath,
                            'n_train': self.num_train,
                            'n_test': self.num_val,
                            'epochs': self.epochs})

        return self.train_losses, self.val_losses, final_val_acc
