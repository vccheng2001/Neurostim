
# Util
import os
import numpy as np
import csv
import argparse
from datetime import datetime

# Dataloader
from dataloader import ApneaDataloader
# Models
from cnn import CNN
from lstm import LSTM 

# Torch
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Graphing
import matplotlib.pyplot as plt
# Stats
from scipy.stats import zscore
# don't use scientific notation
np.set_printoptions(suppress=True) 
# logger 
import wandb

class Model:
    def __init__(self, cfg=None):

        self.cfg = cfg
        # hyper-parameters

        self.init_lr = float(cfg.learning_rate)
        self.decay_factor = 0.7

        self.dataset = cfg.dataset
        self.apnea_type = cfg.apnea_type
        self.excerpt = cfg.excerpt
        self.batch_size = int(cfg.batch_size)
        self.epochs = int(cfg.epochs)
        self.sample_rate = int(cfg.sample_rate)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.logger = cfg.logger

        # directories 
        self.root_dir = cfg.root_dir
        self.results_root = os.path.join(self.root_dir, "results/")
        self.results_file = cfg.results_file
        self.save_model_root = os.path.join(self.root_dir, "saved_models/")

        # dataset 
        self.data = ApneaDataloader(cfg)

        # Create train, validation data loaders
        self.train_loader, self.val_loader = self.data.get_data()
        self.num_train = len(self.data.train_data)
        self.num_val = len(self.data.val_data)
        print('Train dataset size: ', self.num_train)
        print('Validation dataset size: ', self.num_val)

        # Default base model path
        self.base_model_file = cfg.base_model_path
        
        # Input/output dimensions
        input_size = 1
        output_size = 2 # binary
    
        # Model type
        if cfg.model_type == "cnn":
            self.model = CNN(input_size=input_size, \
                            output_size=output_size).double()
        else:
            self.model = LSTM(input_size=input_size,
                          output_size=output_size).double()

        self.model.to(self.device)

        # Logger 

        if self.logger: # logger must be True
            wandb.watch(self.model, log_freq=10)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        # self.optim = SGD(self.model.parameters(), lr=self.init_lr, momentum=0.9)
        self.optim = Adam(self.model.parameters(), lr=self.init_lr)

        # LR scheduler 
        self.scheduler = ReduceLROnPlateau(self.optim, 'min',  factor=self.decay_factor, patience=5)
        
        # Save paths 
        self.save_base_path = f"{self.save_model_root}{self.dataset}/excerpt{self.excerpt}/{self.dataset}_{self.apnea_type}_ex{self.excerpt}_ep{self.epochs}_b{self.batch_size}_lr{self.init_lr}" 
        if not os.path.isdir(self.save_base_path):
            os.makedirs(self.save_base_path)
        self.save_model_path = self.save_base_path + ".ckpt"


    # Training/Validation 
    def train(self, save_model=False, 
                    plot_loss=False, 
                    retrain=False):

        self.model.train()

        self.train_losses = []
        self.train_errors = []

        self.val_losses = []
        self.val_errors = []

        # Load pre-trained model to continue training 
        if retrain:
            print('Retraining, loading params')
            self.model.load_state_dict(torch.load(self.base_model_file))

        for epoch in range(self.epochs):
            print(f"Epoch #{epoch}")
            batch_losses = []
            batch_errors = []
            

            ''' -------------Train-------------'''
            for n_batch, (seq, label, file) in enumerate(self.train_loader):
                    
                self.optim.zero_grad()

                print('FILE', file)
                print('seq', seq.shape, seq)
                # feed sequence of dim (B, T, C) through model, outputs a prediction
                pred = self.model(seq)
                # binary prediction
                pred_bin = torch.argmax(pred, dim=1)

                print(pred)
                print(pred_bin)
                print(label)
                print('\n')

                # compute loss using prediction, label
                loss = self.criterion(pred, label)
                batch_losses += [loss.item()]

            

                # calculate error rate across current batch 
                errs = torch.count_nonzero(pred_bin - label)
                err_rate = errs/len(pred_bin)
                batch_errors += [err_rate]
                
                loss.backward()
                self.optim.step() 

                # self.scheduler.step(loss)

                # Log 
                if (n_batch) % 5 == 0:
                    print("Epoch: [{}/{}], Batch: {}, Loss: {}, Acc: {}".format(
                        epoch, self.epochs, n_batch, loss.item(), 1-err_rate))


            epoch_loss = np.mean(batch_losses)
            epoch_errs = np.mean(batch_errors) 

            # Log epoch train loss, errors 
            if self.logger:
                wandb.log({"train_loss": epoch_loss})
                wandb.log({"train_errors": epoch_errs})

            self.train_losses.append(epoch_loss)
            self.train_errors.append(epoch_errs)

            print(f"Train Loss for epoch {epoch}: {epoch_loss}")
        

            # '''------------------Validation--------------'''
            # with torch.no_grad():
            #     self.model.eval()

            #     batch_val_losses = []
            #     batch_val_errors = []

            #     for n_batch, (seq, label, file) in enumerate(self.val_loader):

            #         # feed sequence of dim (B, T, C) through model, outputs a prediction
            #         pred = self.model(seq)
            #         # binary prediction
            #         pred_bin = torch.argmax(pred, dim=1)

            #         # compute loss using prediction, label
            #         loss = self.criterion(pred, label)
            #         batch_val_losses += [loss.item()]

            #         # calculate error rate across current batch 
            #         errs = torch.count_nonzero(pred_bin - label)
            #         err_rate = errs/len(pred_bin)
            #         batch_val_errors += [err_rate]
        
            #     epoch_val_loss = np.mean(batch_val_losses)
            #     epoch_val_errs = np.mean(batch_val_errors) 

            #     # Log 
            #     # if self.logger:
            #         # wandb.log({"val_loss": epoch_val_loss})
            #         # wandb.log({"val_errors": epoch_val_errs})

            #     # Log epoch val loss, errors
            #     self.val_losses.append(epoch_val_loss)
            #     self.val_errors.append(epoch_val_errs)
            #     print(f"Validation loss for epoch {epoch}: {epoch_val_loss}")
        
        # '--------------Done training------------'
        # self.final_test_acc = 1 - self.val_errors[-1]
        # print('Final validation accuracy', self.final_test_acc)

        ''' ---------- Plot losses ----------- '''
        if plot_loss:
            plt.plot(range(self.epochs), self.train_losses, 'r--')
            # plt.plot(range(self.epochs), self.val_losses, 'b-')

            plt.legend(['Training Loss'])#, 'Validation loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # save model
            # plt.savefig(self.save_base_path + ".png")
            plt.show()


        self.model.eval()
        test_losses = []
        test_errors = []
        print("Testing")
        with torch.no_grad():
            test_errors = []
            
            for n_batch, (seq, label, file) in enumerate(self.val_loader):
                pred = self.model(seq)

                # binary prediction
                pred_bin = torch.argmax(pred, dim=1)


                # compute loss using prediction, label
                test_loss = self.criterion(pred, label)
                test_losses += [test_loss.item()]

                print('pred',pred_bin)
                print('label:',label)

                # calculate error rate across current batch 
                errs = torch.count_nonzero(pred_bin - label)
                err_rate = errs/len(pred_bin)
                test_errors.append(err_rate)

                if self.logger:
                    wandb.log({"batch_val_loss": loss.item()})
                    wandb.log({"batch_val_errors": err_rate})

                # np.savetxt(f"{self.save_base_path}test_batch{n_batch}.csv", np.hstack((pred.detach().numpy(), pred_bin.detach().numpy(), label.detach().numpy())), delimiter=",")
                print(f"batch #{n_batch} loss: {test_loss.item()}, acc: {1-err_rate}")

            figure, axes = plt.subplots(nrows=1, ncols=4, figsize=(12,12))
            # plot 
            for i in range(4):
                t = torch.arange(0, seq.shape[1] / self.sample_rate, 1/self.sample_rate)
                v = seq[i]
                axes[i].plot(t.squeeze().numpy(), v.squeeze().numpy())
                axes[i].set_title(f'Pred: {pred_bin[i]}, Label: {label[i]}')
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.25)
            plt.xlabel(f'Timestamp (seconds)')
            plt.ylabel('Signal Value')
            plt.show()

            self.final_test_error = np.mean(test_errors)

            self.final_test_acc = 1 - self.final_test_error
            print('final test error:', self.final_test_error)
            if self.logger:
                wandb.log({"final_test_error": self.final_test_error})
            print(f"Average test accuracy: {1-self.final_test_error}")



        if save_model:

            if retrain:
                print("Retrain, saving to base model... ", self.base_model_file)
                torch.save(self.model.state_dict(), self.base_model_file)
            else:
                print("Saving to... ", self.save_model_path)
                torch.save(self.model.state_dict(),self.save_model_path)

        print('Finished training')
        



        results_file = self.results_root + self.results_file
        print(f'----------Writing results to {results_file}----------')

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
                            'test_acc': self.final_test_acc,
                            'file': file_relpath,
                            'n_train': self.num_train,
                            'n_test': self.num_val,
                            'epochs': self.epochs})

        return self.train_losses, self.val_losses, self.final_test_acc
