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
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import matplotlib.pyplot as plt
import pickle
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"device: {device}")

timesteps = {'dreams': 104,
                'mit': 120,
                'dublin': 160}
def main():
    print('in train, cwd: ', os.getcwd())
    # hyper-parameters
    num_epochs = 15
    batch_size = 32
    
    init_lr = 0.01
    decay_factor = 0.7
    test_frac = 0.2
    pos_pred_threshold = 0.7


    # dataset/excerpt parameters 

    save_model_root = "saved_models/"
    predictions_root = "predictions/"

    data_root = "data/"

    data = ApneaDataset(data_root,dataset,apnea_type,excerpt)
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
            writer.add_scalar("Loss/train", train_loss, epoch)
            pred_bin = torch.where(pred > pos_pred_threshold, 1, 0)
            N = len(pred)
            # print(pred_bin, label)

            # check prediction output 
            # np.savetxt("sample_out.csv", np.hstack((pred.detach().numpy(), pred_bin.detach().numpy(), label.detach().numpy())), delimiter=",")

            errs = torch.count_nonzero(pred_bin - label)
            err_rate = errs/N
            train_errors += err_rate

            
            loss.backward()
            optim.step() 
            scheduler.step(loss)
            if (n_batch) % 5 == 0:
                print("Epoch: [{}/{}], Batch: {}, Loss: {}, Acc: {}".format(
                    epoch, num_epochs, n_batch, loss.item(), 1-err_rate))

        writer.flush()
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
        avg_test_error = sum(test_errors)/n_batch
        print(f"Average test accuracy: {1-avg_test_error}")
    
    with open("test_loss.txt", "wb") as fp_test:   #Pickling
        pickle.dump(test_losses, fp_test)

    # Visualize loss history
    plt.plot(range(num_epochs), training_losses, 'r--')
    plt.plot(range(num_epochs), training_errors, 'b-')

    plt.legend(['Training Loss', 'Training error'])
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.show()


    # save model, predictions, train loss image

    save_base_path = f"{save_model_root}{dataset}/excerpt{excerpt}/{apnea_type}_ep_{num_epochs}_b_{batch_size}_lr_{init_lr}" 
    if not os.path.isdir(save_base_path):
        os.makedirs(save_base_path)
    save_model_path = save_base_path + ".ckpt"
    print("Saving to... ", save_model_path)
    torch.save(model.state_dict(), save_model_path)


    save_pred_path = save_base_path + ".csv"
    np.savetxt(save_pred_path + ".csv", np.array([avg_test_error]),  delimiter=",")

    train_loss_img = save_base_path +".png"
    plt.savefig(train_loss_img)

    writer.close()
    


if __name__ == "__main__":
    ''' parses command line arguments, runs main() '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",    default="dreams", help="dataset (dreams, dublin, or mit)")
    parser.add_argument("-a", "--apnea_type", default="osa",    help="type of apnea (osa, osahs, or all)")
    parser.add_argument("-ex","--excerpt",    default=1,        help="excerpt number to use")
    parser.add_argument("-ep","--epochs",     default=10,       help="number of epochs to train")
    parser.add_argument("-b", "--batch_size", default=16,       help="batch size")
    parser.add_argument('--test', action='store_true', help="only make prediction using existing model")

    # parse args 
    args = parser.parse_args()

    # print(args)
    # store args 
    dataset   = args.dataset
    apnea_type = args.apnea_type
    excerpt = int(args.excerpt)
    epochs    = args.epochs
    batch_size  = args.batch_size
    test        = args.test 
    labels      = {'positive/':1, 'negative/':0}
    test_frac =  0.3 # default ratio for train-test-split
    main()
