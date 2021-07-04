import argparse
from flatline_detection import FlatlineDetection
from lstm import LSTM 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import plotly
from plotly.subplots import make_subplots

'''
Program to run end-to-end flatline detection (non-GUI)
To view args: python3 end2end.py -h 
Example command: python3 end2end.py -d dreams -a osa -ex 1 -sr 8 -sc 1 -f -b 64 -ep 10
'''

def main(args):

    print('********************************')
    print(f'****  Dataset: {args.dataset}  ****')
    print(f'****  Apnea type: {args.apnea_type}  ****')
    print(f'****  Excerpt: {args.excerpt}  ****')
    print(f'****  Sample rate: {args.sample_rate}  ****')
    print(f'****  Scale factor: {args.scale_factor}  ****')
    print('*********************************')




    # visualize original data
    fd = FlatlineDetection(root_dir=".",
                           dataset=args.dataset,
                           apnea_type=args.apnea_type,
                           excerpt= args.excerpt,
                           sample_rate=args.sample_rate,
                           scale_factor=args.scale_factor)

    print('----------------Visualize original signal--------------------')

    fig = fd.visualize()
    print('----------------Flatline Detection---------------------')

    # extract flatline events
    
    flatline_fig, flatline_times, nonflatline_fig, nonflatline_times = fd.annotate_events(15, 0.1, 0.95)
    fig = make_subplots(rows=2, cols=1)

    for i in range(len(flatline_fig['data'])):
        fig.add_trace(flatline_fig['data'][i], row=1, col=1)
    for i in range(len(nonflatline_fig['data'])):
        fig.add_trace(nonflatline_fig['data'][i], row=2, col=1)

    fig.show()

    if args.full:
        print('----------------Output flatline events--------------------')

        fd.output_apnea_files(flatline_times, nonflatline_times)

        print('----------------Train and test -------------------')

        model = LSTM(root_dir=args.root_dir, 
                     dataset=args.dataset,
                     apnea_type=args.apnea_type,
                     excerpt=args.excerpt,
                     batch_size=int(args.batch_size),
                     epochs=int(args.epochs))

        training_losses, training_errors, test_error = model.train(save_model=False,
                                                                   plot_loss=False)

        print(f'--------------Final test accuracy: {1-test_error}----------------')
        plt.plot(range(int(args.epochs)), training_losses, 'r--')
        plt.plot(range(int(args.epochs)), training_errors, 'b-')

        plt.legend(['Training Loss', 'Training error'])
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        # save model
        plt.show()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir",    default=".", help="root directory (parent folder of data/)")
    parser.add_argument("-d", "--dataset",    default="dreams", help="dataset (dreams, dublin, or mit)")
    parser.add_argument("-a", "--apnea_type", default="osa",    help="type of apnea (osa, osahs, or all)")
    parser.add_argument("-ex","--excerpt",    default=1,        help="excerpt number to use")
    parser.add_argument("-sr","--sample_rate",    default=8,        help="number of samples per second")
    parser.add_argument("-sc","--scale_factor",    default=1,        help="scale factor for normalization")
    parser.add_argument("-b","--batch_size",    default=16,        help="batch size")    
    parser.add_argument("-ep","--epochs",    default=15,        help="num epochs to train")
    parser.add_argument("-f", "--full", nargs='?', const=True, default=False)


    # parse args 
    args = parser.parse_args()

    main(args)
