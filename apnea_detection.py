import argparse
# Modules
from onset_extraction import OnsetExtraction
from train import Model

# Torch
import torch 

# Graphing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly
from plotly.subplots import make_subplots
from scipy.stats import zscore

# Logger
import wandb 

'''
Main program to run end-to-end apnea detection

Instructions to Run:
---------------------------------------------------
There are two options to run the script.

1) To run from command line/using argparser, change the default 
   arguments as desired. 

    python3 apnea_detection.py -h/--help to view arguments 

    Example: To run Dreams OSA Excerpt 3:
    python3 apnea_detection.py -d dreams -a osa -ex 3 

2) To run using default config:
    Instantiate DefaultConfig() class with desired parameters,
    then pass it into the main function.

        cfg = DefaultConfig()
        main(cfg)
'''

def main(cfg):
    print('***************************************')
    print(f'****        Dataset: {cfg.dataset}   ****')
    print(f'****        Apnea type: {cfg.apnea_type}    ****')
    print(f'****        Excerpt: {cfg.excerpt}         ****')
    print(f'****        Sample rate: {cfg.sample_rate}     ****')
    print('***************************************')
    # import pdb; pdb.set_trace();

    ''' --------------- Setting up experiment and logger-------------------------'''
    if cfg.logger: 
        # Add desired tags to experiment
        tags = [cfg.dataset, cfg.apnea_type, str(cfg.excerpt), "test_on_trained_patchAllisterW11"]
        if 'Box' in str(cfg.excerpt):
            tags.append('box')


        # Initialize project
        wandb.init(entity="neurostim", 
                  project="apnea_detection_v2", 
                  config=cfg,
                  tags=tags)

        # Initialize run
        wandb.run.name = f'{cfg.dataset}_{cfg.apnea_type}_{cfg.excerpt}'
        wandb.run.save()


    print('------------- Onset Detection----------------')
    if not cfg.skip_preprocess: 
        print('Preprocessing')

        oe = OnsetExtraction(cfg=cfg)
                                 
        if cfg.normalize:
            # Normalize 
            oe.normalize(slope_threshold=float(cfg.slope_threshold),
                         scale_factor_high=float(cfg.scale_factor_high),
                         scale_factor_low=float(cfg.scale_factor_low))

        # Extract onset, non-onset  events
        oe.extract_onset_events()

        # Plot extracted onset events
        oe.plot_extracted_events()

        # Save positive/negative sequences to files
        oe.write_output_files()

    else:
        print('Skipping Preprocessing step')

    print('---------------- Train and test -------------------')

    model = Model(cfg=cfg)

    # only inference
    if cfg.test_only:
        print("Testing only")
        model.test()
    else:
        print("Training and Testing")
        model.train(plot_loss=True)
        model.test()




    print('---------- Finished, Saving experiment ------------')

    # wandb.finish()


''' To use this config, instantiate an object of this class and 
    change any desired parameters and simply run the program. For example: 
    
    cfg = DefaultConfig()
    main(cfg)
        
'''



class DefaultConfig():
    def __init__(self, root_dir=".",
                       dataset="dreams",
                       apnea_type="osa",
                       excerpt=1,
                       sample_rate=8,

                       test_frac=0.2,

                       skip_preprocess=False,
                       normalize=False,
                       slope_threshold=3,
                       scale_factor_high=1,
                       scale_factor_low=0.1,
                       seconds_before_apnea=10,
                       seconds_after_apnea=5,
                       negative_dir=None,
                       positive_dir=None,

                       test_only=False,
                       model_type="cnn",
                       retrain=False,
                       learning_rate=0.001,
                       batch_size=64,
                       epochs=15,
                       
                       logger=False,
                       results_file="results.csv",
                       save_model=False,
                       base_model_path="base_model.ckpt"): 

        self.root_dir = root_dir
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.dataset = dataset
        self.apnea_type = apnea_type
        self.excerpt = excerpt
        self.sample_rate = sample_rate
        
        self.test_frac = test_frac

        self.skip_preprocess = skip_preprocess
        self.normalize = normalize
        self.slope_threshold = slope_threshold
        self.scale_factor_high = scale_factor_high
        self.scale_factor_low = scale_factor_low
        self.seconds_before_apnea = seconds_before_apnea
        self.seconds_after_apnea = seconds_after_apnea
        self.negative_dir = negative_dir
        self.positive_dir = positive_dir

        self.test_only = test_only
        self.model_type = model_type
        self.retrain = retrain
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.logger = logger
        self.results_file = results_file
        self.save_model = save_model
        self.base_model_path = base_model_path

# # Initialize cfg 
# cfg = DefaultConfig()
# main(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir",             default=".",        help="root directory (where source files are stored)")
    parser.add_argument("-d", "--dataset",              default="dreams",   help="dataset (dreams, dublin, mit, patch..)")
    parser.add_argument("-a", "--apnea_type",           default="osa",      help="type of apnea (osa)")
    parser.add_argument("-ex","--excerpt",              default=1,          help="excerpt number to use")
    parser.add_argument("-tf","--test_frac",            default=0.2,        help="ratio of dataset to hold out for testing")
   
    parser.add_argument("-sr","--sample_rate",          default=8,          help="number of samples per second")

    parser.add_argument("-sp", "--skip_preprocess",     default=False,      help="user can specify to skip normalization/onset extraction step", action='store_true')
    parser.add_argument("-n", "--normalize",            default=False,      help="normalize", action='store_true')
    parser.add_argument("-st","--slope_threshold",      default=3,          help="slope threshold for nonlinear normalization")
    parser.add_argument("-sfh","--scale_factor_high",   default=1,          help="high scale factor for nonlinear normalization")
    parser.add_argument("-sfl","--scale_factor_low",    default=0.1,        help="lowscale factor for nonlinear normalization")
    parser.add_argument("-sba","--seconds_before_apnea",default=10,         help="number of seconds before flatline starts")
    parser.add_argument("-saa","--seconds_after_apnea", default=5,          help="number of seconds after flatline starts")
    parser.add_argument("-nd", "--negative_dir",        default=None,       help="negative directory to write files to")
    parser.add_argument("-pd", "--positive_dir",        default=None,       help="positive directory to write files to")

    parser.add_argument("-t","--test_only",               default=False,         help="skips training and only performs inference", action='store_true')
    parser.add_argument("-m", "--model_type",           default="cnn",      help="model type")
    parser.add_argument("-re", "--retrain",             default=False,      help="retrain", action='store_true')
    parser.add_argument("-lr", "--learning_rate",       default=0.001,      help="learning rate")
    parser.add_argument("-b","--batch_size",            default=64,         help="batch size")    
    parser.add_argument("-ep","--epochs",               default=15,         help="number of epochs to train")

    parser.add_argument("-l", "--logger",               default=False,      help="log run", action='store_true')
    parser.add_argument("-rf", "--results_file",        default="results.csv",   help="results file (csv)")
    parser.add_argument("-s", "--save_model",           default=False,      help="save model", action='store_true')
    parser.add_argument("-bm", "--base_model_path",     default="base_model.ckpt",help="base model path")

    # parse args 
    cfg = parser.parse_args()

    main(cfg)


# def main(cfg):
#     for i in [3,12, 25,27,28]:
#         dc = DefaultConfig()
#         dc.model_type = "cnn"
#         dc.dataset = "dublin"
#         dc.excerpt = str(i) + "Box"
#         dc.epochs = 20
#         dc.batch_size = 16
#         dc.skip_preprocess = True
#         dc.logger=True
#         apnea_detection(dc)
        # wandb.finish()