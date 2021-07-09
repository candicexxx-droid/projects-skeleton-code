import argparse

import os
import torch
#lsfjalsdghlksghklghsaklghsadklghas fkldwgasglasgjslkfjsfklas 

import constants
from datasets.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train
from config import config
# os.chdir('/Users/candicecai/Desktop/Sophomore-Spring-ACM-AI-Project/projects-skeleton-code')
import importlib
# SUMMARIES_PATH = "training_summaries"
parser = argparse.ArgumentParser()
parser.add_argument('config_name', default='config', help='config file. Default is config.py file.')
args = parser.parse_args()
# get experiment config
config_module = importlib.import_module(args.config_name)
config = config_module.config

def main(config):
    # Get command line arguments
    # args = parse_arguments()
    #commit test

    # Create path for training summaries
    # summary_path = None
    # if args.logdir is not None:
    #     summary_path = f"{SUMMARIES_PATH}/{args.logdir}"
    #     os.makedirs(summary_path, exist_ok=True)

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print("Summary path:", summary_path)
    print("Epochs:", config['epochs'])
    print("Batch size:", config['batch_size'])

    # Initalize dataset and model. Then train the model!
    train_loader, valid_loader, test_loader= StartingDataset(batch_size=config['batch_size']) 
    model = StartingNetwork().to(device)
    starting_train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        network=model,
        num_epochs=config['epochs'],
        test = config['test']
    )


# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("epochs", type=int, default=constants.EPOCHS)
#     parser.add_argument("batch_size", type=int, default=constants.BATCH_SIZE)
#     parser.add_argument("test", type=bool, default=constants.TEST)
#     # parser.add_argument("--n_eval", type=int, default=constants.N_EVAL)
#     # parser.add_argument("--logdir", type=str, default=None)
#     return parser.parse_args()


if __name__ == "__main__":
    main(config)
