from utils.datsetio import dataset_loading
# from utils.model import VFANet

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


from tqdm import tqdm
import random

'''
This is the main script that orchestrates the training process.
1. It generates the dataset using the MPC rollout.
2. It trains the neural network on the generated dataset.
3. It uses the trained model for the next stage.
'''

NUM_STAGES = 1
prev_model = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for stage in range(1, NUM_STAGES + 1):
    print(f"\n--- Stage {stage} ---")

    # Load or generate dataset
    dataset = dataset_loading(stage, prev_model, device=device)
    # split the dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # print some samples of train dataset
    # for i in range(10):
    #     sample = train_dataset[i]
    #     print(f"Sample {i}: t={sample['t']}, x={sample['x']}, V_hat={sample['V_hat']}")

    '''  
    Initialize the model and trian it on the datset that learns to out put value function
    with input as the state and time and K
    '''
    # model = VFANet()
    # model.to(device)
    # model.train()
    '''
    Training loop:

    '''
    # print("Training the model...")
    # # save the model and pass it to the next stage
    # accelerator.save_state(model, f"model_stage{stage}.pt")
    # prev_model = model
    
