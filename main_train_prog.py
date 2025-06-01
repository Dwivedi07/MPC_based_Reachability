from utils.datsetio import dataset_loading
from utils.model import SingleBVPNet
from utils.util import compute_recursive_value
from mpc.mpc_rollout import VerticalDroneDynamics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os

from tqdm import tqdm
import random
import wandb
import subprocess
import time


'''
This is the main script that orchestrates the training process.
1. It generates the dataset using the MPC rollout.
2. It trains the neural network on the generated dataset.
3. It uses the trained model for the next stage.
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
save_dir = f"model_checkpoints_prog"
os.makedirs(save_dir, exist_ok=True)

MPCdata_visual = False
train_from_checkpoint = False
train_from_begining = True
use_wandb = True

NUM_STAGES = 4
num_epochs = 50
prev_models = []
N_p = 5                 # Number of progressive steps per stage
H = 0.3                 # Total horizon (seconds) to regress over

dynamics = VerticalDroneDynamics(device=device)

for stage in range(1, NUM_STAGES + 1):
    print(f"\n--- Stage {stage} ---")

    # Load or generate dataset
    # we will feed the previous stage model in data generation process and also for terminal bounary constraint in training
    dataset = dataset_loading(dynamics, stage, prev_models, device=device)
    T_s = dynamics.T_terminals[stage].item()

    for prog_i in range(1, N_p+1):
        t_min = T_s - (prog_i * H / N_p)
        t_max = T_s
        print(f"\n Progressive Sub-stage {prog_i}/{N_p}: t in [{t_min:.2f}, {t_max:.2f}]")

        # Filter dataset to the apt time window
        filtered_dataset = [sample for sample in dataset if t_min <= sample['t'].item() <= t_max]
        if len(filtered_dataset) < 128:
            print(" Very few samples. Consider adjusting N_p or H.")
            continue

        train_size = int(0.9 * len(filtered_dataset))
        val_size = len(filtered_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(filtered_dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        '''  
        Initialize the model and trian it on the datset that learns to output value function
        with input as the state and time and K
        '''
        model = SingleBVPNet(out_features=1,  # V(state, t)
                            in_features=4,  # z, vz, k, t
                            hidden_features=512, # 256 first set of training I did
                            num_hidden_layers=3).to(device)
        
        
        '''
        Training the model
        '''
        model_path = os.path.join(save_dir, f"stage_{stage}_progressive_{prog_i}_best.pt")
                
        # Load checkpoint if exists
        if os.path.exists(model_path) and not train_from_checkpoint:
            print(f"Loading checkpoint from {model_path}")
            model.load_state_dict(torch.load(model_path))
        elif train_from_checkpoint or train_from_begining:
            if os.path.exists(model_path):
                print("Training the loaded checkpoint")
                model.load_state_dict(torch.load(model_path))
            else:
                print("Training from scratch")
            
            # Load from previous sub-stage for warm-starting
            if prog_i > 1:
                prev_model_path = os.path.join(save_dir, f"stage_{stage}_progressive_{prog_i-1}_best.pt")
                print(f"Warm-starting from {prev_model_path}")
                model.load_state_dict(torch.load(prev_model_path))

            ##### Initialize wandb
            if use_wandb:
                wandb.init(
                    project="value-function-reachability-progreesive",
                    name=f"Stage-{stage}-prog-stage-{prog_i}",
                    config={
                        "learning_rate": 2e-5,
                        "epochs": num_epochs,
                        "batch_size": train_dataloader.batch_size,
                        "stage": stage,
                        "progressive_step": prog_i,
                    }
                )
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=2e-5)
            best_val_loss = float("inf")
            
            for epoch in tqdm(range(num_epochs),  desc=f"[Stage {stage} | Prog {prog_i}]", ncols=80, unit="epoch"):
                model.train()
                train_loss = 0.0

                for batch in train_dataloader:
                    # wait_for_gpu_cooldown(threshold_temp=78, wait_seconds=80, gpu_id=0)  # Add this line
                    x = batch['x'].to(device)       # [batch_size, 3]
                    t = batch['t'].to(device).unsqueeze(1)  # [batch_size, 1]
                    V_mpc = batch['V_hat'].to(device).unsqueeze(1)  # [batch_size, 1]

                    input_tensor = torch.cat([x, t], dim=1)  # [batch_size, 4]

                    O_pred = model(input_tensor)  # [batch_size, 1]  # O_{\theta}
                    T_stage = dynamics.T_terminals[stage].to(device) # temrinal time for first stage T_teminal_1 
                    T_stage = T_stage.expand(x.shape[0],1)
                    # Imposing the terminal Constriant
                    if stage == 1:
                        l_x = dynamics.compute_l(x)
                        delta_T = (T_stage - t)  
                        V_pred = l_x.unsqueeze(1) + delta_T * O_pred
                        
                    else:
                        V_prev = compute_recursive_value(x,
                                                        T_stage,
                                                        dynamics, 
                                                        prev_models)
                        delta_T = (T_stage - t)   # [N, 1]
                        V_pred = V_prev + delta_T * O_pred 
                    
                    assert V_pred.shape[1] == 1, f"V_pred has shape {V_pred.shape}, expected (*, 1)"
                    loss = criterion(V_pred, V_mpc)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * x.size(0)

                train_loss /= len(train_dataloader.dataset)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_dataloader:
                        
                        x = batch['x'].to(device)
                        t = batch['t'].to(device).unsqueeze(1)
                        V_mpc = batch['V_hat'].to(device).unsqueeze(1)

                        input_tensor = torch.cat([x, t], dim=1)
                        O_pred = model(input_tensor)
                        T_stage = dynamics.T_terminals[stage].to(device) # temrinal time for first stage T_teminal_1 
                        T_stage = T_stage.expand(x.shape[0],1)
                        # Imposing the terminal Constriant
                        if stage == 1:
                            l_x = dynamics.compute_l(x)
                            delta_T = T_stage - t
                            V_pred = l_x.unsqueeze(1) + delta_T * O_pred
                        else:
                            V_prev = compute_recursive_value(x,
                                                            T_stage, # in batch format
                                                            dynamics, 
                                                            prev_models)
                            delta_T = (T_stage - t)   # [N, 1]
                            V_pred = V_prev + delta_T * O_pred 

                        assert V_pred.shape[1] == 1, f"V_pred has shape {V_pred.shape}, expected (*, 1)"
                        loss = criterion(V_pred, V_mpc)
                        val_loss += loss.item() * x.size(0)
                
                val_loss /= len(val_dataloader.dataset)

                ##### wandb logging
                if use_wandb:
                    wandb.log({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "epoch": epoch
                    })
                # if epoch%5 == 0:
                #     print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                ###### Save model
                torch.save(model.state_dict(), model_path)
                    
                    
            if use_wandb:
                wandb.finish()

            print(f"Model for stage {stage}, progressive step {prog_i} trained and saved.")
    
    final_model_path = os.path.join(save_dir, f"stage_{stage}_progressive_{N_p}_best.pt")
    model.load_state_dict(torch.load(final_model_path))
    prev_models.append(model.eval())