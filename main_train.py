from utils.datsetio import dataset_loading
from utils.model import SingleBVPNet
from mpc.mpc_rollout import VerticalDroneDynamics, compute_recursive_value
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


'''
This is the main script that orchestrates the training process.
1. It generates the dataset using the MPC rollout.
2. It trains the neural network on the generated dataset.
3. It uses the trained model for the next stage.
'''

NUM_STAGES = 4
prev_models = []
train_from_checkpoint = True
train_from_begining = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dynamics = VerticalDroneDynamics(device=device)


for stage in range(1, NUM_STAGES + 1):
    print(f"\n--- Stage {stage} ---")

    # Load or generate dataset
    # we will feed the previous stage model in data generation process and also for terminal bounary constraint in training
    dataset = dataset_loading(dynamics, stage, prev_models, device=device)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # print some samples of train dataset
    # for i in range(10):
    #     sample = train_dataset[i]
    #     print(f"Sample {i}: t={sample['t']}, x={sample['x']}, V_hat={sample['V_hat']}")

    '''  
    Initialize the model and trian it on the datset that learns to output value function
    with input as the state and time and K
    '''
    model = SingleBVPNet(out_features=1,  # V(state, t)
                        type='sine',
                        in_features=4,  # z, vz, k, t
                        mode='mlp',
                        hidden_features=256, 
                        num_hidden_layers=3).to(device)
    
    '''
    Training the model
    '''
    num_epochs = 2000
    save_dir = f"model_checkpoints"

    # Path to saved checkpoint
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"stage_{stage}_best.pt")

    # Load checkpoint if exists
    if os.path.exists(model_path) and not train_from_checkpoint:
        print(f"Loading checkpoint from {model_path}")
        model.load_state_dict(torch.load(model_path))
    elif train_from_checkpoint or train_from_begining:
        if os.path.exists(model_path):
            print("Training the loaded checkpoint")
            model.load_state_dict(torch.load(model_path))
        else:
            print("Trainign from scratch")

        ##### Initialize wandb
        wandb.init(
            project="value-function-reachability",
            name=f"Stage-{stage}",
            config={
                "learning_rate": 1e-4,
                "epochs": num_epochs,
                "batch_size": train_dataloader.batch_size,
                "stage": stage,
            }
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for batch in train_dataloader:
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
                    V_pred = l_x + (T_stage.reshape(-1) - t.reshape(-1)) * O_pred
                else:
                    V_prev = compute_recursive_value(x,
                                                    T_stage,
                                                    dynamics, 
                                                    prev_models)
                    V_pred = V_prev + (T_stage.reshape(-1) - t.reshape(-1)) * O_pred

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
                        V_pred = l_x + (T_stage.reshape(-1) - t.reshape(-1)) * O_pred
                    else:
                        V_prev = compute_recursive_value(x,
                                                        T_stage, # in batch format
                                                        dynamics, 
                                                        prev_models)
                        V_pred = V_prev + (T_stage.reshape(-1) - t.reshape(-1)) * O_pred

                    loss = criterion(V_pred, V_mpc)
                    val_loss += loss.item() * x.size(0)
            
            val_loss /= len(val_dataloader.dataset)

            ##### wandb logging
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch
            })

            # print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            ###### Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f"stage_{stage}_best.pt")
                torch.save(model.state_dict(), model_path)
                print('Saved')
        
        wandb.finish()
        print(f"Model for stage {stage} trained")
    
    prev_models.append(model.eval())
    


    
    
