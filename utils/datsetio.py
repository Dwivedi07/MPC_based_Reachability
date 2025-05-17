import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from mpc.mpc_rollout import VerticalDroneDynamics, generate_dataset

class MPCDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t, x, V_hat = self.samples[idx]
        return {
            't': torch.tensor(t, dtype=torch.float32),
            'x': torch.tensor(x, dtype=torch.float32),
            'V_hat': torch.tensor(V_hat, dtype=torch.float32)
        }

# ----------------------------
def dataset_loading(stage=1, prev_model=None, device='cuda'):
    path = f"dataset/stage{stage}/dataset.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with open(path, 'rb') as f:
            print(f"Loading dataset from: {path}")
            samples = pickle.load(f)
    else:
        print(f"Generating dataset and saving to: {path}")
        dynamics = VerticalDroneDynamics(device=device)

        return_trajectories = False
        samples, all_trajs, all_controls = generate_dataset(
                    dynamics=dynamics,
                    size=300,
                    N=10,
                    R=1,
                    H=200,  # in 1 sec with dt = 0.01 , H = 100, total 4 H of 25 each
                    u_std=0.1,
                    device=device,
                    return_trajectories=return_trajectories
                )
        print(f"Generated {len(samples)} samples.")

        if return_trajectories:
            dynamics.plot_trajectories(all_trajs[0], all_controls[0])
        
        with open(path, 'wb') as f:
            pickle.dump(samples, f)

    dataset = MPCDataset(samples)
    
    return dataset
