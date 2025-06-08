import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from mpc.dynamics import VerticalDroneDynamics
from utils.util import  generate_dataset, generate_dataset_multi

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
def dataset_loading(dynamics, dyn_name, stage=1, prev_models=None, device='cuda'):
    # path = f"dataset/{dyn_name}/stage{stage}/dataset.pkl"            # grid based dataset
    path = f"dataset/{dyn_name}/stage{stage}/dataset_random_s2000.pkl"     # random dataset
    os.makedirs(os.path.dirname(path), exist_ok=True)   
    return_trajectories = True

    if os.path.exists(path):
        with open(path, 'rb') as f:
            print(f"Loading dataset from: {path}")
            samples = pickle.load(f)
    else:
        print(f"Generating dataset from stage {stage} and saving to: {path}")
        '''
        IN paper H_R = 0.2 sec with dt = 0.02 sec so each rollout 10 steps
        safe set converges in 1.2 second
        we will use 0.3 as horizon length for each stage with H = 30 max steps with dt 0.01
        '''
        # samples, all_trajs, all_controls = generate_dataset(
        #             dynamics=dynamics,
        #             size=700,
        #             N=800,
        #             R=30,
        #             H=30,  
        #             u_std=0.1,
        #             stage= stage,
        #             device=device,
        #             prev_stage_models= prev_models,
        #             return_trajectories=return_trajectories,
        #             use_grid_sampling = False
        #         )
        samples, all_trajs, all_controls = generate_dataset_multi(
                    dynamics=dynamics,
                    size=2000, #700
                    N=800,
                    R=30,
                    H=30,  
                    u_std=0.1,
                    stage= stage,
                    device=device,
                    prev_stage_models= prev_models,
                    return_trajectories=return_trajectories,
                    use_grid_sampling = False
                )

        with open(path, 'wb') as f:
            pickle.dump(samples, f)

        if return_trajectories:
            r = 0  # index of rollout to select from each sample
            trajs_to_plot = [traj_tensor[r] for traj_tensor in all_trajs]  # list of [H_n+1, 3] tensors
            controls_to_plot = [control_tensor[r] for control_tensor in all_controls]  # list of [H_n] tensors
            dynamics.plot_trajectories_all(trajs_to_plot, controls_to_plot, stage)

    print(f"Loaded {len(samples)} samples.")
    dataset = MPCDataset(samples)

    return dataset
