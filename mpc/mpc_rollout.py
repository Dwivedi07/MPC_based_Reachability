import torch
import matplotlib.pyplot as plt
import numpy as np
import os

class VerticalDroneDynamics:
    def __init__(self, device):
        self.g = 9.8       # g
        self.dt = 0.01 # each stage horizon 0.3 seconds long with 100 steps
        self.horizon= 0.3
        self.device = device
        self.input_multiplier = 12.0   # K
        self.input_magnitude_max = 1.0     # u_max
        self.K_range = torch.tensor([0.0, 12.0]).to(device)  # K
        self.T_terminals = {
                            1: torch.tensor([1.2]),
                            2: torch.tensor([0.9]),
                            3: torch.tensor([0.6]),
                            4: torch.tensor([0.3]),
                        }

        self.state_range_ = torch.tensor([[-4, 4],[-0.5, 3.5]]).to(device) # not inside failure set
        self.control_range_ =torch.tensor([[-self.input_magnitude_max, self.input_magnitude_max]]).to(device)
        self.nominal_control_init = torch.ones(1, device = device)*self.g/self.input_multiplier


    def sample_state(self, batch_size=1):
        z = torch.empty(batch_size, device=self.device).uniform_(*self.state_range_[1])
        vz = torch.empty(batch_size, device=self.device).uniform_(*self.state_range_[0])
        K = torch.empty(batch_size, device=self.device).uniform_(*self.K_range)
        return torch.stack([z, vz, K], dim=-1)  # shape: [B, 3]

    def step(self, state, control):
        """
        Args:
            state: Tensor of shape [B, 3] -> [z, vz, K]
            control: Tensor of shape [B] or [B, 1]
        Returns:
            next_state: Tensor of shape [B, 3]
        """
        z, vz, K = state[:, 0], state[:, 1], state[:, 2]
        u = control.squeeze(-1)
        a = K * u - self.g
        z_next = z + vz * self.dt
        vz_next = vz + a * self.dt
        return torch.stack([z_next, vz_next, K], dim=1)

    def compute_l(self, traj):
        """
        Args:
            traj: Tensor of shape [N, H+1, 3] pr [N, 3]
        Returns:
            cost: Tensor of shape [N, H+1] - reward at each step
        """
        z = traj[..., 0]
        return -torch.abs(z - 1.5) + 1.5
    
    def sample_time(self, stage, device):
        '''
        Check the stage and pick the sample accordingly from the horizon
        1: [T-H, T]
        2: [T-2H, T-H]
        3: [T-3H, T-2H]
        4: [T-4H, T-3H]
        '''
        if stage == 1:
            t_i = torch.round((0.9 + torch.rand(1, device=device) * self.horizon) * 100) / 100   # t_i in [0.9, 1.2] with 2 decimal places
            t_f = torch.tensor([self.T_terminals[stage]])
        elif stage == 2:
            t_i = torch.round((0.6 + torch.rand(1, device=device) * self.horizon) * 100) / 100   # t_i in [0.6, 0.9] with 2 decimal places
            t_f = torch.tensor([self.T_terminals[stage]])
        elif stage == 3:
            t_i = torch.round((0.3 + torch.rand(1, device=device) * self.horizon) * 100) / 100   # t_i in [0.3, 0.6] with 2 decimal places
            t_f = torch.tensor([self.T_terminals[stage]])
        else:
            t_i = torch.round((0.0 + torch.rand(1, device=device) * self.horizon) * 100) / 100   # t_i in [0.0, 0.3] with 2 decimal places
            t_f = torch.tensor([self.T_terminals[stage]])
        
        return (t_i, t_f.to(device))


    def plot_trajectories_all(self, trajs, controls, stage, dt=None):
        """
        Plot multiple trajectories and corresponding controls in shared figures.
        
        Args:
            trajs: List of tensors shape [H', 3] — [z, vz, K]
            controls: List of tensor of shape [H']
            dt: timestep (optional, overrides self.dt)
        """
        if dt is None:
            dt = self.dt
       
        # --- Plot trajectories ---
        plt.figure(figsize=(8, 4))
        for i, traj in enumerate(trajs):
            H_plus_1 = traj.shape[0]
            time_traj = torch.arange(H_plus_1) * dt
            z_vals = traj[:, 0].cpu().numpy()
            plt.plot(time_traj, z_vals, label=f'Traj {i+1}', alpha=0.6)
        
        plt.axhspan(-1, 0, color='red', alpha=0.1, label="Failure Region")
        plt.axhspan(3, 4, color='red', alpha=0.1)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(3.0, color='gray', linestyle='--', linewidth=0.5)
        plt.xlabel('Time [s]')
        plt.ylabel('z (height)')
        plt.title('Drone Altitude Trajectories')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'outputs/drone_altitude_{stage}.png')
        plt.show()

        # --- Plot controls ---
        plt.figure(figsize=(8, 4))
        for i, control in enumerate(controls):
            H = control.shape[0]
            time_ctrl = torch.arange(H) * dt
            control_vals = control.cpu().numpy()
            plt.plot(time_ctrl, control_vals, label=f'Control {i+1}', alpha=0.6)
        
        plt.xlabel('Time [s]')
        plt.ylabel('Control u')
        plt.title('Control Sequences')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'outputs/drone_controls_{stage}.png')
        plt.show()
