import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os

from utils import quaternion

class VerticalDroneDynamics:
    def __init__(self, device):
        self.g = 9.8       # g
        self.dt = 0.01 # each stage horizon 0.3 seconds long with 30 steps
        self.horizon= 0.3
        self.device = device
        self.state_dim = 3  # [z, vz, K]
        self.input_dim = 4  # include time
        self.control_dim = 1  # u

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

    def sample_state(self, n, size, batch_size=1, use_grid_sampling=False):
        '''
        Args:
            n: nth sample to return
            size: total number of samples to generate
            batch_size: batch 
            use_grid_sampling: if doing grid based sampling or uniform random
        Returns:
            Tensor of shape (batch_size, 3)
        '''
        if use_grid_sampling:
            grid_dim = int(size ** (1 / 3)) + 1

            # Unpack min/max ranges
            vz_min, vz_max = self.state_range_[0]
            z_min, z_max   = self.state_range_[1]
            K_min, K_max   = self.K_range

            # Grid sampling for each dimension
            z_vals  = torch.linspace(z_min, z_max, steps=grid_dim)
            vz_vals = torch.linspace(vz_min, vz_max, steps=grid_dim)
            K_vals  = torch.linspace(K_min, K_max, steps=grid_dim)

            # Create grid and reshape
            zz, vv, kk = torch.meshgrid(z_vals, vz_vals, K_vals, indexing='ij')
            state_grid = torch.stack([zz.reshape(-1), vv.reshape(-1), kk.reshape(-1)], dim=1)
            state_grid = state_grid[:size].to(self.device)
            return state_grid[n:n+1]  # shape: [1, 3]

        else:
            vz_min, vz_max = self.state_range_[0]
            z_min, z_max   = self.state_range_[1]
            K_min, K_max   = self.K_range

            # Random uniform sampling
            z  = torch.empty(batch_size, device=self.device).uniform_(z_min, z_max)
            vz = torch.empty(batch_size, device=self.device).uniform_(vz_min, vz_max)
            K  = torch.empty(batch_size, device=self.device).uniform_(K_min, K_max)

            return torch.stack([z, vz, K], dim=-1)  # shape: [B, 3]

    # def sample_state(self, n, size, batch_size=1, use_grid_sampling= False):
    #     '''
    #     Args:
    #         n: nth smaple to return
    #         size: the number of total samples to generate
    #         batch_size: batch 
    #         use_grod_sampling: if doing grid based search or normal
    #     Returns:
    #         Tensor of shape (batch_size, 13)
    #     '''
    #     if use_grid_sampling:
    #         grid_dim = int(size ** (1 / 3)) + 1
    #         z_vals = torch.linspace(self.state_range_[1], steps=grid_dim)
    #         vz_vals = torch.linspace(self.state_range_[0], steps=grid_dim)
    #         K_vals = torch.linspace(self.K_range, steps=grid_dim)

    #         zz, vv, kk = torch.meshgrid(z_vals, vz_vals, K_vals, indexing='ij')
    #         state_grid = torch.stack([zz.reshape(-1), vv.reshape(-1), kk.reshape(-1)], dim=1)
    #         state_grid = state_grid[:size].to(self.device)
    #         return state_grid[n:n+1]  # [B=1, 3]
    #     else:
    #         z = torch.empty(batch_size, device=self.device).uniform_(*self.state_range_[1])
    #         vz = torch.empty(batch_size, device=self.device).uniform_(*self.state_range_[0])
    #         K = torch.empty(batch_size, device=self.device).uniform_(*self.K_range)
    #         return torch.stack([z, vz, K], dim=-1)  # shape: [B=1, 3]

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
        # t_i = torch.round((self.T_terminals[stage].item() - self.horizon + torch.rand(1, device=device) * self.horizon) * 100) / 100   
        # t_f = torch.tensor([self.T_terminals[stage]])

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
    
    def visualize_dataset(self, dataset, stage, K_min = 11.0, K_max = 12.0):
        """
        Visualize a heatmap of the value function V_hat over (z, vz),
        for points where K in [K_min, K_max].

        Args:
            dataset: MPCDataset
        """
        z_vals, vz_vals, V_vals = [], [], []

        for i in range(len(dataset)):
            sample = dataset[i]
            x = sample['x']
            V_hat = sample['V_hat'].item()
            
            z, vz, K = x.tolist()
            if K_min <= K <= K_max:
                z_vals.append(z)
                vz_vals.append(vz)
                V_vals.append(V_hat)

        if not z_vals:
            print(f"No datapoints found with K in [{K_min}, {K_max}]")
            return

        # Convert to numpy arrays for heatmap plotting
        z_vals = np.array(z_vals)
        vz_vals = np.array(vz_vals)
        V_vals = np.array(V_vals)

        # Create heatmap grid using scatter plot + interpolation
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(vz_vals, z_vals, c=V_vals, cmap='viridis', s=10)
        plt.colorbar(scatter, label=r'$\hat{V}$(z, v_z)')
        
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(3.0, color='gray', linestyle='--', linewidth=0.5)
        plt.axhspan(-1, 0, color='red', alpha=0.1, label="Failure Region")
        plt.axhspan(3, 4, color='red', alpha=0.1)

        plt.ylabel('z (height)')
        plt.xlabel(r'$v_z$ (vertical velocity)')
        plt.title(r'Value Function Heatmap over (z, $v_z$)$')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'outputs/MPCdataset{stage}.png')


class Quadrotor13D:
    def __init__(self, device, collisionR = 0.5, collective_thrust_max = 20.0):
        '''
        Quadrotor dynamics in 13D state space
        '''
        self.dt = 0.01  
        self.horizon= 0.25
        self.state_dim = 13
        self.input_dim = 14
        self.control_dim = 4
        self.collective_thrust_max = collective_thrust_max
        
        self.m = 1  # mass
        self.arm_l = 0.17  # quadrator as disk of radius 0.17m
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8
        self.g = 9.8      

        self.dwx_max = 8
        self.dwy_max = 8
        self.dwz_max = 4
        self.dist_dwx_max = 0
        self.dist_dwy_max = 0
        self.dist_dwz_max = 0
        self.dist_f = 0

        self.collisionR = collisionR  # collision raduis of cylinder obstacle
        self.device = device

        # State and control limits
        self.state_upper = torch.tensor([3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5], dtype=torch.float32).to(device)
        self.state_lower = -self.state_upper
        self.control_upper = torch.tensor([20, 8, 8, 4], dtype=torch.float32).to(device)
        self.control_lower = -self.control_upper

        # For safety function
        self.keepout_radius = 0.5    # cylinder radius
        self.drone_radius = 0.17     # quadrotor disk radius

        self.T_terminals = {
                            1: torch.tensor([1.0]),
                            2: torch.tensor([0.75]),
                            3: torch.tensor([0.50]),
                            4: torch.tensor([0.25]),
                        }

        self.state_range_ =  torch.tensor([
                            [-3.0, 3.0],
                            [-3.0, 3.0],
                            [-3.0, 3.0],
                            [-1.0, 1.0],
                            [-1.0, 1.0],
                            [-1.0, 1.0],
                            [-1.0, 1.0],
                            [-5.0, 5.0],
                            [-5.0, 5.0],
                            [-5.0, 5.0],
                            [-5.0, 5.0],
                            [-5.0, 5.0],
                            [-5.0, 5.0],
                            ]).to(device)
        self.control_range_ = torch.tensor([[-self.collective_thrust_max, self.collective_thrust_max],
                                            [-self.dwx_max, self.dwx_max],
                                            [-self.dwy_max, self.dwy_max],
                                            [-self.dwz_max, self.dwz_max]]).to(device)
        self.nominal_control_init = torch.tensor([-self.Gz*0.0,0,0,0]).to(device)

    

    def state_limits(self):
        return self.state_upper, self.state_lower

    def control_limits(self):
        return self.control_upper, self.control_lower

    def safe_mask(self, state):
        xy_norm = torch.sqrt(state[:, 0]**2 + state[:, 1]**2)
        return xy_norm > self.keepout_radius

    def failure_mask(self, state):
        xy_norm = torch.sqrt(state[:, 0]**2 + state[:, 1]**2)
        return xy_norm < self.keepout_radius

    def normalize_q(self, state):
        # normalize quaternion
        normalized_x = state*1.0
        q_tensor = state[..., 3:7]
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2,dim=-1)  # normalize quaternion
        normalized_x[..., 3:7] = q_tensor
        return normalized_x
    
    def clamp_state_input(self, state_input):
        return self.normalize_q(state_input)

    def control_range(self, state):
        return [[-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max]]

    def f_x(self, state):
        PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
        PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [state[:, i] for i in range(13)]

        f = torch.zeros_like(state)
        f[:, PXi] = VX
        f[:, PYi] = VY
        f[:, PZi] = VZ
        f[:, QWi] = -0.5 * (WX * QX + WY * QY + WZ * QZ)
        f[:, QXi] =  0.5 * (WX * QW + WZ * QY - WY * QZ)
        f[:, QYi] =  0.5 * (WY * QW - WZ * QX + WX * QZ)
        f[:, QZi] =  0.5 * (WZ * QW + WY * QX - WX * QY)
        f[:, VZi] = -self.g
        f[:, WXi] = -5 * WY * WZ / 9.0
        f[:, WYi] =  5 * WX * WZ / 9.0
        return f

    def g_x(self, state):
        PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
        PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [state[:, i] for i in range(13)]

        g = torch.zeros((state.shape[0], 13, 4), device=state.device)
        g[:, VXi, 0] = 2 * (QW * QY + QX * QZ)
        g[:, VYi, 0] = 2 * (QY * QZ - QW * QX)
        g[:, VZi, 0] = 1 - 2 * QX**2 - 2 * QY**2
        g[:, WXi, 1] = 1.0
        g[:, WYi, 2] = 1.0
        g[:, WZi, 3] = 1.0
        return g

    def step(self, state, u):
        """
        One step forward in time using control-affine dynamics.

        Args:
            state: [batch_size, 13] - current state
            u: [batch_size, 4]  - control input

        Returns:
            x_next: [batch_size, 13] - next state
        """
        fx = self.f_x(state)                       # [B, 13]
        gx = self.g_x(state)                       # [B, 13, 4]
        control_effect = torch.einsum('bij,bj->bi', gx, u)  # g(state) * u

        state_next = state + self.dt * (fx + control_effect)
        return state_next

    def sample_state(self, n, size, batch_size=1, use_grid_sampling= False):
        '''
        Sample a batch of valid 13D quadrotor states, uniformly within the defined state ranges.
        Ensures the quaternion part is normalized.
        Args:
            n: nth smaple to return
            size: the number of total samples to generate
            batch_size: batch 
            use_grod_sampling: if doing grid based search or normal
        Returns:
            Tensor of shape (batch_size, 13)
        '''
        sampled_state = torch.empty((batch_size, self.state_dim), device=self.device)
        for i in range(self.state_dim):
            low = self.state_range_[i, 0]
            high = self.state_range_[i, 1]
            sampled_state[:, i] = torch.empty(batch_size, device=self.device).uniform_(low, high)

        # after generating the random sample clamp the quaternion by calling self.clamp_state_input
        # Normalize quaternion part [q_x, q_y, q_z, q_w] which are at indices 3:7
        sampled_state = self.clamp_state_input(sampled_state)

        return sampled_state

    def sample_time(self, stage, device):
        '''
        Check the stage and pick the sample accordingly from the horizon
        1: [T-H, T]
        2: [T-2H, T-H]
        3: [T-3H, T-2H]
        4: [T-4H, T-3H]
        '''
        t_i = torch.round((self.T_terminals[stage].item() - self.horizon + torch.rand(1, device=device) * self.horizon) * 100) / 100   # t_i in [0.9, 1.2] with 2 decimal places
        t_f = torch.tensor([self.T_terminals[stage]])
        
        return (t_i, t_f.to(device))

    def compute_l(self, state, a = 0.0, b= 0.0): 
        '''
        The safety function l(x) is
        defined as the signed distance from the disk to an infinitely
        long cylinder with a radius of 0.5 m centered at the origin
        
        a = 0.0
        b = 0.0  for avoid problem
        Args:
            traj: Tensor of shape [N, H+1, 3] pr [N, 3]
        Returns:
            cost: Tensor of shape [N, H+1] - reward at each step
        '''
        state_=state*1.0
        state_[...,0]=state_[...,0]- a
        state_[...,1]=state_[...,1]- b

        # create normal vector
        v = torch.zeros_like(state_[..., 4:7])
        v[..., 2] = 1
        v = quaternion.quaternion_apply(state_[..., 3:7], v)
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        # compute vector from center of quadrotor to the center of cylinder
        px = state_[..., 0]
        py = state_[..., 1]
        # get full body distance
        dist = torch.norm(state_[..., :2], dim=-1)
        # return dist- self.collisionR
        dist = dist- torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
                           + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))
        # take torch.min(_this_returned_val, dim=-1).values for cost of traj

        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR
     
    def plot_trajectories_all(self, trajs, controls, stage, dt=None):
        """
        Plot XY trajectories and control inputs for a batch of 13D quadrotor trajectories.
        
        Args:
            trajs: List of tensors of shape [H'+1, 13] — full state 
            controls: List of tensors of shape [H', 4] 
            stage: current training stage or name identifier
            dt: optional timestep override
        """
        if dt is None:
            dt = self.dt

        # --- Plot XY Trajectories ---
        fig, ax = plt.subplots(figsize=(6, 6))
        for i, traj in enumerate(trajs):
            x_vals = traj[:, 0].cpu().numpy()
            y_vals = traj[:, 1].cpu().numpy()
            ax.plot(x_vals, y_vals, alpha=0.7)

            # # Draw drone as circles at a few positions (start, mid, end)
            # for t in [0, len(traj)//2, -1]:
            #     drone_circle = Circle(
            #         (x_vals[t], y_vals[t]), 
            #         radius=self.drone_radius, 
            #         color='blue', 
            #         alpha=0.2)
            #     ax.add_patch(drone_circle)

        # Obstacle (cylinder) at origin (0, 0)
        obstacle = Circle((0, 0), radius=self.keepout_radius, color='red', alpha=0.3, label='Obstacle')
        ax.add_patch(obstacle)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Quadrotor XY Trajectories')
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"outputs/quadrotor_xy_traj_{stage}.png")
        plt.show()

        # --- Plot each control dimension ---
        control_labels = ['Collective Thrust', 'dωx', 'dωy', 'dωz']
        fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        
        for i in range(4):  # control_dim = 4
            for j, control in enumerate(controls):
                H = control.shape[0]
                time = torch.arange(H) * dt
                axs[i].plot(time.cpu().numpy(), control[:, i].cpu().numpy(), label=f'Control {j+1}', alpha=0.7)
            axs[i].set_ylabel(control_labels[i])
            axs[i].grid(True)

        axs[-1].set_xlabel('Time [s]')
        fig.suptitle('Control Inputs over Time', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"outputs/quadrotor_controls_{stage}.png")
        plt.show()
