import torch
import matplotlib.pyplot as plt
import numpy as np

class VerticalDroneDynamics:
    def __init__(self, device):
        self.g = 9.8       # g
        self.dt = 0.01
        self.device = device
        self.input_multiplier = 12.0   # K
        self.input_magnitude_max = 1.0     # u_max
        self.K_range = torch.tensor([0.0, 12.0]).to(device)  # K

        self.state_range_ = torch.tensor([[-4, 4],[-0.5, 3.5]]).to(device)
        self.control_range_ =torch.tensor([[-self.input_magnitude_max, self.input_magnitude_max]]).to(device)
        self.nominal_control_init = torch.ones(1)*self.g/self.input_multiplier

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
            traj: Tensor of shape [N, H+1, 3]
        Returns:
            cost: Tensor of shape [N, H+1] - reward at each step
        """
        z = traj[..., 0]
        return -torch.abs(z - 1.5) + 1.5
    
    def plot_trajectories(self, traj_batch, u_batch, max_plot=5):
        """
        traj_batch: [B, H+1, 3]
        u_batch: [B, H]
        Plots z, vz, K and u with failure set highlighted.
        """
        B, H_plus_1, _ = traj_batch.shape
        H = H_plus_1 - 1
        t_traj = torch.linspace(0, H * self.dt, H + 1)
        t_ctrl = torch.linspace(0, (H - 1) * self.dt, H)

        B_plot = min(max_plot, B)
        fig, axs = plt.subplots(B_plot, 2, figsize=(12, 3.5 * B_plot))

        if B_plot == 1:
            axs = axs[None, :]  # Ensure 2D indexing for single row

        for i in range(B_plot):
            z = traj_batch[i, :, 0].cpu()
            vz = traj_batch[i, :, 1].cpu()
            K = traj_batch[i, :, 2].cpu()
            u = u_batch[i].cpu()

            failure_mask = (z < 0) | (z > 3)

            # Plot state trajectory
            axs[i, 0].plot(t_traj, z, label='z (height)', color='blue')
            axs[i, 0].plot(t_traj, vz, label='vz (velocity)', color='green')

            axs[i, 0].axhspan(-1, 0, facecolor='red', alpha=0.2, label='Failure Set')
            axs[i, 0].axhspan(3, 5, facecolor='red', alpha=0.2)
            axs[i, 0].scatter(t_traj[failure_mask], z[failure_mask], color='red', s=25, label='Failure Points')

            axs[i, 0].set_title(f"Trajectory {i+1} - State")
            axs[i, 0].set_xlabel("Time [s]")
            axs[i, 0].set_ylabel("Value")
            axs[i, 0].legend()
            axs[i, 0].grid(True)

            # Plot control
            axs[i, 1].plot(t_ctrl, u, label='u (control)', color='purple')
            axs[i, 1].set_title(f"Trajectory {i+1} - Control -Thrust gain K={K[0].item():.2f}")
            axs[i, 1].set_xlabel("Time [s]")
            axs[i, 1].set_ylabel("u")
            axs[i, 1].legend()
            axs[i, 1].grid(True)

        plt.tight_layout()
        plt.show()
    

    def plot_trajectories_all(self, trajs, controls, dt=None):
        """
        Plot multiple trajectories and corresponding controls in shared figures.
        
        Args:
            trajs: Tensor of shape [N, H+1, 3] — [z, vz, K]
            controls: Tensor of shape [N, H]
            dt: timestep (optional, overrides self.dt)
        """
        if dt is None:
            dt = self.dt

        N, H_plus_1, _ = trajs.shape
        H = H_plus_1 - 1
        time_traj = torch.arange(H_plus_1) * dt
        time_ctrl = torch.arange(H) * dt

        z_vals = trajs[..., 0].numpy()
        control_vals = controls.numpy()

        # --- Plot trajectories ---
        plt.figure(figsize=(8, 4))
        for i in range(N):
            plt.plot(time_traj, z_vals[i], label=f'Traj {i+1}', alpha=0.6)
        plt.axhspan(-1, 0, color='red', alpha=0.1, label="Failure Region")
        plt.axhspan(3, 4, color='red', alpha=0.1)

        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(3.0, color='gray', linestyle='--', linewidth=0.5)

        plt.xlabel('Time [s]')
        plt.ylabel('z (height)')
        plt.title('Drone Altitude Trajectories')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot controls ---
        plt.figure(figsize=(8, 4))
        for i in range(N):
            plt.plot(time_ctrl, control_vals[i], label=f'Control {i+1}', alpha=0.6)
        plt.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(-1.0, color='gray', linestyle='--', linewidth=0.5)

        plt.xlabel('Time [s]')
        plt.ylabel('Control u')
        plt.title('Control Sequences')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# --------------------------------------------------------
def generate_dataset(dynamics, size, N, R, H, u_std, device, return_trajectories=False):
    """
    Parallelized version of generate_dataset using GPU tensor ops.
    Args:
        dynamics: instance of dynamics class (must support .sample_state, .step, .compute_l)
        size: Number of data points
        N: Number of hallucinated trajectories
        R: Number of sampling rounds
        H: Horizon length
        u_std: Gaussian noise std
    Returns:
        List of (t_i, x_i, V̂(t_i, x_i))
        all_trajs: Tensor of shape [size, N, H+1, 3] (if return_trajectories=True)
        all_controls: Tensor of shape [size, N, H] (if return_trajectories=True)
    """
    u_nom = dynamics.nominal_control_init.expand(size, H)  # [size, H]
    
    # Sample all initial states and times at once
    x_i = dynamics.sample_state(batch_size=size)  # [size, 3]
    t_i = torch.rand(size, device=device)  # [size]

    all_trajs = None
    all_controls = None

    for _ in range(R):
        # Expand nominal control: [size, H] -> [size, N, H, 1]
        u_nom_expanded = u_nom[:, None, :, None].expand(size, N, H, 1)
        noise = u_std * torch.randn(size, N, H, 1, device=device)
        u_samples = torch.clamp(u_nom_expanded + noise, -1.0, 1.0)  # [size, N, H, 1]

        # Repeat states for N hallucinated trajectories
        xi_n = x_i[:, None, :].expand(size, N, 3)  # [size, N, 3]
        traj = [xi_n]  # will become [size, N, H+1, 3]

        for h in range(H): # unroll over horizon
            u_h = u_samples[:, :, h, :]  # [size, N, 1]
            xi_n = dynamics.step(xi_n.reshape(-1, 3), u_h.reshape(-1, 1))  # [size*N, 3]
            xi_n = xi_n.view(size, N, 3)
            traj.append(xi_n)

        # Stack into trajectory tensor: [size, N, H+1, 3]
        traj_tensor = torch.stack(traj, dim=2)

        # Compute rewards and min cost per trajectory
        rewards = dynamics.compute_l(traj_tensor.view(-1, H+1, 3))  # [size*N, H+1]
        rewards = rewards.view(size, N, H+1)
        J_n = rewards.min(dim=2).values  # [size, N]

        best_n = torch.argmax(J_n, dim=1)  # [size]
        V_hat = J_n[torch.arange(size), best_n]  # [size]
        u_nom = u_samples[torch.arange(size), best_n, :, 0]  # update nominal control: [size, H]

        if return_trajectories:
            all_trajs = traj_tensor.detach().cpu()  # [size, N, H+1, 3]
            all_controls = u_samples.squeeze(-1).detach().cpu()  # [size, N, H

    # Convert to CPU and Python-native types for saving
    samples = [
        (t_i[i].item(), x_i[i].cpu().numpy(), V_hat[i].item())
        for i in range(size)
    ]
    
    return samples, all_trajs, all_controls
    
