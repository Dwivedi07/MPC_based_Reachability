import torch
import matplotlib.pyplot as plt
import numpy as np

class VerticalDroneDynamics:
    def __init__(self, device):
        self.g = 9.8       # g
        self.dt = 0.03 # each stage horizon 0.3 seconds long with 100 steps
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
        elif stage == 2:
            t_i = torch.round((0.6 + torch.rand(1, device=device) * self.horizon) * 100) / 100   # t_i in [0.6, 0.9] with 2 decimal places
        elif stage == 3:
            t_i = torch.round((0.3 + torch.rand(1, device=device) * self.horizon) * 100) / 100   # t_i in [0.3, 0.6] with 2 decimal places
        else:
            t_i = torch.round((0.0 + torch.rand(1, device=device) * self.horizon) * 100) / 100   # t_i in [0.0, 0.3] with 2 decimal places
        
        return t_i


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
        ### for second algo
        # print(trajs.shape, controls.shape)
        # trajs = trajs.squeeze(1)
        # controls = controls.squeeze(1)
        ###
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
        plt.savefig('drone_altitude.png')
        plt.show()

        # --- Plot controls ---
        plt.figure(figsize=(8, 4))
        for i in range(N):
            plt.plot(time_ctrl, control_vals[i], label=f'Control {i+1}', alpha=0.6)
        # plt.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
        # plt.axhline(-1.0, color='gray', linestyle='--', linewidth=0.5)

        plt.xlabel('Time [s]')
        plt.ylabel('Control u')
        plt.title('Control Sequences')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('drone_controls.png')
        plt.show()
#-----------------------------------------------
def compute_recursive_value(x, t, dynamics, models):
    """
    Recursively computes the predicted value function at (x, T_stage) using trained models.
    - x: [batch_size, 3]
    - T_stage: [batch_size, 1]
    - models: list of trained models from stage 1 to current-1
    - stage
    
    V_pred_current(x_sample,t_smaple) = V_pred_prev[x_sample, T_terminal_current] + (T_terminal_current - t)*O_pred_2(x_sample,t_smaple)
    V_pred_prev[x_sample, T_terminal_current] = l_x + (T_terminal_prev - T_terminal)*O_pred_prev[x_sample, T_terminal_current]
    """
    l_x = dynamics.compute_l(x)
    T_terminals = dynamics.T_terminals
    V = l_x
    if V.dim() == 1:
        V = V.unsqueeze(1)
    T_i = t

    for i, model in enumerate(models):  # stages 1 to current-1
        stage_i = i + 1
        T_ip1 = T_terminals[stage_i+1].to(x.device).expand(x.shape[0], 1)
        input_tensor = torch.cat([x, T_ip1], dim=1) # O_thistage eval at terminal of prev state
        with torch.no_grad():
            O_pred = model(input_tensor)
        
        delta_T = (T_i - T_ip1)  # [N, 1]
        V = V + delta_T * O_pred  # all [N, 1]
        T_i = T_ip1
    
    return V  # shape: [batch_size, 1]
#=---------------------------------------------
def generate_dataset(dynamics, size, N, R, H, u_std, stage, device, prev_stage_models = None, 
                    return_trajectories=False):
    """
    Parallelized version of generate_dataset using GPU tensor ops.
    Args:
        dynamics: instance of dynamics class (must support .sample_state, .step, .compute_l)
        size: Number of data points |D_MPC|
        N: Number of hallucinated trajectories
        R: Number of sampling rounds for one sampled data points corresponding to one initial state
        H: Horizon length
        u_std: Gaussian noise std
        device: CUDA or CPU device
        prev_stage_models : previous stage models
        return_trajectories: If True, returns raw trajectories and control inputs

    Returns:
        samples: List of tuples (t_i, x_i, V̂(t_i, x_i))
        all_trajs: Tensor [size, R, H+1, 3] (if return_trajectories=True)
        all_controls: Tensor [size, R, H]   (if return_trajectories=True)
    """
    samples = []
    all_trajs = []
    all_controls = []

    u_nom_init = dynamics.nominal_control_init.to(device).expand(H)  # [H]

    for n in range(size):
        x_i = dynamics.sample_state(batch_size=1).to(device)  # [1, 3]
        t_i = dynamics.sample_time(stage, device)  #  [1]

        u_nom = u_nom_init.clone()
        best_V_hat = None

        traj_r_list = []
        control_r_list = []

        for _ in range(R):
            u_nom_expanded = u_nom[None, :].expand(N, H)  # [N, H]
            noise = u_std * torch.randn(N, H, device=device)
            u_samples = torch.clamp(u_nom_expanded + noise, -1.0, 1.0)  # [N, H]

            xi_n = x_i.expand(N, 3)  # [N, 3]
            traj = [xi_n]

            for h in range(H):
                u_h = u_samples[:, h]  # [N]
                xi_n = dynamics.step(xi_n, u_h)
                xi_n = xi_n.view(N, 3)
                traj.append(xi_n)
            
            traj_tensor = torch.stack(traj, dim=1)  # [N, H+1, 3]
            rewards = dynamics.compute_l(traj_tensor)  # [N, H+1]
            min_l_x_h = rewards.min(dim=1).values  # [N]

            # we will take min between the l_x and prev_model_val_fnc
            if stage != 1:
                ''' 
                evaluate the value-fuction from just previous stage model evaluated 
                at terminal state and end of this stage's horizon
                V(x,t) = min(min{t}(l(x(t))), V_prev(x_term,t_term))
                As for stage i>1:
                    t \in [T - iH, T - (i-1)H)]
                    so sampled start state t >= T - iH
                    so t + dt*H = t + 0.3 = t + H \in [t - (i-1)H, t - (i-2)H]
                    so valid call of O_theta
                '''
                x_terminal = traj_tensor[:, -1, :]  # [N, 3]
                t_terminal = t_i.expand(N, 1) + H * dynamics.dt  # [N, 1]
                # recursive call
                V_at_terminal_state = compute_recursive_value(x_terminal, t_terminal, dynamics, prev_stage_models) 
                J_n = torch.minimum(V_at_terminal_state.reshape(-1), min_l_x_h)  # [N] : min( min(l), V_theta)
            else:
                J_n = min_l_x_h  # fallback if no previous model

            best_n = torch.argmax(J_n)  # scalar
            V_hat = J_n[best_n]

            if best_V_hat is None or V_hat > best_V_hat:
                best_V_hat = V_hat

            # Update nominal control and store best trajectory + control
            u_nom = u_samples[best_n]
            best_traj = traj_tensor[best_n]  # [H+1, 3]
            best_control = u_samples[best_n]  # [H]
            
            # BOOTSTRAP: Add intermediate points along the best trajectory to dataset
            # We compute: V̂(t + h·dt, x_h) = min(l(x_h:)), where h in [0, H]
            l_vals = dynamics.compute_l(best_traj)  # [H+1]
            running_min = torch.minimum(torch.cummin(l_vals.flip(0), dim=0).values.flip(0), best_V_hat)  # [H+1]

            for h in range(H+1):
                t_h = float(t_i + h * dynamics.dt)
                x_h = best_traj[h].cpu().numpy()
                V_h = running_min[h].item()
                samples.append((t_h, x_h, V_h))

            traj_r_list.append(best_traj.cpu())
            control_r_list.append(best_control.cpu())
        
        # samples.append((float(t_i), x_i.squeeze(0).cpu().numpy(), best_V_hat.item()))
        all_trajs.append(torch.stack(traj_r_list))     # shape [R, H+1, 3]
        all_controls.append(torch.stack(control_r_list))  # shape [R, H]

    if return_trajectories:
        return samples, torch.stack(all_trajs), torch.stack(all_controls)
    else:
        return samples
