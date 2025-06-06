import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def compute_value_function_stagewise(x, t, dynamics, models):
    """
    Computes V(x, t) using models trained on multiple backward-in-time stages.
    
    Args:
        x: Tensor of shape [N, 3] (z, vz, K)
        t: Tensor of shape [N, 1] (times at which to evaluate V)
        dynamics: includes compute_l(x) and T_terminals {stage: torch.tensor([T])}
        models: list of trained models for each stage (stages 1 through 4)
    
    Returns:
        V: Tensor of shape [N, 1] containing value function estimates at (x, t)
    """
    device = x.device
    l_x = dynamics.compute_l(x)
    if l_x.dim() == 1:
        l_x = l_x.unsqueeze(1)
    V = l_x
    T_i = t[0][0].item()

    '''
    Check first where does T_i values lie : in first stage/second/third/fourth
    if first: return l_x + (T_terminal_stage_1 - T_i)* O_pred_stage_1_model(x,T_i)
    if second:  return l_x + (T_terminal_stage_1 - T_terminal_stage_2)* O_pred_stage_1_model(x,T_terminal_stage_2) + (T_terminal_stage_2 - T_i)* O_pred_stage_2_model(x, T_i)
    and so on till stage 4
    '''

    for i, model in enumerate(models):  
        stage = i + 1
        T_end = dynamics.T_terminals[stage].to(x.device)       
        T_start = dynamics.T_terminals[stage+1].to(x.device) if stage < len(models) else torch.tensor(0.0).to(device)
       
        if T_start.item() <= T_i and T_i <= T_end.item():
            # In this stage
            print('This value is in stage', stage)
            delta_T = T_end - T_i
            input_tensor = torch.cat([x, t], dim=1)  # t has same value across batch
            with torch.no_grad():
                O_pred = model(input_tensor)
            V = V + delta_T * O_pred
            break

        else:
            # Outside this stage — use full duration of model
            print('Overpassing the stage', stage)
            input_tensor = torch.cat([x, T_start.expand(x.shape[0], 1)], dim=1)
            with torch.no_grad(): 
                O_pred = model(input_tensor) # prediction at start point stamp as not in this stage
            V = V + (T_end - T_start) * O_pred

    return V



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
    # print('this is', T_i)

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

def generate_dataset(dynamics, size, N, R, H, u_std, stage, device, prev_stage_models = None, 
                    return_trajectories=False, use_grid_sampling=False):
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
        use_grid_sampling: Whether to use grid-based round-robin sampling of states

    Returns:
        samples: List of tuples (t_i, x_i, V̂(t_i, x_i))
        all_trajs: Tensor [size, R, H+1, 3] (if return_trajectories=True)
        all_controls: Tensor [size, R, H]   (if return_trajectories=True)
    """
    samples = []
    all_trajs = []
    all_controls = []

    for n in tqdm(range(size), desc="D_MPC samples", ncols=80, unit="sample"):
        x_i = dynamics.sample_state(n, size, batch_size=1, use_grid_sampling = use_grid_sampling).to(device)  # [1, 3]
        t_i, t_f = dynamics.sample_time(stage, device)  #  [1] [1]
        H_rollout = torch.ceil((t_f - t_i) / dynamics.dt).long() 
        
        u_nom_init = dynamics.nominal_control_init.to(device).expand(H_rollout)  # [H_r]
        u_nom = u_nom_init.clone()
        best_V_hat = None

        traj_r_list = []
        control_r_list = []

        for _ in range(R):
            u_nom_expanded = u_nom[None, :].expand(N, H_rollout)  # [N, H_r]
            noise = u_std * torch.randn(N, H_rollout, device=device) # std* mean zero and variance 1 guassian distr
            u_samples = torch.clamp(u_nom_expanded + noise, -1.0, 1.0)  # [N, H_r]

            xi_n = x_i.expand(N, 3)  # [N, 3]
            traj = [xi_n]
            '''
            Rollout each traj to the terminal time of that stage
            '''
            for h in range(H_rollout): # H_rollout is possible horizon for this sample
                u_h = u_samples[:, h]  # [N]
                xi_n = dynamics.step(xi_n, u_h)
                xi_n = xi_n.view(N, 3)
                traj.append(xi_n)
            
            traj_tensor = torch.stack(traj, dim=1)  # [N, H_r+1, 3]
            rewards = dynamics.compute_l(traj_tensor)  # [N, H_r+1]
            min_l_x_h = rewards.min(dim=1).values  # [N]

            # we will take min between the l_x and prev_model_val_fnc
            if stage != 1:
                ''' 
                Evaluate the value-fuction from just previous stage model evaluated 
                at terminal state and end of this stage's horizon
                V(x,t) = min(min{t}(l(x(t))), V_prev(x_term,T_term))
               
                '''
                x_terminal = traj_tensor[:, -1, :]  # [N, 3]
                t_terminal = t_f.expand(N, 1)       # [N, 1]
                assert t_terminal.shape[0] == N
                assert t_terminal.shape[1] == 1 
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
            best_traj = traj_tensor[best_n]  # [H_r+1, 3]
            best_control = u_samples[best_n]  # [H_r]
            
            # BOOTSTRAP: Add intermediate points along the best trajectory to dataset
            # We compute: V̂(t + h·dt, x_h) = min(l(x_h:)), where h in [0, H]
            l_vals = dynamics.compute_l(best_traj)  # [H_r+1]
            h_min = torch.argmin(l_vals).item()  # scalar: the index where l_vals is minimal

            l_vals_truncated = l_vals[:h_min + 1]  
            best_traj_truncated = best_traj[:h_min + 1]

            running_min = torch.minimum(torch.cummin(l_vals_truncated.flip(0), dim=0).values.flip(0), best_V_hat)  
            # Add truncated samples to the dataset
            for h in range(h_min + 1):
                t_h = float(t_i + h * dynamics.dt)
                x_h = best_traj_truncated[h].cpu().numpy()
                V_h = running_min[h].item()
                samples.append((t_h, x_h, V_h))
        
            traj_r_list.append(best_traj.cpu())
            control_r_list.append(best_control.cpu())
        
        all_trajs.append(torch.stack(traj_r_list))     # shape [R, H_r'+1, 3]
        all_controls.append(torch.stack(control_r_list))  # shape [R, H_r']

    if return_trajectories:
        return samples, all_trajs, all_controls
    else:
        return samples


def generate_dataset_multi(dynamics, size, N, R, H, u_std, stage, device, prev_stage_models = None, 
                    return_trajectories=False, use_grid_sampling=False):
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
        use_grid_sampling: Whether to use grid-based round-robin sampling of states

    Returns:
        samples: List of tuples (t_i, x_i, V̂(t_i, x_i))
        all_trajs: Tensor [size, R, H+1, 3] (if return_trajectories=True)
        all_controls: Tensor [size, R, H]   (if return_trajectories=True)
    """
    samples = []
    all_trajs = []
    all_controls = []

    for n in tqdm(range(size), desc="D_MPC samples", ncols=80, unit="sample"):
        x_i = dynamics.sample_state(n, size, batch_size=1, use_grid_sampling = use_grid_sampling).to(device)  # [1, 3]
        t_i, t_f = dynamics.sample_time(stage, device)  #  [1] [1]
        H_rollout = torch.ceil((t_f - t_i) / dynamics.dt).long() 
        
        # u_nom_init = dynamics.nominal_control_init.to(device).expand(H_rollout)  # [H_r]
        u_nom_init = dynamics.nominal_control_init.to(device).unsqueeze(0).expand(H_rollout, -1)
        u_nom = u_nom_init.clone()
        best_V_hat = None

        traj_r_list = []
        control_r_list = []

        for _ in range(R):
            # u_nom_expanded = u_nom[None, :].expand(N, H_rollout)  # [N, H_r]
            # noise = u_std * torch.randn(N, H_rollout, device=device) # std* mean zero and variance 1 guassian distr
            # u_samples = torch.clamp(u_nom_expanded + noise, -1.0, 1.0)  # [N, H_r]
            u_nom_expanded = u_nom.unsqueeze(0).expand(N, H_rollout, -1)  # [N, H_r, control_dim]
            noise = u_std * torch.randn(N, H_rollout, dynamics.control_dim, device=device)  # [N, H_r, control_dim]
            u_samples = torch.clamp(u_nom_expanded + noise, -1.0, 1.0)  # [N, H_r, control_dim]

            xi_n = x_i.expand(N, dynamics.state_dim)  # [N, state_dim]
            traj = [xi_n]
            '''
            Rollout each traj to the terminal time of that stage
            '''
            for h in range(H_rollout): # H_rollout is possible horizon for this sample
                # u_h = u_samples[:, h]  # [N]
                u_h = u_samples[:, h, :]  # [N, control_dim
                xi_n = dynamics.step(xi_n, u_h)
                xi_n = xi_n.view(N, dynamics.state_dim)
                traj.append(xi_n)
            
            traj_tensor = torch.stack(traj, dim=1)  # [N, H_r+1, state_dim]
            rewards = dynamics.compute_l(traj_tensor)  # [N, H_r+1]
            min_l_x_h = rewards.min(dim=1).values  # [N]

            # we will take min between the l_x and prev_model_val_fnc
            if stage != 1:
                ''' 
                Evaluate the value-fuction from just previous stage model evaluated 
                at terminal state and end of this stage's horizon
                V(x,t) = min(min{t}(l(x(t))), V_prev(x_term,T_term))
               
                '''
                x_terminal = traj_tensor[:, -1, :]  # [N, state_dim]
                t_terminal = t_f.expand(N, 1)       # [N, 1]
                assert t_terminal.shape[0] == N
                assert t_terminal.shape[1] == 1 
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
            best_traj = traj_tensor[best_n]  # [H_r+1, 3]
            best_control = u_samples[best_n]  # [H_r, control_dim]
            
            # BOOTSTRAP: Add intermediate points along the best trajectory to dataset
            # We compute: V̂(t + h·dt, x_h) = min(l(x_h:)), where h in [0, H]
            l_vals = dynamics.compute_l(best_traj)  # [H_r+1]
            h_min = torch.argmin(l_vals).item()  # scalar: the index where l_vals is minimal

            l_vals_truncated = l_vals[:h_min + 1]  
            best_traj_truncated = best_traj[:h_min + 1]

            running_min = torch.minimum(torch.cummin(l_vals_truncated.flip(0), dim=0).values.flip(0), best_V_hat)  
            # Add truncated samples to the dataset
            for h in range(h_min + 1):
                t_h = float(t_i + h * dynamics.dt)
                x_h = best_traj_truncated[h].cpu().numpy()
                V_h = running_min[h].item()
                samples.append((t_h, x_h, V_h))
        
            traj_r_list.append(best_traj.cpu())
            control_r_list.append(best_control.cpu())
        
        all_trajs.append(torch.stack(traj_r_list))     # shape [R, H_r'+1, 3]
        all_controls.append(torch.stack(control_r_list))  # shape [R, H_r', control_dim]

    if return_trajectories:
        return samples, all_trajs, all_controls
    else:
        return samples