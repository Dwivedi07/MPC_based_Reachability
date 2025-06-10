import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Circle

from utils.model import SingleBVPNet
from mpc.dynamics import Quadrotor13D
from utils.util import compute_value_function_stagewise, compute_terminal_value


# ----------------------------
# Settings
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 0.0 # Start time
T_end = 1.0 # Final time
mT = T - T_end
num_stages = 4
Np = 1 # 1 for the decoupled method, 5 for the progressive method

random_ = True  ## toggle this for grid/random trained models
method1 = False
dynamics = Quadrotor13D(device=device)
dyn_name = dynamics.__class__.__name__ 
# ----------------------------
# Grid setup
# at (p z , q ω , q x , q y , q z , v x , v y , v z , ω x , ω y , ω z )
# = 0.54, 0.44, −0.45, 0.27, −0.73, 5.00, −1.07, −3.34, 3.19, −2.80, 3.43)
# ----------------------------
# Fixed values for slicing
p_z, q_omeg, q_x, q_y, q_z = 0.54, 0.44, -0.45, 0.27, -0.73
v_x, v_y, v_z = 5.00, -1.07, -3.34
omeg_x, omeg_y, omeg_z = 3.19, -2.80, 3.43

# X-Y mesh grid
x_vals = np.linspace(-3.0, 3.0, 200)
y_vals = np.linspace(-3.0, 3.0, 200)
x, y = np.meshgrid(x_vals, y_vals)
xy_points = np.stack([x.ravel(), y.ravel()], axis=1)  # [N, 2]

# Construct full state tensor with fixed slices
N = xy_points.shape[0]
other_dims = np.array([p_z, q_omeg, q_x, q_y, q_z, v_x, v_y, v_z, omeg_x, omeg_y, omeg_z])
other_dims_array = np.tile(other_dims, (N, 1))  # [N, 11]
full_state = np.hstack([xy_points, other_dims_array])  # [N, 13]
T_array = np.full((N, 1), T)
x_tensor = torch.tensor(full_state, dtype=torch.float32).to(device)
t_tensor = torch.tensor(T_array, dtype=torch.float32).to(device)

#  ----------------------------
# Load all models from stage_1 to stage_(num_stages-1)
# ----------------------------
models = []
if method1:
    for stage in range(1, num_stages+1): 
        if random_:
            path = f'checkpoints/model_{dyn_name}_checkpoints_random_search_2/stage_{stage}_progressive_{Np}_best.pt'   # random dataset trained model
        else:
            # change the path as per different checkpoints
            path = f'checkpoints/model_{dyn_name}_checkpoints_grid_lrm4/stage_{stage}_progressive_{Np}_best.pt'  # grid based search trained model
            # path = f'checkpoints/model_{dyn_name}_checkpoints_grid_search/stage_{stage}_progressive_{Np}_best.pt'  # grid based search trained model
        model = SingleBVPNet(
            in_features=dynamics.state_dim + 1,  # state_dim + time
            out_features=1,
            hidden_features=512,
            num_hidden_layers=3
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        # print the numebr of parameters in the model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
        print(f"Stage {stage} model loaded with {num_params:.2f} M parameters.")
        models.append(model)

        # ----------------------------
        # Compute Value Function
        # ----------------------------
        with torch.no_grad():
            V_hat = compute_value_function_stagewise(x_tensor, t_tensor, dynamics, models)
            V_hat = V_hat.cpu().numpy().reshape(x.shape)

else:
    # decoupled method
    # Load the model for the last stage
    if random_:
        path = f'checkpoints/model_{dyn_name}_checkpoints_grid_search_decopled/stage_{num_stages}_progressive_{Np}_best.pt'
    else:
        path = f'checkpoints/model_{dyn_name}_checkpoints_grid_search_decopled/stage_{num_stages}_progressive_{Np}_best.pt'
    
    model = SingleBVPNet(out_features=1,  # V(state, t)
                            in_features=dynamics.input_dim,  # z, vz, k, t
                            hidden_features=512, # hidden dimension
                            num_hidden_layers=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    # print the numebr of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    print(f"Stage {num_stages} model loaded with {num_params:.2f} M parameters.")
    models.append(model)

    # ----------------------------
    # Compute Value Function
    # ----------------------------
    with torch.no_grad():
        V_hat = compute_terminal_value(x_tensor, t_tensor, dynamics, models)
        V_hat = V_hat.cpu().numpy().reshape(x.shape)


# ----------------------------
# Plotting
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(x_vals, y_vals, V_hat, levels=100, cmap='RdBu')
plt.colorbar(cf)
ax.contour(x_vals, y_vals, V_hat, levels=[0], colors='black', linestyles='-', linewidths=2)

# Failure set lines red by cylinder at center
obstacle = Circle((0, 0), radius=dynamics.keepout_radius, color='red', alpha=0.3, label='Obstacle')
ax.add_patch(obstacle)

# labels
ax.set_xlabel('$X$ (position x)')
ax.set_ylabel('$y$ (postiion y)')
ax.set_title(f'V(x,T = {mT:.2f}):Value Function Heatmap with Safe and Failure Boundaries')

# legend
safe_proxy = mlines.Line2D([], [], color='black', linestyle='-', label='Safe set boundary ($\\hat{V}=0$)')
fail_proxy = mlines.Line2D([], [], color='red', linestyle='--', label='Failure set')
ax.legend(handles=[safe_proxy, fail_proxy], loc='upper right')
ax.set_aspect('equal')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
plt.tight_layout()

out_path = f'outputs/{dyn_name}_stage_{num_stages}_{Np}_prog_{"random" if random_ else "grid"}.png'
plt.savefig(out_path)
plt.show()
