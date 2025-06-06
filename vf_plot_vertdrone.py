import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from utils.model import SingleBVPNet
from mpc.dynamics import VerticalDroneDynamics 
from utils.util import compute_value_function_stagewise


# ----------------------------
# Settings
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 0.0 # Start time
T_end = 1.2 # Final time
mT = T - T_end
K = 12.0  # Fixed gain
num_stages = 4
Np = 7
dynamics = VerticalDroneDynamics(device=device)
dyn_name = dynamics.__class__.__name__ 
# ----------------------------
# Grid setup
# ----------------------------
z_vals = np.linspace(-0.5, 3.5, 200)
vz_vals = np.linspace(-4.0, 4.0, 200)
VZ, Z = np.meshgrid(vz_vals, z_vals) 
input_points = np.stack([Z.ravel(), VZ.ravel()], axis=1)  # Shape: [N, 2]

# Build input tensor with K and T
K_array = np.full((input_points.shape[0], 1), K)
T_array = np.full((input_points.shape[0], 1), T)
x_tensor = torch.tensor(np.hstack([input_points, K_array]), dtype=torch.float32).to(device)  # [N, 3]
t_tensor = torch.tensor(T_array, dtype=torch.float32).to(device)  # [N, 1]

random_ = False
#  ----------------------------
# Load all models from stage_1 to stage_(num_stages-1)
# ----------------------------
models = []
for stage in range(1, num_stages+1): 
    if random_:
        path = f'checkpoints/model_{dyn_name}_checkpoints_random_search/stage_{stage}_progressive_{Np}_best.pt'   # random dataset trained model
    else:
        # change the path as per different checkpoints
        path = f'checkpoints/model_{dyn_name}_checkpoints_grid_lrm4/stage_{stage}_progressive_{Np}_best.pt'  # grid based search trained model
        # path = f'checkpoints/model_{dyn_name}_checkpoints_grid_search/stage_{stage}_progressive_{Np}_best.pt'  # grid based search trained model
    model = SingleBVPNet(
        in_features=4,
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


with torch.no_grad():
    V_hat = compute_value_function_stagewise(x_tensor, t_tensor, dynamics, models)
    V_hat = V_hat.cpu().numpy().reshape(Z.shape)

# Plot
plt.figure(figsize=(8, 6))
cf = plt.contourf(vz_vals, z_vals, V_hat, levels=100, cmap='RdBu')
plt.colorbar(cf)
plt.contour(vz_vals, z_vals, V_hat, levels=[0], colors='black', linestyles='-', linewidths=2)

# Failure set lines red
plt.axhline(y=0.0, color='red', linestyle='--', linewidth=2)
plt.axhline(y=3.0, color='red', linestyle='--', linewidth=2)

# labels
plt.xlabel('$v_z$ (velocity)')
plt.ylabel('$z$ (height)')
plt.title(f'V(x,T = {mT:.2f}):Value Function Heatmap with Safe and Failure Boundaries')

# legend
safe_proxy = mlines.Line2D([], [], color='black', linestyle='-', label='Safe set boundary ($\\hat{V}=0$)')
fail_proxy = mlines.Line2D([], [], color='red', linestyle='--', label='Failure set')
plt.legend(handles=[safe_proxy, fail_proxy], loc='upper right')

plt.tight_layout()
if random_:
    plt.savefig(f'outputs/{dyn_name}_stage_{num_stages}_{Np}_prog_random.png')
else:
    plt.savefig(f'outputs/{dyn_name}_stage_{num_stages}_{Np}_prog_grid.png')
plt.show()
