import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from utils.model import SingleBVPNet
from mpc.dynamics import VerticalDroneDynamics 
# ----------------------------
# Settings
# ----------------------------
checkpoint_path = 'model_checkpoints_prog/stage_1_progressive_5_best.pt'  # Update if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 0.9 # start time
T_end = 1.20 # final time
K = 5.0  # Fixed gain
dynamics = VerticalDroneDynamics(device=device)

# Grid setup
z_vals = np.linspace(-0.5, 3.5, 200)
vz_vals = np.linspace(-4.0, 4.0, 200)
VZ, Z = np.meshgrid(vz_vals, z_vals) 
input_points = np.stack([Z.ravel(), VZ.ravel()], axis=1)  # Shape: [N, 2]

# Add K to each point to get shape [N, 3]
K_vals = np.full((input_points.shape[0], 1), K)
input_points_with_K_np = np.hstack([input_points, K_vals])  # Shape: [N, 3]

# Convert to PyTorch tensor
input_points_with_K = torch.tensor(input_points_with_K_np, dtype=torch.float32).to(device)

# Compute l(x)
l_x = dynamics.compute_l(input_points_with_K)  # Shape: [N]
T_tensor_end = torch.tensor(T_end, dtype=torch.float32).to(device)
T_tensor = torch.tensor(T, dtype=torch.float32).to(device)

# Add K and t to each point
K_array = np.full((input_points.shape[0], 1), K)
T_array = np.full((input_points.shape[0], 1), T)
input_tensor = torch.tensor(np.hstack([input_points, K_array, T_array]), dtype=torch.float32).to(device)

# ----------------------------
# Load model
# ----------------------------
model = SingleBVPNet(
    in_features=4,
    out_features=1,
    hidden_features=512,
    num_hidden_layers=3
).to(device)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Calculate the Value function 
with torch.no_grad():
    O_pred = model(input_tensor)

V_hat = l_x.unsqueeze(1) + (T_tensor_end - T_tensor) * O_pred
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
plt.title('Value Function Heatmap with Safe and Failure Boundaries')

# legend
safe_proxy = mlines.Line2D([], [], color='black', linestyle='-', label='Safe set boundary ($\\hat{V}=0$)')
fail_proxy = mlines.Line2D([], [], color='red', linestyle='--', label='Failure set')
plt.legend(handles=[safe_proxy, fail_proxy], loc='upper right')

plt.tight_layout()
plt.savefig('outputs/stage1.png')
plt.show()
