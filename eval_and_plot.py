import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.model import SingleBVPNet
import matplotlib.lines as mlines

# ----------------------------
# Settings
# ----------------------------
checkpoint_path = 'model_checkpoints/stage_1_best.pt'  # Update if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 0.9 # Final time
K = 12.0  # Fixed gain

# Grid setup
z_vals = np.linspace(-0.5, 3.5, 200)
vz_vals = np.linspace(-4.0, 4.0, 200)
Z, VZ = np.meshgrid(z_vals, vz_vals)
input_points = np.stack([Z.ravel(), VZ.ravel()], axis=1)  # Shape: [N, 2]

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


with torch.no_grad():
    V_hat = model(input_tensor).cpu().numpy().reshape(Z.shape)


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
