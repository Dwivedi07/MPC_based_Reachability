import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import imageio
import os

from utils.model import SingleBVPNet
from mpc.dynamics import VerticalDroneDynamics 
from utils.util import compute_value_function_stagewise, compute_terminal_value

# ----------------------------
# Settings
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_end = 1.2
T_start = 0.0
K = 12.0
num_stages = 4
Np = 5
dynamics = VerticalDroneDynamics(device=device)
dyn_name = dynamics.__class__.__name__ 
gif_frames_dir = f"gif_frames_{dyn_name}"
os.makedirs(gif_frames_dir, exist_ok=True)

# ----------------------------
# Grid setup
# ----------------------------
z_vals = np.linspace(-0.5, 3.5, 200)
vz_vals = np.linspace(-4.0, 4.0, 200)
VZ, Z = np.meshgrid(vz_vals, z_vals)
input_points = np.stack([Z.ravel(), VZ.ravel()], axis=1)

K_array = np.full((input_points.shape[0], 1), K)
x_tensor_base = torch.tensor(np.hstack([input_points, K_array]), dtype=torch.float32).to(device)

# ----------------------------
# Load models
# ----------------------------
random_ = False
method1 = False
models = []

if method1:
    for stage in range(1, num_stages + 1):
        if random_:
            path = f'checkpoints/model_{dyn_name}_checkpoints_random_search/stage_{stage}_progressive_{Np}_best.pt'
        else:
            path = f'checkpoints/model_{dyn_name}_checkpoints_grid_third_iter/stage_{stage}_progressive_{Np}_best.pt'
        model = SingleBVPNet(dynamics.input_dim, 1, 512, 3).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
else:
    path = f'checkpoints/model_{dyn_name}_checkpoints_grid_search_decopled/stage_{num_stages}_progressive_{Np}_best.pt'
    model = SingleBVPNet(dynamics.input_dim, 1, 512, 3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    models.append(model)

# ----------------------------
# Generate frames for GIF
# ----------------------------
frames = []
time_vals = np.linspace(T_end, T_start, num=13)  # From 1.2 to 0.0 in 0.1 steps
for i, T in enumerate(time_vals):
    mT = T - T_end
    T_array = np.full((input_points.shape[0], 1), T)
    t_tensor = torch.tensor(T_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        if method1:
            V_hat = compute_value_function_stagewise(x_tensor_base, t_tensor, dynamics, models)
        else:
            V_hat = compute_terminal_value(x_tensor_base, t_tensor, dynamics, models)
        V_hat = V_hat.cpu().numpy().reshape(Z.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(vz_vals, z_vals, V_hat, levels=100, cmap='RdBu')
    plt.colorbar(cf, ax=ax)
    ax.contour(vz_vals, z_vals, V_hat, levels=[0], colors='black', linestyles='-', linewidths=2)
    ax.axhline(y=0.0, color='red', linestyle='--', linewidth=2)
    ax.axhline(y=3.0, color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('$v_z$ (velocity)')
    ax.set_ylabel('$z$ (height)')
    ax.set_title(f'V(x,T = {mT:.2f})')

    safe_proxy = mlines.Line2D([], [], color='black', linestyle='-', label='$\hat{V}=0$')
    fail_proxy = mlines.Line2D([], [], color='red', linestyle='--', label='Failure set')
    ax.legend(handles=[safe_proxy, fail_proxy], loc='upper right')
    
    fname = os.path.join(gif_frames_dir, f'frame_{i:03d}.png')
    plt.tight_layout()
    plt.savefig(fname)
    frames.append(fname)
    plt.close(fig)

# ----------------------------
# Create GIF
# ----------------------------
gif_path = f"outputs/value_function_{dyn_name}_T12_to_T0.gif"
os.makedirs("outputs", exist_ok=True)
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

print(f"GIF saved to {gif_path}")
