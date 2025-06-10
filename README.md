# MPC-Based Reachability with Neural Value Function Approximation

This repository contains code for training and evaluating neural network models that approximate value functions for reachability analysis in dynamical systems. We use datasets generated from Model Predictive Control (MPC) simulations for both a 2D vertical drone and a 13D quadrotor.

## 📁 Repository Structure

MPC_based_Reachability/
│
├── checkpoints/ # Trained model checkpoints
│ └── model_{dynamics}checkpoints/
│
├── dataset/ # Trajectory datasets from MPC rollouts
│ ├── VerticalDroneDynamics/
│ └── Quadrotor13D/
│ └── stage{1,2,3,4}/
│ └── dataset{grid, random, multidim}/
│
├── mpc/
│ └── dynamics.py # Dynamics for VerticalDrone and Quadrotor13D
│
├── utils/ # Utility functions and model definition
│ ├── datasetio.py # Dataset loading and preprocessing
│ ├── model.py # Neural network model architecture
│ ├── quaternion.py # Quaternion utilities for 13D quadrotor
│ └── util.py # Miscellaneous helper functions
│
├── eval_and_plot.py # Script for evaluation and visualization
├── gt_pred.ipynb # Jupyter Notebook for GT vs. prediction comparison
│
├── main_train.py # Standard training loop
├── main_train_decoupled.py # Decoupled dynamics training
├── main_train_prog.py # Progressive horizon training
├── main_train_quad.py # Quadrotor-specific training script
│
├── vf_gif.py # Value function animation (gif)
├── vf_plot_quad.py # Value function plotting for Quadrotor
├── vf_plot_vertdrone.py # Value function plotting for Vertical Drone
│
├── finalreport/ # Final report or related documentation
├── requirements.txt # Python dependencies
└── README.md # You're here!
---

## 🛠️ Getting Started

### 🔧 Environment Setup

We recommend using a `conda` environment to manage dependencies. Run the following commands:

```bash
# Create and activate a new conda environment
conda create -n mpc_reachability python=3.10 -y
conda activate mpc_reachability

# Install dependencies
pip install -r requirements.txt
```
# 🚀 Usage Guide
## 🔁 Training
```bash
# Standard training for Vertical Drone
python main_train.py

# Progressive horizon training
python main_train_prog.py

# Decoupled training strategy
python main_train_decoupled.py

# Training for 13D Quadrotor system
python main_train_quad.py
```
Checkpoints are saved to checkpoints/model_{dynamics}_checkpoints.

## 📊 Evaluation & Visualization
After training, use the following tools to evaluate and visualize value function predictions:
```bash
python eval_and_plot.py                 # Evaluate trained models
python vf_plot_vertdrone.py            # Plot for vertical drone
python vf_plot_quad.py                 # Plot for quadrotor
python vf_gif.py                       # Generate GIF visualizations
```

You can also launch the Jupyter Notebook for detailed comparison of ground truth vs. predicted values:
```bash
jupyter notebook gt_pred.ipynb
```
## 📂 Dataset Structure
dataset/
├── VerticalDroneDynamics/
│   └── stage1/
│       ├── dataset_grid/
│       ├── dataset_random/
│       └── dataset_multidim/
│   └── ...
└── Quadrotor13D/
    └── stage{1-4}/
        └── dataset_{grid,random,multidim}/
Each dataset contains (t, x, V̂(t, x)) tuples sampled using MPC rollouts.
# Models
The neural network architectures for value function approximation are defined in utils/model.py. They support progressive training strategies and decoupled dynamics representations.

