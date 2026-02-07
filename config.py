import torch
import numpy as np

# ==========================================
# Paths
# ==========================================
DATA_ROOT = './data'
CHECKPOINT_DIR = './checkpoints'
LOG_DIR = './logs'

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 5
WEIGHT_DECAY = 1e-4

# ==========================================
# Physical Parameters (Organic Transistor)
# ==========================================
# 1. Simulation of light intensity for Poisson noise
MAX_PHOTONS = 300  

# 2. Spectral Responsivity Coefficients (RGB Responsivity)
# Input_Channel *= Responsivity_Factor
RGB_RESPONSIVITY = [0.706, 1.0, 0.758] # [R, G, B]

# 3. Real LTP/LTD Response Data (Absolute Current in Amperes)
REAL_LTP_DATA = np.abs([ 
    -1.59E-10, -3.95E-10, -6.12E-10, -8.27E-10, -1.00E-09, -1.16E-09, 
    -1.30E-09, -1.41E-09, -1.51E-09, -1.58E-09, -1.63E-09, -1.67E-09, 
    -1.71E-09, -1.75E-09, -1.78E-09, -1.81E-09, -1.84E-09, -1.86E-09, 
    -1.88E-09, -1.89E-09, -1.90E-09, -1.91E-09, -1.91E-09, -1.92E-09, 
    -1.92E-09, -1.93E-09, -1.93E-09, -1.94E-09, -1.94E-09, -1.94E-09 
]) 

REAL_LTD_DATA = np.abs([ 
    -1.51E-09, -1.18E-09, -9.99E-10, -8.83E-10, -7.71E-10, -6.68E-10, 
    -5.69E-10, -5.00E-10, -4.47E-10, -4.04E-10, -3.72E-10, -3.41E-10, 
    -3.17E-10, -2.95E-10, -2.79E-10, -2.63E-10, -2.49E-10, -2.37E-10, 
    -2.27E-10, -2.18E-10, -2.11E-10, -2.02E-10, -1.96E-10, -1.89E-10, 
    -1.82E-10, -1.74E-10, -1.72E-10, -1.67E-10, -1.62E-10 
]) 

# 4. Conductance Range Mapping
# Neural Network Weights [0, 1] <-> Physical Current [Min, Max]
# Note: Since the network weights are typically centered around 0 (e.g. He initialization),
# we might map [-1, 1] or use a positive-only constraint. 
# However, user mentioned "Normalize this data to a [0, 1] range".
# Standard Conv2d weights are signed. 
# Strategy: We will assume the physical device represents the magnitude or a bias-shifted version.
# For simplicity in `model.py`, we will map the *normalized* weight [0,1] to physical current.
PHYSICAL_MIN_CURRENT = np.min(REAL_LTP_DATA) # approx 1.59e-10 
PHYSICAL_MAX_CURRENT = np.max(REAL_LTP_DATA) # approx 1.94e-9 

# Device Cycle-to-Cycle Variation (5% of dynamic range)
DEVICE_NOISE_STD = (PHYSICAL_MAX_CURRENT - PHYSICAL_MIN_CURRENT) * 0.05

# ==========================================
# Polynomial Fitting for Update Rules
# ==========================================
# We want to model dG/dPulse vs G (normalized).
# 1. Normalize Data to [0, 1]
_ltp_norm = (REAL_LTP_DATA - PHYSICAL_MIN_CURRENT) / (PHYSICAL_MAX_CURRENT - PHYSICAL_MIN_CURRENT)
_ltd_norm = (REAL_LTD_DATA - PHYSICAL_MIN_CURRENT) / (PHYSICAL_MAX_CURRENT - PHYSICAL_MIN_CURRENT)

# 2. Calculate Delta (Change per pulse)
# We assume the data points are sequential pulses.
_delta_ltp = np.diff(_ltp_norm)
_delta_ltd = np.diff(_ltd_norm)

# 3. Fit Polynomial: Delta = P(Current_Value)
# For LTP, we predict delta based on current value REAL_LTP_DATA[:-1]
# For LTD, we predict delta based on current value REAL_LTD_DATA[:-1]
# Degree 3 polynomial should be sufficient to capture saturation.
LTP_POLY = np.polyfit(_ltp_norm[:-1], _delta_ltp, 3)
LTD_POLY = np.polyfit(_ltd_norm[:-1], _delta_ltd, 3)

# ==========================================
# Hardware/System
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4
