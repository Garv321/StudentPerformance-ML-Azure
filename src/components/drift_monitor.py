
import numpy as np
import mlflow

def calculate_psi(expected, actual):
    return np.mean((actual - expected) * np.log((actual+1e-6)/(expected+1e-6)))

def log_drift(expected, actual):
    psi = calculate_psi(expected, actual)
    mlflow.log_metric("psi_drift", psi)
    return psi
