from pathlib import Path
import sys

repo_root = Path.cwd().resolve().parent if Path.cwd().name == "scripts_dropout" else Path.cwd().resolve()
sys.path.insert(0, str(repo_root))

import greenlearning as gl
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

dataset_name = str(os.getenv('EQUATION'))
example_path = "../examples/datasets/"
dropout_rate = 0.02
mc_samples = 50

N_train = int(os.getenv('N_TRAIN'))
noise_ratio = 0e-2
resample = 1
repeat = int(os.getenv('REPEAT'))
N_test = 20

np.random.seed(repeat)  # Set seed for reproducibility of noise

def relative_L2_error(y_true, y_pred, eps=1e-12):
    """
    Uniform-grid discrete relative L2 error, expressed as a %, averaged over examples on axis=0.
    y_true, y_pred shape: (B, ...grid axes...)
    """
    reduce_axes = tuple(range(1, y_true.ndim))
    num = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=reduce_axes))
    den = np.sqrt(np.sum(y_true ** 2, axis=reduce_axes))
    rel = num / (den + eps)
    return 100 * np.mean(rel)

def empirical_joint_log_likelihood(y_true, y_samples, eps=1e-6):
    """
    Gaussian plug-in log likelihood of the full test set under the empirical
    predictive distribution induced by the Monte Carlo samples.

    y_true shape: (Nx, N_test)
    y_samples shape: (S, Nx, N_test)
    """
    sample_mean = np.mean(y_samples, axis=0)
    sample_std = np.std(y_samples, axis=0)
    sample_std = np.maximum(sample_std, eps)
    log_probs = -0.5 * np.log(2 * np.pi * sample_std ** 2) - 0.5 * ((y_true - sample_mean) / sample_std) ** 2
    return np.sum(log_probs), np.mean(log_probs)


path = f'{example_path}{dataset_name}.mat'
raw_data = sp.io.loadmat(path)

u = raw_data['F'][::2 * resample, -N_test:]
print('u.shape', u.shape)
v = raw_data['U'][::resample, -N_test:]
print('v.shape', v.shape)

domain_u = raw_data['X'][::resample, 0]
print('domain_u.shape', domain_u.shape)

records = []

G_network = gl.matrix_networks([2] + [50] * 4 + [1], "rational", (1, 1), dropout_rate=dropout_rate)
U_hom_network = gl.matrix_networks([1] + [50] * 4 + [1], "rational", (1,), dropout_rate=0.0)

model = gl.Model(
    G_network,
    U_hom_network,
    N_train=N_train,
    noise_ratio=noise_ratio,
    resample=resample,
    repeat=repeat,
    adam_steps=10, # 1e3
    lbfgs_steps=10, # 5e4
    dropout_rate=dropout_rate,
    mc_samples=mc_samples,
)

model.train(example_path, dataset_name)
model.save_results(to_file=False)

X_G, Y_G = np.meshgrid(model.x_G, model.y_G)
x_G_star = X_G.flatten()[:, None]
y_G_star = Y_G.flatten()[:, None]
input_data = np.concatenate((x_G_star, y_G_star), axis=1)

G_tensor = model.G_network[0][0].evaluate(input_data, keep_prob=model.keep_prob)
_, _, G_flat_samples = model.mc_predict(G_tensor, mc_samples=mc_samples)
G_samples = G_flat_samples.reshape((mc_samples,) + X_G.shape)

integrand_samples = G_samples[:, :, :, None] * u[None, None, :, :]
v_samples = np.trapz(integrand_samples, x=domain_u, axis=2)

predictive_mean = np.mean(v_samples, axis=0)
predictive_mean_error = relative_L2_error(v.T, predictive_mean.T)
empirical_joint_ll, mean_log_likelihood_per_entry = empirical_joint_log_likelihood(v, v_samples)

records.append({
    'dataset': dataset_name,
    'N_train': N_train,
    'noise_ratio': noise_ratio,
    'resample': resample,
    'repeat': repeat,
    'dropout_rate': dropout_rate,
    'mc_samples': mc_samples,
    'predictive_mean_relative_L2_error': predictive_mean_error,
    'empirical_joint_log_likelihood': empirical_joint_ll,
    'mean_log_likelihood_per_entry': mean_log_likelihood_per_entry,
})

model.sess.close()

df = pd.DataFrame.from_records(records)

filename = f'../results/alleq_error_vs_N_train_concurrent_dropout/{dataset_name}_error_vs_N_train.csv'

# if file exists, append without header
if os.path.isfile(filename):
    df.to_csv(filename, mode="a", header=False, index=False)
else:
    # create file with header
    df.to_csv(filename, mode="w", header=True, index=False)