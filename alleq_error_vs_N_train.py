# Import the library
import greenlearning as gl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

np.random.seed(42)

def relative_L2_error(y_true, y_pred, eps=1e-12):
    """
    Uniform-grid discrete relative L2 error, expressed as a %, averaged over examples on axis=0.
    y_true, y_pred shape: (B, ...grid axes...)
    """
    # reduce over all non-batch axes → per-example norms
    reduce_axes = tuple(range(1, y_true.ndim))
    num = np.sqrt(np.sum((y_pred - y_true)**2, axis=reduce_axes))
    den = np.sqrt(np.sum(y_true**2,          axis=reduce_axes))
    rel = num / (den + eps)          # shape (B,)
    return 100 * np.mean(rel)             # scalar

# Parameters to be varied
N_train = int(os.getenv('N_TRAIN'))
print('N_train = ', N_train)
noise_ratio = 1e-2
resample = 1
repeats = range(5)

eq = str(os.getenv('EQUATION'))

filename = f'results/alleq_error_vs_N_train_concurrent/{eq}_error_vs_N_train.csv'

for i in repeats:
    # Construct neural networks for G and homogeneous solution
    G_network = gl.matrix_networks([2] + [50] * 4 + [1], "rational", (1,1))
    U_hom_network = gl.matrix_networks([1] + [50] * 4 + [1], "rational", (1,))

    time_1 = time.time()
    # Define the model
    model = gl.Model(G_network, U_hom_network, N_train=N_train, noise_ratio=noise_ratio, resample=resample, repeat=i)
    model.path_csv = "results_csv"

    # Train the model on the dataset "helmholtz" in the path "examples/datasets/"
    model.train("examples/datasets/", eq)
    time_2 = time.time()

    train_wall_time = time_2 - time_1

    # Save the NNs evaluated at a grid in a csv file
    G_pred = model.save_results(to_file=False)

    # Close the TensorFlow session
    model.sess.close()

    # calculate metrics
    data_path = f'examples/datasets/{eq}.mat'
    raw_data = sp.io.loadmat(data_path)

    N_test = 20

    u = raw_data['F'][::2 * resample, -N_test:] # (100, N_test)
    v = raw_data['U'][::resample, -N_test:] # (100, N_test)

    domain_u = raw_data['X'][::resample, 0] # (100,)

    time_3 = time.time()
    integrand = G_pred[:, :, None] * u[None, :, :]  # (100, 100, N_test)  # broadcast u over columns (y-axis)
    v_est = np.trapz(integrand, x=domain_u, axis=1) # (100, N_test)  # integrate over y (axis=1)
    time_4 = time.time()

    pred_wall_time = time_4 - time_3

    vT = v.T
    v_estT = v_est.T

    error = relative_L2_error(vT, v_estT)

    print(f'relative L2 error: {error:.3f} %')

    df = pd.DataFrame({'eq': [eq], 'N_train': [N_train], 'noise_ratio': [noise_ratio], 'resample': [resample], 'repeat': [i], 'train_wall_time': [train_wall_time], 'pred_wall_time': [pred_wall_time], 'rel_L2_error': [error]})

    # if file exists, append without header
    if os.path.isfile(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        # create file with header
        df.to_csv(filename, mode="w", header=True, index=False)

