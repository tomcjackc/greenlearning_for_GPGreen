# Import the library
import greenlearning as gl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

np.random.seed(42)

# Parameters to be varied
N_train = int(os.getenv('N_TRAIN'))
print('N_train = ', N_train)
noise_ratio = 0e-2
resample = 1
repeats = range(5)

filename = 'results_csv_wall_time/train_wall_time_vs_N_train.csv'

for i in repeats:
    # Construct neural networks for G and homogeneous solution
    G_network = gl.matrix_networks([2] + [50] * 4 + [1], "rational", (1,1))
    U_hom_network = gl.matrix_networks([1] + [50] * 4 + [1], "rational", (1,))

    s_time = time.time()
    # Define the model
    model = gl.Model(G_network, U_hom_network, N_train=N_train, noise_ratio=noise_ratio, resample=resample, repeat=i)

    # Train the model on the dataset "helmholtz" in the path "examples/datasets/"
    model.train("examples/datasets/","helmholtz")
    e_time = time.time()

    time_elapsed = e_time - s_time

    df = pd.DataFrame({'N_train': [N_train], 'noise_ratio': [noise_ratio], 'resample': [resample], 'repeat': [i], 'train_wall_time': [time_elapsed]})

    # if file exists, append without header
    if os.path.isfile(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        # create file with header
        df.to_csv(filename, mode="w", header=True, index=False)

    # Close the TensorFlow session
    model.sess.close()