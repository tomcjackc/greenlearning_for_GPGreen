# Import the library
import greenlearning as gl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os

# Parameters to be varied
N_train = 75
noise_ratio = float(os.getenv('NOISE_RATIO'))
print('noise_ratio = ', noise_ratio)
resample = 1
repeats = range(5)

for i in repeats:
    # Construct neural networks for G and homogeneous solution
    G_network = gl.matrix_networks([2] + [50] * 4 + [1], "rational", (1,1))
    U_hom_network = gl.matrix_networks([1] + [50] * 4 + [1], "rational", (1,))

    # Define the model
    model = gl.Model(G_network, U_hom_network, N_train=N_train, noise_ratio=noise_ratio, resample=resample, repeat=i)
    model.path_csv = "results_csv"

    # Train the model on the dataset "helmholtz" in the path "examples/datasets/"
    model.train("examples/datasets/","helmholtz")

    # Save the NNs evaluated at a grid in a csv file
    model.save_results()

    # Close the TensorFlow session
    model.sess.close()