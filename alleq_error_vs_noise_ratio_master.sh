#!/bin/bash

eqs=(
    "advection_diffusion_jump"
    "advection_diffusion"
    "airy_equation"
    "biharmonic"
    "boundary_layer"
    "cubic_helmholtz"
    "cusp"
    "dawson"
    "helmholtz"
    "identity"
    "interior_layer"
    "jump_green"
    "laplace"
    "mean_condition"
    "negative_helmholtz"
    "nonlinear_biharmonic"
    "nonlinear_SL"
    "periodic_helmholtz"
    "potential_barrier"
    "schrodinger"
    "third_order"
    "variable_coeffs"
    "viscous_shock"
)

for E in "${eqs[@]}";
do
    export EQUATION=$E

    for N in 0.01020408 0.02040816 0.03061224 0.04081633 0.05102041 0.06122449 0.07142857 0.08163265 0.09183673 0.10204082 0.1122449  0.12244898 0.13265306 0.14285714 0.15306122 0.16326531 0.17346939 0.18367347 0.19387755 0.20408163 0.21428571 0.2244898  0.23469388 0.24489796 0.25510204 0.26530612 0.2755102  0.28571429 0.29591837 0.30612245 0.31632653 0.32653061 0.33673469 0.34693878 0.35714286 0.36734694 0.37755102 0.3877551  0.39795918 0.40816327 0.41836735 0.42857143 0.43877551 0.44897959 0.45918367 0.46938776 0.47959184 0.48979592 0.5;
    do
    export NOISE_RATIO=$N
    sbatch alleq_error_vs_noise_ratio_single.sh
    done
done
