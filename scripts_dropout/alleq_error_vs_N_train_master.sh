#!/bin/bash

eqs=(
    "advection_diffusion_jump"
    # "advection_diffusion"
    # "airy_equation"
    # "biharmonic"
    # "boundary_layer"
    # "cubic_helmholtz"
    # "cusp"
    # "dawson"
    # "helmholtz"
    # "identity"
    # "interior_layer"
    # "jump_green"
    # "laplace"
    # "mean_condition"
    # "negative_helmholtz"
    # "nonlinear_biharmonic"
    # "nonlinear_SL"
    # "periodic_helmholtz"
    # "potential_barrier"
    # "schrodinger"
    # "third_order"
    # "variable_coeffs"
    # "viscous_shock"
)

for E in "${eqs[@]}";
do
    export EQUATION=$E

    for N in 2; # 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 35 40 45 50 55 60 65 70 75;
    do
        export N_TRAIN=$N

        for R in 0; # 1 2 3 4;
        do
            export REPEAT=$R
            sbatch alleq_error_vs_N_train_single.sh
        done
    done
done
