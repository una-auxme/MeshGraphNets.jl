#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using MeshGraphNets

import OrdinaryDiffEq: Euler, Tsit5
import Optimisers: Adam

######################
# Network parameters #
######################

message_steps = 15
layer_size = 128
hidden_layers = 2
batch = 1
epo = 1
ns = 10e6
norm_steps = 1000
cuda = true
cp_derivative = 10000
cp_solver = 10
ad = :Zygote

########################
# Node type parameters #
########################

types_inflow = [4]
types_updated = [0, 5]
types_noisy = [0]
noise_stddevs = [0.02f0]

########################
# Optimiser parameters #
########################

learning_rate_derivative = 1.0f-4
opt_derivative = Adam(learning_rate_derivative)
# opt_derivative = nothing

learning_rate_solver = 1.0f-2
opt_solver = Adam(learning_rate_solver)
# opt_solver = nothing

#########################
# Paths to data folders #
#########################

ds_path = "../data/CylinderFlow/data"
chk_path = "../data/CylinderFlow/chk"
eval_path = "../data/CylinderFlow/eval"

#######################
# Simulation interval #
#######################

tstart = 0.0f0
dt = 0.01f0
tstop = 5.99f0

# timesteps at which the mean squared error is calculated and printed during evaluation
mse_steps = vcat(collect(tstart:1.0f0:tstop), tstop)

###########
# Solvers #
###########

solver_train = Tsit5()
solver_eval_fixed_timesteps = Euler()
solver_eval_adaptive_timesteps = Tsit5()

#################
# Train network #
#################

# with DerivativeTraining

# train_network(
#     opt_derivative, ds_path, chk_path; mps = message_steps, layer_size = layer_size,
#     hidden_layers = hidden_layers, batchsize = batch, epochs = epo, steps = Int(ns),
#     use_cuda = cuda, checkpoint = cp_derivative, norm_steps = 1000, types_inflow = types_inflow,
#     types_updated = types_updated, types_noisy = types_noisy, noise_stddevs = noise_stddevs,
#     training_strategy = DerivativeTraining(), ad = ad
# )

# with SolverTraining

train_network(
    opt_solver, ds_path, chk_path; mps = message_steps, layer_size = layer_size,
    hidden_layers = hidden_layers, batchsize = batch, epochs = epo, steps = Int(ns),
    use_cuda = cuda, checkpoint = cp_solver, norm_steps = 1000, types_inflow = types_inflow,
    types_updated = types_updated, types_noisy = types_noisy, noise_stddevs = noise_stddevs,
    training_strategy = SolverTraining(tstart, dt, tstop, solver_train), ad = ad
)

####################
# Evaluate network #
####################

# with fixed timesteps

eval_network(
    ds_path, chk_path, eval_path, solver_eval_fixed_timesteps; start = tstart, stop = tstop,
    dt = dt, saves = tstart:dt:tstop, mse_steps = collect(mse_steps), mps = message_steps,
    layer_size = layer_size, hidden_layers = hidden_layers, use_cuda = cuda
)

# with adaptive timesteps

eval_network(
    ds_path, chk_path, eval_path, solver_eval_adaptive_timesteps; start = tstart,
    stop = tstop, saves = tstart:dt:tstop, mse_steps = collect(mse_steps),
    mps = message_steps, layer_size = layer_size, hidden_layers = hidden_layers,
    use_cuda = cuda
)
