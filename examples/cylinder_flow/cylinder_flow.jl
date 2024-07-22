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
cuda = true
cp = 10000

########################
# Node type parameters #
########################

noise = [0.02f0]
types_updated = [0, 5]
types_noisy = [0]

########################
# Optimiser parameters #
########################

learning_rate = 1.0f-4
opt = Adam(learning_rate)

#########################
# Paths to data folders #
#########################

ds_path = "data/CylinderFlow/datasets"
chk_path = "data/CylinderFlow/chk"
eval_path = "data/CylinderFlow/eval"

#################
# Train network #
#################

# with DerivativeTraining

train_network(
    noise, opt, ds_path, chk_path; mps = message_steps, layer_size = layer_size,
    hidden_layers = hidden_layers, batchsize = batch,
    epochs = epo, steps = Int(ns), use_cuda = cuda, checkpoint = cp,
    norm_steps = 1000, types_updated = types_updated,
    types_noisy = types_noisy, training_strategy = DerivativeTraining(),
    solver_valid = Euler(), solver_valid_dt = 0.01f0
)

# with SolverTraining

train_network(
    noise, opt, ds_path, chk_path; mps = message_steps, layer_size = layer_size,
    hidden_layers = hidden_layers, batchsize = batch, epochs = epo,
    steps = Int(ns), use_cuda = cuda, checkpoint = 10, norm_steps = 1000,
    types_updated = types_updated, types_noisy = types_noisy,
    training_strategy = SolverTraining(
        0.0f0, 0.01f0, 5.99f0, Euler(); adaptive = false, tstops = 0.0f0:0.01f0:5.99f0)
)

####################
# Evaluate network #
####################

# with Euler (fixed timestep)

eval_network(
    ds_path, chk_path, eval_path, Euler(); start = 0.0f0,
    stop = 5.99f0, dt = 0.01f0, saves = 0.0f0:0.01f0:5.99f0,
    mse_steps = collect(0.0f0:1.0f0:5.99f0), mps = message_steps, layer_size = layer_size,
    hidden_layers = hidden_layers, use_cuda = cuda
)

# with Tsit5 (adaptive timestep)

eval_network(
    ds_path, chk_path, eval_path, Tsit5(); start = 0.0f0,
    stop = 5.99f0, saves = 0.0f0:0.01f0:5.99f0,
    mse_steps = collect(0.0f0:1.0f0:5.99f0), mps = message_steps, layer_size = layer_size,
    hidden_layers = hidden_layers, use_cuda = cuda
)
