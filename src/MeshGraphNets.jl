# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from the original MeshGraphNets software for this Julia project.
# Copyright (c) 2023 Julian Trommer
# Copyright (c) 2024 Leonard Heber
# Copyright (c) 2024 Luca Kahlenberg
# SPDX-License-Identifier: Apache-2.0
# See LICENSE-APACHE and NOTICE for details.

module MeshGraphNets

using GraphNetCore

using CUDA
using Lux, LuxCUDA
using MLUtils
using Optimisers
using Wandb
using Zygote

import OrdinaryDiffEq: OrdinaryDiffEqAlgorithm, Tsit5
import ProgressMeter: Progress
import SciMLBase: ODEProblem

import Base: @kwdef
import HDF5: h5open, create_group, open_group
import ProgressMeter: next!, update!, finish!
import SciMLBase: solve, remake
import Statistics: mean

include("utils.jl")
include("graph.jl")
include("solve.jl")
include("dataset.jl")

export SolverTraining, MultipleShooting, DerivativeTraining

export train_network, eval_network, data_minmax, data_meanstd

@kwdef mutable struct Args
    mps::Integer = 15
    layer_size::Integer = 128
    hidden_layers::Integer = 2
    batchsize::Integer = 1
    epochs::Integer = 1
    steps::Integer = 10e6
    checkpoint::Integer = 10000
    norm_steps::Integer = 1000
    max_norm_steps::Integer = 10.0f6
    types_inflow::Vector{Integer} = [4]
    types_updated::Vector{Integer} = [0, 5]
    types_noisy::Vector{Integer} = [0]
    noise_stddevs::Vector{Float32} = [0.0f0]
    training_strategy::TrainingStrategy = DerivativeTraining()
    use_cuda::Bool = true
    gpu_device::Union{Nothing, CuDevice} = CUDA.functional() ? CUDA.device() : nothing
    cell_idxs::Vector{Integer} = [0]
    use_valid::Bool = true
    solver_valid::OrdinaryDiffEqAlgorithm = Tsit5()
    solver_valid_dt::Union{Nothing, Float32} = nothing
    wandb_logger::Union{Nothing, Wandb.WandbLogger} = nothing
    reset_valid::Bool = false
    ad::Symbol = :Zygote
end

"""
    calc_norms(dataset, device)

Initializes the normalisers based on the given dataset and its metadata.

## Arguments
- `dataset`: Dataset on which the normalisers should be initialized on.
- `device`: Device where the normaliser should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).

## Returns
- Sum of each dimension of feature.
- Dictionary of each edge feature and its normaliser as key-value pair.
- Dictionary of each node feature and its normaliser as key-value pair.
- Dictionary of each output feature and its normaliser as key-value pair.
"""
function calc_norms(dataset, device, args::Args)
    quantities = 0
    n_norms = Dict{String, Union{NormaliserOffline, NormaliserOnline}}()
    o_norms = Dict{String, Union{NormaliserOffline, NormaliserOnline}}()

    if haskey(dataset.meta, "edges")
        if haskey(dataset.meta["edges"], "data_min") &&
           haskey(dataset.meta["edges"], "data_max")
            e_norms = NormaliserOfflineMinMax(
                Float32(dataset.meta["edges"]["data_min"]),
                Float32(dataset.meta["edges"]["data_max"]))
        elseif haskey(dataset.meta["edges"], "data_mean") &&
               haskey(dataset.meta["edges"], "data_std")
            e_norms = NormaliserOfflineMeanStd(
                Float32(dataset.meta["edges"]["data_mean"]),
                Float32(dataset.meta["edges"]["data_std"]))
        else
            e_norms = NormaliserOnline(
                typeof(dataset.meta["dims"]) <: AbstractArray ?
                length(dataset.meta["dims"]) + 1 : dataset.meta["dims"] + 1,
                device)
        end
    else
        e_norms = NormaliserOnline(
            typeof(dataset.meta["dims"]) <: AbstractArray ?
            length(dataset.meta["dims"]) + 1 : dataset.meta["dims"] + 1,
            device)
    end

    for feature in dataset.meta["feature_names"]
        if feature == "mesh_pos" || feature == "cells"
            continue
        end
        feature_meta = dataset.meta["features"][feature]

        if getfield(Base, Symbol(uppercasefirst(feature_meta["dtype"]))) == Bool
            quantities += 1
            n_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
            if feature in dataset.meta["target_features"]
                o_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
            end
        elseif getfield(Base, Symbol(uppercasefirst(feature_meta["dtype"]))) == Int32 &&
               haskey(feature_meta, "onehot") && feature_meta["onehot"]
            quantities += feature_meta["data_max"] - feature_meta["data_min"] + 1
            # check for node feature norm
            if haskey(feature_meta, "target_min") && haskey(feature_meta, "target_max")
                n_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0,
                    Float32(feature_meta["target_min"]),
                    Float32(feature_meta["target_max"]))
            else
                n_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
            end
            # check for output feature norm
            if feature in dataset.meta["target_features"]
                if haskey(feature_meta, "output_target_min") &&
                   haskey(feature_meta, "output_target_max")
                    o_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0,
                        Float32(feature_meta["output_target_min"]),
                        Float32(feature_meta["output_target_max"]))
                else
                    o_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
                end
            end
        else
            quantities += feature_meta["dim"]
            # check for node feature norm
            if haskey(feature_meta, "data_min") && haskey(feature_meta, "data_max")
                if haskey(feature_meta, "target_min") && haskey(feature_meta, "target_max")
                    n_norms[feature] = NormaliserOfflineMinMax(
                        Float32(feature_meta["data_min"]),
                        Float32(feature_meta["data_max"]),
                        Float32(feature_meta["target_min"]),
                        Float32(feature_meta["target_max"]))
                else
                    n_norms[feature] = NormaliserOfflineMinMax(
                        Float32(feature_meta["data_min"]),
                        Float32(feature_meta["data_max"]))
                end
            elseif haskey(feature_meta, "data_mean") && haskey(feature_meta, "data_std")
                n_norms[feature] = NormaliserOfflineMeanStd(
                    Float32(feature_meta["data_mean"]),
                    Float32(feature_meta["data_std"]))
            else
                n_norms[feature] = NormaliserOnline(
                    feature_meta["dim"], device; max_acc = Float32(args.max_norm_steps))
            end

            # check for output feature norm
            if feature in dataset.meta["target_features"]
                if haskey(feature_meta, "output_min") && haskey(feature_meta, "output_max")
                    if haskey(feature_meta, "output_target_min") &&
                       haskey(feature_meta, "output_target_max")
                        o_norms[feature] = NormaliserOfflineMinMax(
                            Float32(feature_meta["output_min"]),
                            Float32(feature_meta["output_max"]),
                            Float32(feature_meta["output_target_min"]),
                            Float32(feature_meta["output_target_max"]))
                    else
                        o_norms[feature] = NormaliserOfflineMinMax(
                            Float32(feature_meta["output_min"]),
                            Float32(feature_meta["output_max"]))
                    end
                elseif haskey(feature_meta, "output_mean") &&
                       haskey(feature_meta, "output_std")
                    o_norms[feature] = NormaliserOfflineMeanStd(
                        Float32(feature_meta["output_mean"]),
                        Float32(feature_meta["output_std"]))
                else
                    o_norms[feature] = NormaliserOnline(
                        feature_meta["dim"], device; max_acc = Float32(args.max_norm_steps))
                end
            end
        end
    end

    return quantities, e_norms, n_norms, o_norms
end

"""
    train_network(opt, ds_path, cp_path; kws...)

Starts the training process with the given configuration.

## Arguments
- `opt`: Optimiser that is used for training.
- `ds_path`: Path to the dataset folder.
- `cp_path`: Path where checkpoints are being saved to.
- `kws`: Keyword arguments that customize the training process.

## Keyword Arguments
- `mps = 15`: Number of message passing steps.
- `layer_size = 128`: Latent size of the hidden layers inside MLPs.
- `hidden_layers = 2`: Number of hidden layers inside MLPs.
- `batchsize = 1`: Size per batch *(not implemented yet)*.
- `epochs = 1`: Number of epochs.
- `steps = 10e6`: Number of training steps.
- `checkpoint = 10000`: Number of steps after which checkpoints are created.
- `norm_steps = 1000`: Number of steps before training (accumulate normalization stats).
- `max_norm_steps = 10f6`: Number of steps after which no more normalization stats are collected.
- `types_updated = [0, 5]`: Array containing node types which are updated after each step.
- `types_noisy = [0]`: Array containing node types which noise is added to.
- `noise_stddevs = [0.0f0]`: Array containing the standard deviation of noise that is added to the target features.
- `training_strategy = DerivativeTraining()`: Methods used for training. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/).
- `use_cuda = true`: Whether a GPU is used for training or not (if available). Currently only CUDA GPUs are supported.
- `gpu_device = CUDA.device()`: Current CUDA device (aka GPU). See *nvidia-smi* for reference.
- `cell_idxs = [0]`: Indices of cells that are plotted during validation (if enabled).
- `use_valid = true`: Whether the last checkpoint of validation should be used, last training checkpoint otherwise.
- `solver_valid = Tsit5()`: Which solver should be used for validation during training.
- `solver_valid_dt = nothing`: If set, the solver for validation will use fixed timesteps.
- `wandb_logger` = nothing: If set, a [Wandb](https://github.com/avik-pal/Wandb.jl) WandbLogger will be used for logging the training.
- `reset_valid = false`: If set, the previous minimal validation loss will be overwritten.

## Training Strategies
- `DerivativeTraining`
- `SolverTraining`
- `MultipleShooting`

See [CylinderFlow Example](https://una-auxme.github.io/MeshGraphNets.jl/dev/cylinder_flow) for reference.

## Returns
- Trained network as a [`GraphNetwork`](@ref) struct.
- Minimum of validation loss (for hyperparameter tuning).
"""
function train_network(opt, ds_path, cp_path; kws...)
    args = Args(; kws...)

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU..."
        CUDA.device!(args.gpu_device)
        CUDA.allowscalar(false)
        if args.ad == :Zygote
            @info "Using Zygote as AD framework..."
            device = gpu_device()
        elseif args.ad == :Enzyme
            throw(ArgumentError("Enzyme is not supported yet!"))
            # @info "Using Enzyme (+ Reactant) as AD framework..."
            # Reactant.set_default_backend("gpu")
            # device = reactant_device()
        else
            throw(ArgumentError("Invalid AD framework. Possible values are: [:Zygote, :Enzyme]"))
        end
    else
        @info "Training on CPU..."
        device = cpu_device()
    end

    @info "Training with $(typeof(args.training_strategy))..."

    println("Loading training data...")
    ds_train = Dataset(:train, ds_path, args)
    ds_train.meta["types_inflow"] = args.types_inflow
    ds_train.meta["types_updated"] = args.types_updated
    ds_train.meta["types_noisy"] = args.types_noisy
    ds_train.meta["noise_stddevs"] = args.noise_stddevs
    ds_train.meta["device"] = device
    ds_valid = Dataset(:valid, ds_path, args)
    ds_valid.meta["types_inflow"] = args.types_inflow
    ds_valid.meta["types_updated"] = args.types_updated
    ds_valid.meta["types_noisy"] = args.types_noisy
    ds_valid.meta["noise_stddevs"] = args.noise_stddevs
    ds_valid.meta["device"] = device
    ds_valid.meta["training_strategy"] = nothing
    clear_log(1, false)
    @info "Training data loaded!"
    Threads.nthreads() < 2 &&
        @warn "Julia is currently running on a single thread! Start Julia with more threads to speed up data loading."

    println("Building model...")

    quantities, e_norms, n_norms, o_norms = calc_norms(ds_train, device, args)

    dims = ds_train.meta["dims"]
    outputs = 0
    for tf in ds_train.meta["target_features"]
        outputs += ds_train.meta["features"][tf]["dim"]
    end

    mgn, train_state,
    df_train,
    df_valid = load(
        quantities, typeof(dims) <: AbstractArray ? length(dims) : dims,
        e_norms, n_norms, o_norms, outputs, args.mps,
        args.layer_size, args.hidden_layers, opt, device, cp_path)

    if isnothing(train_state)
        train_state = Lux.Training.TrainState(mgn.model, mgn.ps, mgn.st, opt)
    end
    Lux.trainmode(mgn.st)

    clear_log(1, false)
    @info "Model built!"
    print("Compiling code...")
    print("\u1b[1G")

    min_validation_loss = train_mgn!(
        mgn, train_state, ds_train, ds_valid, df_train, df_valid, cp_path, args)

    return mgn, min_validation_loss
end

"""
    train_mgn!(mgn, opt_state, ds_train, ds_valid, df_train, df_valid, cp_path, args)

Initializes the network and performs the training loop.

## Arguments
- `mgn`: [GraphNetwork](@ref) that should be trained.
- `train_state`: TrainingState.
- `ds_train`: Dataset containing the training data and metadata.
- `ds_valid`: Dataset containing the validation data and metadata.
- `df_train`: [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) DataFrame that stores the train losses at the checkpoints.
- `df_valid`: [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) DataFrame that stores the validation losses at the checkpoints (only improvements are saved).
- `cp_path`: Path where checkpoints are saved.
- `args`: Keyword arguments for configuring the training.

## Returns
- Minimum of validation loss (for hyperparameter tuning).
"""
function train_mgn!(mgn::GraphNetwork, train_state, ds_train::Dataset, ds_valid::Dataset,
        df_train, df_valid, cp_path, args::Args)
    checkpoint = length(df_train.step) > 0 ? last(df_train.step) : 0
    step = checkpoint
    cp_progress = 0
    if args.reset_valid
        min_validation_loss = Inf32
    else
        min_validation_loss = length(df_valid.loss) > 0 ? last(df_valid.loss) : Inf32
    end
    last_validation_loss = min_validation_loss

    pr = Progress(args.epochs * args.steps; desc = "Training progress: ",
        dt = 0.1, barlen = 50, start = checkpoint, showspeed = true)

    local tmp_loss = 0.0f0
    local avg_loss = 0.0f0
    fields = deleteat!(copy(ds_train.meta["feature_names"]),
        findall(x -> x == "node_type" || x == "mesh_pos" || x == "cells",
            ds_train.meta["feature_names"]))

    train_tuple_additional = prepare_training(args.training_strategy)

    train_loader = DataLoader(
        ds_train; batchsize = -1, buffer = false, parallel = true, shuffle = true)
    valid_loader = DataLoader(ds_valid; batchsize = -1, buffer = false, parallel = true)

    while step < args.steps
        for data in train_loader
            delta = get_delta(args.training_strategy, data["trajectory_length"])

            for (data_idx, datapoint) in enumerate(delta)
                train_tuple = init_train_step(args.training_strategy,
                    (mgn, data, ds_train.meta, fields,
                        ds_train.meta["target_features"], data["node_type"],
                        data["edge_features"], data["senders"], data["receivers"],
                        datapoint, data["mask"], data["val_mask"]),
                    train_tuple_additional)

                if step + data_idx > args.norm_steps
                    gs, losses = train_step(args.training_strategy, train_tuple)
                    Lux.Training.apply_gradients!(train_state, gs[1])
                    mgn.ps = train_state.parameters
                    tmp_loss += sum(losses)

                    update!(pr, step + data_idx;
                        showvalues = [
                            (:train_step, "$(step + data_idx)/$(args.epochs*args.steps)"),
                            (:train_loss, sum(losses)),
                            (:checkpoint,
                                length(df_train.step) > 0 ? last(df_train.step) : 0),
                            (:data_interval,
                                delta isa Vector ?
                                "$datapoint : [$(delta[1]),...,$(delta[end])]" : delta),
                            (:min_validation_loss, min_validation_loss),
                            (:last_validation_loss, last_validation_loss)])
                    if !isnothing(args.wandb_logger)
                        Wandb.log(args.wandb_logger, Dict("train_loss" => sum(losses)))
                    end
                else
                    gs, losses = train_step(args.training_strategy, train_tuple)
                    update!(pr, step + data_idx;
                        showvalues = [
                            (:step, "$(step + data_idx)/$(args.epochs*args.steps)"),
                            (:loss, "acc norm stats..."), (:checkpoint, 0)])
                end
            end

            cp_progress += length(delta)
            step += length(delta)
            tmp_loss /= length(delta)

            if step > args.norm_steps && cp_progress >= args.checkpoint
                push!(df_train, [step, avg_loss / Float32(step / length(delta))])

                traj_idx = 1
                valid_error = 0.0f0
                pr_valid = Progress(ds_valid.meta["n_trajectories"];
                    desc = "Validation progress: ", barlen = 50)
                print("\n\n\n\n\n\n\n")

                for data_valid in valid_loader
                    print("\n\n\n")
                    pr_solver = ProgressUnknown(;
                        desc = "Trajectory $(traj_idx)/$(length(valid_loader)): ",
                        showspeed = true)
                    ve = validation_step(args.training_strategy,
                        (
                            mgn, data_valid,
                            ds_valid.meta,
                            length(get_delta(args.training_strategy, data_valid["trajectory_length"])),
                            args.solver_valid,
                            args.solver_valid_dt, fields, data_valid["node_type"],
                            data_valid["edge_features"], data_valid["senders"],
                            data_valid["receivers"], data_valid["mask"],
                            data_valid["val_mask"], data_valid["inflow_mask"], pr_solver
                        ))

                    valid_error += ve

                    clear_log(3)
                    next!(pr_valid;
                        showvalues = [
                            (:trajectory, "$traj_idx/$(ds_valid.meta["n_trajectories"])"),
                            (:valid_loss, "$(valid_error / traj_idx)")])
                    traj_idx += 1
                end
                clear_log(6)

                if !isnothing(args.wandb_logger)
                    Wandb.log(args.wandb_logger,
                        Dict("validation_loss" =>
                            valid_error /
                            ds_valid.meta["n_trajectories"]))
                end

                if valid_error / ds_valid.meta["n_trajectories"] < min_validation_loss
                    push!(df_valid, [step, valid_error / ds_valid.meta["n_trajectories"]])
                    save!(mgn, train_state, df_train, df_valid,
                        step, joinpath(cp_path, "valid"))
                    min_validation_loss = valid_error / ds_valid.meta["n_trajectories"]
                end
                last_validation_loss = valid_error / ds_valid.meta["n_trajectories"]

                save!(mgn, train_state, df_train, df_valid, step, cp_path)
                avg_loss = 0.0f0
                cp_progress = 0
            end
        end
    end

    return min_validation_loss
end

"""
    eval_network(ds_path, cp_path, out_path, solver; start, stop, dt, saves, mse_steps, kws...)

Starts the evaluation process with the given configuration.

## Arguments
- `ds_path`: Path to the dataset folder.
- `cp_path`: Path where checkpoints are being saved to.
- `out_path`: Path where the result is being saved to.
- `solver`: Solver that is used for evaluating the system.
- `start`: Start time of the simulation.
- `stop`: Stop time of the simulation.
- `dt = nothing`: If provided, changes the solver to use fixed step sizes.
- `saves`: Time steps where the solution is saved at.
- `mse_steps`: Time steps where the relative error is printed at.
- `kws`: Keyword arguments that customize the training process. **The configuration of the system has to be the same as during training**.

## Keyword Arguments
- `mps = 15`: Number of message passing steps.
- `layer_size = 128`: Latent size of the hidden layers inside MLPs.
- `hidden_layers = 2`: Number of hidden layers inside MLPs.
- `types_updated = [0, 5]`: Array containing node types which are updated after each step.
- `use_cuda = true`: Whether a GPU is used for training or not (if available). Currently only CUDA GPUs are supported.
- `gpu_device = CUDA.device()`: Current CUDA device (aka GPU). See *nvidia-smi* for reference.
- `use_valid = true`: Whether the last checkpoint with the minimal validation loss should be used.
"""
function eval_network(ds_path, cp_path::String, out_path::String, solver = nothing;
        start, stop, dt = nothing, saves, mse_steps, kws...)
    args = Args(; kws...)

    if CUDA.functional() && args.use_cuda
        @info "Evaluating on CUDA GPU..."
        CUDA.device!(args.gpu_device)
        CUDA.allowscalar(false)
        device = gpu_device()
    else
        @info "Evaluating on CPU..."
        device = cpu_device()
    end

    println("Loading evaluation data...")
    ds_test = Dataset(:test, ds_path, args)
    ds_test.meta["types_inflow"] = args.types_inflow
    ds_test.meta["types_updated"] = args.types_updated
    ds_test.meta["device"] = device
    ds_test.meta["training_strategy"] = nothing
    # dataset = load_dataset(ds_path, false)

    clear_log(1, false)
    @info "Evaluation data loaded!"
    Threads.nthreads() < 2 &&
        @warn "Julia is currently running on a single thread! Start Julia with more threads to speed up data loading."

    println("Building model...")

    quantities, e_norms, n_norms, o_norms = calc_norms(ds_test, device, args)

    dims = ds_test.meta["dims"]
    outputs = 0
    for tf in ds_test.meta["target_features"]
        outputs += ds_test.meta["features"][tf]["dim"]
    end

    mgn, _,
    _,
    _ = load(
        quantities, typeof(dims) <: AbstractArray ? length(dims) : dims, e_norms,
        n_norms, o_norms, outputs, args.mps, args.layer_size, args.hidden_layers,
        nothing, device, args.use_valid ? joinpath(cp_path, "valid") : cp_path)

    Lux.testmode(mgn.st)

    clear_log(1, false)
    @info "Model built!"

    eval_network!(
        solver, mgn, ds_test, out_path, start, stop, dt, saves, mse_steps)
end

"""
    eval_network!(solver, mgn, ds_test, out_path, start, stop, dt, saves, mse_steps)

Initializes the network, performs evaluation for the given number of rollouts and saves the results.

## Arguments
- `solver`: Solver that is used for evaluating the system.
- `mgn`: [GraphNetwork](@ref) that should be evaluated.
- `ds_test`: Dataset containing the test data and metadata.
- `out_path`: Path where the evaluated trajectories are saved at.
- `start`: Start time of the simulation.
- `stop`: End time of the simulation.
- `dt`: If provided, changes the solver to use fixed step sizes.
- `saves`: Time steps where the solution is saved at.
- `mse_steps`: Time steps where the relative error is printed at.
"""
function eval_network!(solver, mgn::GraphNetwork, ds_test::Dataset, out_path, start, stop,
        dt, saves, mse_steps)
    local traj_ops = Dict{Tuple{Int, String}, Array{Float32, 3}}()
    local errors = Dict{Tuple{Int, String}, Array{Float32, 1}}()
    local timesteps = Dict{Tuple{Int, String}, Array{Float32, 1}}()
    local edges = Dict{Tuple{Int, String}, Array{Int32, 2}}()

    test_loader = DataLoader(ds_test; batchsize = -1, buffer = false, parallel = true)

    for (ti, data) in enumerate(test_loader)
        fields = deleteat!(copy(ds_test.meta["feature_names"]),
            findall(x -> x == "node_type" || x == "mesh_pos" || x == "cells",
                ds_test.meta["feature_names"]))

        target_dict = Dict{String, Int32}()
        for tf in ds_test.meta["target_features"]
            target_dict[tf] = ds_test.meta["features"][tf]["dim"]
        end

        gt = vcat([data[tf] for tf in ds_test.meta["target_features"]]...)[
            :, :, 1:length(saves)]

        pr = ProgressUnknown(;
            desc = "Trajectory $ti/$(length(test_loader)): ", showspeed = true)

        sol_u,
        sol_t = rollout(
            solver, mgn, data, fields, ds_test.meta, ds_test.meta["target_features"],
            target_dict, data["node_type"], data["edge_features"], data["senders"],
            data["receivers"], data["val_mask"], data["inflow_mask"], start, stop, dt,
            saves, pr)

        prediction = cat(sol_u...; dims = 3)
        error = cpu_device()(mean(abs2, prediction - gt; dims = (1, 2))[1, 1, :])
        timesteps[(ti, "timesteps")] = sol_t

        println("MSE of state prediction:")
        for horizon in mse_steps
            err = error[findfirst(x -> x == horizon, saves)]
            cum_err = mean(error[1:findfirst(x -> x == horizon, saves)])
            println("  Trajectory $ti | mse t=$(horizon): $err | cum_mse t=$(horizon): $cum_err | cum_rmse t=$(horizon): $(sqrt(cum_err))")
        end

        traj_ops[(ti, "mesh_pos")] = cpu_device()(data["mesh_pos"])
        traj_ops[(ti,
            "gt")] = cpu_device()(vcat([data[field][:, :, 1:size(prediction, 3)]
                                        for field in ds_test.meta["target_features"]]...))
        traj_ops[(ti, "prediction")] = cpu_device()(prediction)
        errors[(ti, "error")] = error
        edges[(ti,
            "edges")] = cpu_device()(permutedims(hcat(
            data["senders"], data["receivers"])))
    end

    eval_path = joinpath(out_path,
        isnothing(solver) ? "derivative_training" : lowercase("$(nameof(typeof(solver)))"))
    mkpath(eval_path)
    jldopen(joinpath(eval_path, "trajectories.jld2"), "w") do f
        for (key, value) in traj_ops
            f["trajectory_$(key[1])/$(key[2])"] = value
        end
        for (key, value) in errors
            f["trajectory_$(key[1])/$(key[2])"] = value
        end
        for (key, value) in timesteps
            f["trajectory_$(key[1])/$(key[2])"] = value
        end
        for (key, value) in edges
            f["trajectory_$(key[1])/$(key[2])"] = value
        end
    end

    @info "Evaluation completed!"
end

end
