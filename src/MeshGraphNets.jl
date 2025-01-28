#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

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

export train_network, eval_network, der_minmax, data_meanstd

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
            e_norms = NormaliserOfflineMinMax(Float32(dataset.meta["edges"]["data_min"]),
                Float32(dataset.meta["edges"]["data_max"]))
        elseif haskey(dataset.meta["edges"], "data_mean") &&
               haskey(dataset.meta["edges"], "data_std")
            e_norms = NormaliserOfflineMeanStd(Float32(dataset.meta["edges"]["data_mean"]),
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
        if getfield(
            Base, Symbol(uppercasefirst(dataset.meta["features"][feature]["dtype"]))) ==
           Bool
            quantities += 1
            n_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
            if feature in dataset.meta["target_features"]
                o_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
            end
        elseif getfield(
            Base, Symbol(uppercasefirst(dataset.meta["features"][feature]["dtype"]))) ==
               Int32
            if haskey(dataset.meta["features"][feature], "onehot") &&
               dataset.meta["features"][feature]["onehot"]
                quantities += dataset.meta["features"][feature]["data_max"] -
                              dataset.meta["features"][feature]["data_min"] + 1
                if haskey(dataset.meta["features"][feature], "target_min") &&
                   haskey(dataset.meta["features"][feature], "target_max")
                    n_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0,
                        Float32(dataset.meta["features"][feature]["target_min"]),
                        Float32(dataset.meta["features"][feature]["target_max"]))
                    if feature in dataset.meta["target_features"]
                        o_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0,
                            Float32(dataset.meta["features"][feature]["target_min"]),
                            Float32(dataset.meta["features"][feature]["target_max"]))
                    end
                else
                    n_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
                    if feature in dataset.meta["target_features"]
                        o_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
                    end
                end
            else
                throw(ArgumentError("Int32 types that are not onehot types are not supported yet."))
            end
        else
            quantities += dataset.meta["features"][feature]["dim"]
            if haskey(dataset.meta["features"][feature], "data_min") &&
               haskey(dataset.meta["features"][feature], "data_max")
                if haskey(dataset.meta["features"][feature], "target_min") &&
                   haskey(dataset.meta["features"][feature], "target_max")
                    n_norms[feature] = NormaliserOfflineMinMax(
                        Float32(dataset.meta["features"][feature]["data_min"]),
                        Float32(dataset.meta["features"][feature]["data_max"]),
                        Float32(dataset.meta["features"][feature]["target_min"]),
                        Float32(dataset.meta["features"][feature]["target_max"]))
                    if feature in dataset.meta["target_features"]
                        if haskey(dataset.meta["features"][feature], "output_min") &&
                           haskey(dataset.meta["features"][feature], "output_max")
                            o_norms[feature] = NormaliserOfflineMinMax(
                                Float32(dataset.meta["features"][feature]["output_min"]),
                                Float32(dataset.meta["features"][feature]["output_max"]),
                                Float32(dataset.meta["features"][feature]["target_min"]),
                                Float32(dataset.meta["features"][feature]["target_max"]))
                        else
                            o_norms[feature] = NormaliserOnline(
                                dataset.meta["features"][feature]["dim"],
                                device; max_acc = Float32(args.max_norm_steps))
                        end
                    end
                else
                    n_norms[feature] = NormaliserOfflineMinMax(
                        Float32(dataset.meta["features"][feature]["data_min"]),
                        Float32(dataset.meta["features"][feature]["data_max"]))
                    if feature in dataset.meta["target_features"]
                        if haskey(dataset.meta["features"][feature], "output_min") &&
                           haskey(dataset.meta["features"][feature], "output_max")
                            o_norms[feature] = NormaliserOfflineMinMax(
                                Float32(dataset.meta["features"][feature]["output_min"]),
                                Float32(dataset.meta["features"][feature]["output_max"]))
                        else
                            o_norms[feature] = NormaliserOnline(
                                dataset.meta["features"][feature]["dim"],
                                device; max_acc = Float32(args.max_norm_steps))
                        end
                    end
                end
            elseif haskey(dataset.meta["features"][feature], "data_mean") &&
                   haskey(dataset.meta["features"][feature], "data_std")
                n_norms[feature] = NormaliserOfflineMeanStd(
                    Float32(dataset.meta["features"][feature]["data_mean"]),
                    Float32(dataset.meta["features"][feature]["data_std"]))
                if feature in dataset.meta["target_features"]
                    if haskey(dataset.meta["features"][feature], "output_min") &&
                       haskey(dataset.meta["features"][feature], "output_max")
                        o_norms[feature] = NormaliserOfflineMeanStd(
                            Float32(dataset.meta["features"][feature]["output_mean"]),
                            Float32(dataset.meta["features"][feature]["output_std"]))
                    else
                        o_norms[feature] = NormaliserOnline(
                            dataset.meta["features"][feature]["dim"],
                            device; max_acc = Float32(args.max_norm_steps))
                    end
                end
            else
                n_norms[feature] = NormaliserOnline(
                    dataset.meta["features"][feature]["dim"],
                    device; max_acc = Float32(args.max_norm_steps))
                if feature in dataset.meta["target_features"]
                    o_norms[feature] = NormaliserOnline(
                        dataset.meta["features"][feature]["dim"],
                        device; max_acc = Float32(args.max_norm_steps))
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
        device = gpu_device()
    else
        @info "Training on CPU..."
        device = cpu_device()
    end

    @info "Training with $(typeof(args.training_strategy))..."

    println("Loading training data...")
    ds_train = Dataset(:train, ds_path, args)
    ds_train.meta["device"] = device
    ds_train.meta["types_updated"] = args.types_updated
    ds_train.meta["types_noisy"] = args.types_noisy
    ds_train.meta["noise_stddevs"] = args.noise_stddevs
    ds_valid = Dataset(:valid, ds_path, args)
    ds_valid.meta["types_updated"] = args.types_updated
    ds_valid.meta["types_noisy"] = args.types_noisy
    ds_valid.meta["noise_stddevs"] = args.noise_stddevs
    ds_valid.meta["device"] = device
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

    mgn, opt_state, df_train, df_valid = load(
        quantities, typeof(dims) <: AbstractArray ? length(dims) : dims,
        e_norms, n_norms, o_norms, outputs, args.mps,
        args.layer_size, args.hidden_layers, opt, device, cp_path)

    if isnothing(opt_state)
        opt_state = Optimisers.setup(opt, mgn.ps)
    end
    Lux.trainmode(mgn.st)

    clear_log(1, false)
    @info "Model built!"
    print("Compiling code...")
    print("\u1b[1G")

    min_validation_loss = train_mgn!(
        mgn, opt_state, ds_train, ds_valid, df_train, df_valid, cp_path, args)

    return mgn, min_validation_loss
end

"""
    train_mgn!(mgn, opt_state, ds_train, ds_valid, df_train, df_valid, cp_path, args)

Initializes the network and performs the training loop.

## Arguments
- `mgn`: [GraphNetwork](@ref) that should be trained.
- `opt_state`: State of the optimiser.
- `ds_train`: Dataset containing the training data and metadata.
- `ds_valid`: Dataset containing the validation data and metadata.
- `df_train`: [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) DataFrame that stores the train losses at the checkpoints.
- `df_valid`: [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) DataFrame that stores the validation losses at the checkpoints (only improvements are saved).
- `cp_path`: Path where checkpoints are saved.
- `args`: Keyword arguments for configuring the training.

## Returns
- Minimum of validation loss (for hyperparameter tuning).
"""
function train_mgn!(mgn::GraphNetwork, opt_state, ds_train::Dataset, ds_valid::Dataset,
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
        dt = 1.0, barlen = 50, start = checkpoint, showspeed = true)

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

            for datapoint in 1:delta
                train_tuple = init_train_step(args.training_strategy,
                    (mgn, data, ds_train.meta, fields,
                        ds_train.meta["target_features"], data["node_type"],
                        data["edge_features"], data["senders"], data["receivers"],
                        datapoint, data["mask"], data["val_mask"]),
                    train_tuple_additional)

                gs, losses = train_step(args.training_strategy, train_tuple)

                tmp_loss += sum(losses)

                if step + datapoint > args.norm_steps
                    for i in eachindex(gs)
                        opt_state, ps = Optimisers.update(opt_state, mgn.ps, gs[i])
                        mgn.ps = ps
                    end
                    update!(pr, step + datapoint;
                        showvalues = [
                            (:train_step, "$(step + datapoint)/$(args.epochs*args.steps)"),
                            (:train_loss, sum(losses)),
                            (:checkpoint,
                                length(df_train.step) > 0 ? last(df_train.step) : 0),
                            (:data_interval, delta == 1 ? "1:end" : 1:delta),
                            (:min_validation_loss, min_validation_loss),
                            (:last_validation_loss, last_validation_loss)])
                    if !isnothing(args.wandb_logger)
                        Wandb.log(args.wandb_logger, Dict("train_loss" => sum(losses)))
                    end
                else
                    update!(pr, step + datapoint;
                        showvalues = [
                            (:step, "$(step + datapoint)/$(args.epochs*args.steps)"),
                            (:loss, "acc norm stats..."), (:checkpoint, 0)])
                end
            end

            cp_progress += delta
            step += delta
            tmp_loss /= delta

            avg_loss += tmp_loss
            tmp_loss = 0.0f0

            if step > args.norm_steps && cp_progress >= args.checkpoint
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
                    ve, g, p = validation_step(args.training_strategy,
                        (
                            mgn, data_valid, ds_valid.meta, delta, args.solver_valid,
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
                clear_log(9)

                if !isnothing(args.wandb_logger)
                    Wandb.log(args.wandb_logger,
                        Dict("validation_loss" => valid_error /
                                                  ds_valid.meta["n_trajectories"]))
                end

                if valid_error / ds_valid.meta["n_trajectories"] < min_validation_loss
                    save!(mgn, opt_state, df_train, df_valid, step,
                        valid_error / ds_valid.meta["n_trajectories"],
                        joinpath(cp_path, "valid"); is_training = false)
                    min_validation_loss = valid_error / ds_valid.meta["n_trajectories"]
                    cp_progress = args.checkpoint
                end
                last_validation_loss = valid_error / ds_valid.meta["n_trajectories"]
            end

            if cp_progress >= args.checkpoint
                save!(mgn, opt_state, df_train, df_valid, step,
                    avg_loss / Float32(step / delta), cp_path)
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

    mgn, _, _, _ = load(
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
    local errors = Dict{Tuple{Int, String}, Array{Float32, 2}}()
    local timesteps = Dict{Tuple{Int, String}, Array{Float32, 1}}()
    local cells = Dict{Tuple{Int, String}, Array{Int32, 3}}()

    test_loader = DataLoader(ds_test; batchsize = -1, buffer = false, parallel = true)

    for (ti, data) in enumerate(test_loader)
        fields = deleteat!(copy(ds_test.meta["feature_names"]),
            findall(x -> x == "node_type" || x == "mesh_pos" || x == "cells",
                ds_test.meta["feature_names"]))

        target_dict = Dict{String, Int32}()
        for tf in ds_test.meta["target_features"]
            target_dict[tf] = ds_test.meta["features"][tf]["dim"]
        end

        pr = ProgressUnknown(;
            desc = "Trajectory $ti/$(length(test_loader)): ", showspeed = true)

        sol_u, sol_t = rollout(
            solver, mgn, data, fields, ds_test.meta, ds_test.meta["target_features"],
            target_dict, data["node_type"], data["edge_features"], data["senders"],
            data["receivers"], data["val_mask"], data["inflow_mask"], start, stop, dt,
            saves, pr)

        prediction = cat(sol_u...; dims = 3)
        error = mean(
            (prediction -
             vcat([data[field][:, :, 1:length(saves)]
                   for field in ds_test.meta["target_features"]]...)) .^
            2;
            dims = 2)
        timesteps[(ti, "timesteps")] = sol_t

        println("MSE of state prediction:")
        for horizon in mse_steps
            err = mean(error[:, 1, findfirst(x -> x == horizon, saves)])
            cum_err = mean(error[:, 1, 1:findfirst(x -> x == horizon, saves)])
            println("  Trajectory $ti | mse t=$(horizon): $err | cum_mse t=$(horizon): $cum_err | cum_rmse t=$(horizon): $(sqrt(cum_err))")
        end

        traj_ops[(ti, "mesh_pos")] = cpu_device()(data["mesh_pos"])
        traj_ops[(ti, "gt")] = cpu_device()(vcat([data[field]
                                                  for field in ds_test.meta["target_features"]]...))
        traj_ops[(ti, "prediction")] = cpu_device()(prediction)
        errors[(ti, "error")] = cpu_device()(error[:, 1, :])
    end

    eval_path = joinpath(out_path,
        isnothing(solver) ? "derivative_training" : lowercase("$(nameof(typeof(solver)))"))
    mkpath(eval_path)
    h5open(joinpath(eval_path, "trajectories.h5"), "w") do f
        for i in 1:maximum(getfield.(keys(traj_ops), 1))
            create_group(f, string(i))
        end
        for (key, value) in traj_ops
            g = open_group(f, string(key[1]))
            sub_g = create_group(g, key[2])
            sub_g["data"] = reshape(value, length(value))
            sub_g["size"] = collect(size(value))
        end
        for (key, value) in errors
            g = open_group(f, string(key[1]))
            sub_g = create_group(g, key[2])
            sub_g["data"] = reshape(value, length(value))
            sub_g["size"] = collect(size(value))
        end
        for (key, value) in timesteps
            g = open_group(f, string(key[1]))
            sub_g = create_group(g, key[2])
            sub_g["data"] = reshape(value, length(value))
            sub_g["size"] = collect(size(value))
        end
        for (key, value) in cells
            g = open_group(f, string(key[1]))
            sub_g = create_group(g, key[2])
            sub_g["data"] = reshape(value, length(value))
            sub_g["size"] = collect(size(value))
        end
    end

    @info "Evaluation completed!"
end

end
