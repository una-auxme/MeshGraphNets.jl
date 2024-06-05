#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module MeshGraphNets

using GraphNetCore

using CUDA
using Lux, LuxCUDA
using Optimisers
using Wandb
using Zygote

import DifferentialEquations: ODEProblem, OrdinaryDiffEqAlgorithm, Tsit5
import ProgressMeter: Progress

import Base: @kwdef
import DifferentialEquations: solve, remake
import HDF5: h5open, create_group, open_group
import ProgressMeter: next!, update!, finish!
import Statistics: mean

include("utils.jl")
include("graph.jl")
include("solve.jl")
include("dataset.jl")

export SingleShooting, MultipleShooting, RandomCollocation, Collocation

export train_network, eval_network, der_minmax

@kwdef mutable struct Args
    mps::Integer = 15
    layer_size::Integer = 128
    hidden_layers::Integer = 2
    batchsize::Integer = 1
    epochs::Integer = 1
    steps::Integer = 10e6
    checkpoint::Integer = 10000
    norm_steps::Integer = 1000
    types_updated::Vector{Integer} = [0, 5]
    types_noisy::Vector{Integer} = [0]
    training_strategy::TrainingStrategy = Collocation()
    use_cuda::Bool = true
    gpu_idx::Integer = CUDA.deviceid()
    cell_idxs::Vector{Integer} = [0]
    num_rollouts::Integer = 10
    use_valid::Bool = true
    solver_valid::OrdinaryDiffEqAlgorithm = Tsit5()
    solver_valid_dt::Union{Nothing, Float32} = nothing
    wandb_logger::Union{Nothing, Wandb.WandbLogger} = nothing
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
        e_norms = NormaliserOffline(Float32(dataset.meta["edges"]["data_min"]), Float32(dataset.meta["edges"]["data_max"]))
    else
        e_norms = NormaliserOnline(length(dataset.meta["dims"]) + 1, device)
    end

    for feature in dataset.meta["feature_names"]
        if feature == "mesh_pos" || feature == "cells"
            continue
        end
        if getfield(Base, Symbol(uppercasefirst(dataset.meta["features"][feature]["dtype"]))) == Bool
            quantities += 1
            n_norms[feature] = NormaliserOffline(0.0f0, 1.0f0)
            if feature in dataset.meta["target_features"]
                o_norms[feature] = NormaliserOffline(0.0f0, 1.0f0)
            end
        elseif getfield(Base, Symbol(uppercasefirst(dataset.meta["features"][feature]["dtype"]))) == Int32
            if haskey(dataset.meta["features"][feature], "onehot") && dataset.meta["features"][feature]["onehot"]
                quantities += dataset.meta["features"][feature]["data_max"] - dataset.meta["features"][feature]["data_min"] + 1
                if haskey(dataset.meta["features"][feature], "target_min") && haskey(dataset.meta["features"][feature], "target_max")
                    n_norms[feature] = NormaliserOffline(0.0f0, 1.0f0, Float32(dataset.meta["features"][feature]["target_min"]), Float32(dataset.meta["features"][feature]["target_max"]))
                    if feature in dataset.meta["target_features"]
                        o_norms[feature] = NormaliserOffline(0.0f0, 1.0f0, Float32(dataset.meta["features"][feature]["target_min"]), Float32(dataset.meta["features"][feature]["target_max"]))
                    end
                else
                    n_norms[feature] = NormaliserOffline(0.0f0, 1.0f0)
                    if feature in dataset.meta["target_features"]
                        o_norms[feature] = NormaliserOffline(0.0f0, 1.0f0)
                    end
                end
            else
                throw(ErrorException("Int32 types that are not onehot types are not supported yet."))
            end
        else
            quantities += dataset.meta["features"][feature]["dim"]
            if haskey(dataset.meta["features"][feature], "data_min") && haskey(dataset.meta["features"][feature], "data_max")
                if haskey(dataset.meta["features"][feature], "target_min") && haskey(dataset.meta["features"][feature], "target_max")
                    n_norms[feature] = NormaliserOffline(Float32(dataset.meta["features"][feature]["data_min"]), Float32(dataset.meta["features"][feature]["data_max"]), Float32(dataset.meta["features"][feature]["target_min"]), Float32(dataset.meta["features"][feature]["target_max"]))
                    if feature in dataset.meta["target_features"]
                        if haskey(dataset.meta["features"][feature], "output_min") && haskey(dataset.meta["features"][feature], "output_max")
                            o_norms[feature] = NormaliserOffline(Float32(dataset.meta["features"][feature]["output_min"]), Float32(dataset.meta["features"][feature]["output_max"]), Float32(dataset.meta["features"][feature]["target_min"]), Float32(dataset.meta["features"][feature]["target_max"]))
                        else
                            o_norms[feature] = NormaliserOnline(dataset.meta["features"][feature]["dim"], device; max_acc = Float32(args.norm_steps))
                        end
                    end
                else
                    n_norms[feature] = NormaliserOffline(Float32(dataset.meta["features"][feature]["data_min"]), Float32(dataset.meta["features"][feature]["data_max"]))
                    if feature in dataset.meta["target_features"]
                        if haskey(dataset.meta["features"][feature], "output_min") && haskey(dataset.meta["features"][feature], "output_max")
                            o_norms[feature] = NormaliserOffline(Float32(dataset.meta["features"][feature]["output_min"]), Float32(dataset.meta["features"][feature]["output_max"]))
                        else
                            o_norms[feature] = NormaliserOnline(dataset.meta["features"][feature]["dim"], device; max_acc = Float32(args.norm_steps))
                        end
                    end
                end
            else
                n_norms[feature] = NormaliserOnline(dataset.meta["features"][feature]["dim"], device; max_acc = Float32(args.norm_steps))
                if feature in dataset.meta["target_features"]
                    o_norms[feature] = NormaliserOnline(dataset.meta["features"][feature]["dim"], device; max_acc = Float32(args.norm_steps))
                end
            end
        end
    end

    return quantities, e_norms, n_norms, o_norms
end

"""
    train_network(noise_stddevs, opt, ds_path, cp_path; kws...)

Starts the training process with the given configuration.

## Arguments
- `noise_stddevs`: Array containing the standard deviations of the noise that is added to the specified node types, where the length is either one if broadcasted or equal to the length of features.
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
- `types_updated = [0, 5]`: Array containing node types which are updated after each step.
- `types_noisy = [0]`: Array containing node types which noise is added to.
- `training_strategy = Collocation()`: Methods used for training. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/).
- `use_cuda = true`: Whether a GPU is used for training or not (if available). Currently only CUDA GPUs are supported.
- `gpu_idx = CUDA.deviceid()`: Index of GPU. See *nvidia-smi* for reference.
- `cell_idxs = [0]`: Indices of cells that are plotted during validation (if enabled).
- `solver_valid = Tsit5()`: Which solver should be used for validation during training.
- `solver_valid_dt = nothing`: If set, the solver for validation will use fixed timesteps.
- `wandb_logger` = nothing: If set, a [Wandb](https://github.com/avik-pal/Wandb.jl) WandbLogger will be used for logging the training.

## Training Strategies
- `Collocation`
- `RandomCollocation`
- `SingleShooting`
- `MultipleShooting`

See [CylinderFlow Example](https://una-auxme.github.io/MeshGraphNets.jl/dev/cylinder_flow) for reference.

## Returns
- Trained network as a [`GraphNetwork`](@ref) struct.
"""
function train_network(noise_stddevs, opt, ds_path, cp_path; kws...)
    args = Args(;kws...)

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU..."
        CUDA.device!(args.gpu_idx)
        CUDA.allowscalar(false)
        device = gpu_device()
    else
        @info "Training on CPU..."
        device = cpu_device()
    end

    @info "Training with $(typeof(args.training_strategy))..."

    println("Loading training data...")
    dataset = load_dataset(ds_path, true)
    clear_log(1, false)
    @info "Training data loaded!"

    println("Building model...")

    quantities, e_norms, n_norms, o_norms = calc_norms(dataset, device, args)

    dims = dataset.meta["dims"]
    outputs = 0
    for tf in dataset.meta["target_features"]
        outputs += dataset.meta["features"][tf]["dim"]
    end

    mgn, opt_state, df_train, df_valid = load(quantities, typeof(dims) <: AbstractArray ? length(dims) : dims, e_norms, n_norms, o_norms, outputs, args.mps, args.layer_size, args.hidden_layers, opt, device, cp_path)

    if isnothing(opt_state)
        opt_state = Optimisers.setup(opt, mgn.ps)
    end
    Lux.trainmode(mgn.st)

    
    clear_log(1, false)
    @info "Model built!"

    print("Compiling code...")
    print("\u1b[1G")

    train_mgn!(mgn, opt_state, dataset, noise_stddevs, df_train, df_valid, device, cp_path, args)

    return mgn
end

"""
    train_mgn!(mgn, opt_state, dataset, noise, df_train, df_valid, device, cp_path, args)

Initializes the network and performs the training loop.

## Arguments
- `mgn`: [GraphNetwork](@ref) that should be trained.
- `opt_state`: State of the optimiser.
- `dataset`: Dataset containing the training, validation data and metadata.
- `noise`: Noise that is added to the node types specified in `args`.
- `df_train`: [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) DataFrame that stores the train losses at the checkpoints.
- `df_valid`: [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) DataFrame that stores the validation losses at the checkpoints (only improvements are saved).
- `device`: Device where the normaliser should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).
- `cp_path`: Path where checkpoints are saved.
- `args`: Keyword arguments for configuring the training.
"""
function train_mgn!(mgn::GraphNetwork, opt_state, dataset::Dataset, noise, df_train, df_valid, device::Function, cp_path, args::Args)
    checkpoint = length(df_train.step) > 0 ? last(df_train.step) : 0
    step = checkpoint
    cp_progress = 0
    min_validation_loss = length(df_valid.loss) > 0 ? last(df_valid.loss) : Inf32
    last_validation_loss = min_validation_loss

    if isnothing(args.wandb_logger)
        pr = Progress(args.epochs*args.steps; desc = "Training progress: ", dt=1.0, barlen=50, start=checkpoint, showspeed=true)
        update!(pr)
    else
        pr = nothing
    end

    local tmp_loss = 0.0f0
    local avg_loss = 0.0f0
    fields = deleteat!(copy(dataset.meta["feature_names"]), findall(x -> x == "node_type" || x == "mesh_pos" || x == "cells", dataset.meta["feature_names"]))

    delta = get_delta(args.training_strategy, dataset.meta["trajectory_length"])

    train_tuple_additional = prepare_training(args.training_strategy)

    for _ in checkpoint:delta:args.steps*args.epochs
        data, meta = next_trajectory!(dataset, device; types_noisy = args.types_noisy, noise_stddevs = noise, ts = args.training_strategy)
        
        mask = Int32.(findall(x -> x in args.types_updated, data["node_type"][1, :, 1])) |> device
        
        val_mask = Float32.(map(x -> x in args.types_updated, data["node_type"][:, :, 1]))
        val_mask = repeat(val_mask, sum(size(data[field], 1) for field in meta["target_features"]), 1) |> device

        node_type, senders, receivers, edge_features = create_base_graph(data, meta["features"]["node_type"]["data_max"], meta["features"]["node_type"]["data_min"], device)

        for datapoint in 1:delta
            train_tuple = init_train_step(args.training_strategy, (mgn, data, meta, fields, meta["target_features"], node_type, edge_features, senders, receivers, datapoint, mask, val_mask), train_tuple_additional)
            
            gs, losses = train_step(args.training_strategy, train_tuple)
                
            tmp_loss += sum(losses)

            if step + datapoint > args.norm_steps
                for i in eachindex(gs)
                    opt_state, ps = Optimisers.update(opt_state, mgn.ps, gs[i])
                    mgn.ps = ps
                end
                if isnothing(args.wandb_logger)
                    next!(pr, showvalues=[(:train_step,"$(step + datapoint)/$(args.epochs*args.steps)"), (:train_loss, sum(losses)), (:checkpoint, length(df_train.step) > 0 ? last(df_train.step) : 0), (:data_interval, delta == 1 ? "1:end" : 1:delta), (:min_validation_loss, min_validation_loss), (:last_validation_loss, last_validation_loss)])
                else
                    Wandb.log(args.wandb_logger, Dict("train_loss" => sum(losses)))
                end
            else
                if isnothing(args.wandb_logger)
                    next!(pr, showvalues=[(:step,"$(step + datapoint)/$(args.epochs*args.steps)"), (:loss,"acc norm stats..."), (:checkpoint, 0)])
                end
            end
        end

        cp_progress += delta
        step += delta
        tmp_loss /= delta

        avg_loss += tmp_loss
        tmp_loss = 0.0f0
        
        if step > args.norm_steps && cp_progress >= args.checkpoint
            valid_error = 0.0f0
            gt = nothing
            prediction = nothing
            pr_valid = Progress(meta["n_trajectories_valid"]; desc = "Validation progress: ", barlen = 50)

            for i in 1:meta["n_trajectories_valid"]
                data_valid, meta_valid = next_trajectory!(dataset, device; types_noisy = args.types_noisy, is_training = false)
                
                node_type_valid, senders_valid, receivers_valid, edge_features_valid = create_base_graph(data_valid, meta["features"]["node_type"]["data_max"], meta["features"]["node_type"]["data_min"], device)
                val_mask_valid = Float32.(map(x -> x in args.types_updated, data_valid["node_type"][:, :, 1]))
                val_mask_valid = repeat(val_mask_valid, sum(size(data_valid[field], 1) for field in meta_valid["target_features"]), 1) |> device

                inflow_mask_valid = repeat(data["node_type"][:, :, 1] .== 1, sum(size(data[field], 1) for field in meta["target_features"]), 1) |> device

                ve, g, p = validation_step(args.training_strategy, (
                    mgn, data_valid, meta_valid, delta, args.solver_valid, args.solver_valid_dt, fields, node_type_valid,
                    edge_features_valid, senders_valid, receivers_valid, mask, val_mask_valid, inflow_mask_valid, data
                ))
                
                valid_error += ve
                if i == 1
                    gt = g
                    prediction = p
                end
                
                next!(pr_valid, showvalues = [(:trajectory, "$i/$(meta["n_trajectories_valid"])"), (:valid_loss, "$((valid_error + ve) / i)")])
            end

            if !isnothing(args.wandb_logger)
                Wandb.log(args.wandb_logger, Dict("validation_loss" => valid_error / meta["n_trajectories_valid"]))
            end

            if valid_error / meta["n_trajectories_valid"] < min_validation_loss
                save!(mgn, opt_state, df_train, df_valid, step, valid_error / meta["n_trajectories_valid"], joinpath(cp_path, "valid"); is_training = false)
                min_validation_loss = valid_error / meta["n_trajectories_valid"]
                cp_progress = args.checkpoint
            end
            last_validation_loss = valid_error / meta["n_trajectories_valid"]
        end

        if cp_progress >= args.checkpoint
            save!(mgn, opt_state, df_train, df_valid, step, avg_loss / Float32(step / delta), cp_path)
            avg_loss = 0.0f0
            cp_progress = 0
        end
    end
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
- `gpu_idx = CUDA.deviceid()`: Index of GPU. See *nvidia-smi* for reference.
- `num_rollouts = 10`: Number of trajectories that are simulated (from the test dataset).
- `use_valid = true`: Whether the last checkpoint with the minimal validation loss should be used.
"""
function eval_network(ds_path, cp_path::String, out_path::String, solver = nothing; start, stop, dt = nothing, saves, mse_steps, kws...)
    args = Args(;kws...)

    if CUDA.functional() && args.use_cuda
        @info "Evaluating on CUDA GPU..."
        CUDA.device!(args.gpu_idx)
        CUDA.allowscalar(false)
        device = gpu_device()
    else
        @info "Evaluating on CPU..."
        device = cpu_device()
    end

    println("Loading evaluation data...")
    dataset = load_dataset(ds_path, false)

    clear_log(1, false)
    @info "Evaluation data loaded!"


    println("Building model...")

    quantities, e_norms, n_norms, o_norms = calc_norms(dataset, device, args)

    dims = dataset.meta["dims"]
    outputs = 0
    for tf in dataset.meta["target_features"]
        outputs += dataset.meta["features"][tf]["dim"]
    end

    mgn, _, _, _ = load(quantities, typeof(dims) <: AbstractArray ? length(dims) : dims, e_norms, n_norms, o_norms, outputs, args.mps, args.layer_size, args.hidden_layers, nothing, device, args.use_valid ? joinpath(cp_path, "valid") : cp_path)
    Lux.testmode(mgn.st)

    clear_log(1, false)
    @info "Model built!"

    eval_network!(solver, mgn, dataset, device, out_path, start, stop, dt, saves, mse_steps, args)
end

"""
    eval_network!(solver, mgn, dataset, device, out_path, start, stop, dt, saves, mse_steps, args)

Initializes the network, performs evaluation for the given number of rollouts and saves the results.

## Arguments
- `solver`: Solver that is used for evaluating the system.
- `mgn`: [GraphNetwork](@ref) that should be evaluated.
- `dataset`: Dataset containing the test data and metadata.
- `device`: Device where the normaliser should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).
- `out_path`: Path where the evaluated trajectories are saved at.
- `start`: Start time of the simulation.
- `stop`: End time of the simulation.
- `dt`: If provided, changes the solver to use fixed step sizes.
- `saves`: Time steps where the solution is saved at.
- `mse_steps`: Time steps where the relative error is printed at.
- `args`: Keyword arguments for configuring the evaluation.
"""
function eval_network!(solver, mgn::GraphNetwork, dataset::Dataset, device::Function, out_path, start, stop, dt, saves, mse_steps, args::Args)
    local traj_ops = Dict{Tuple{Int, String}, Array{Float32, 3}}()
    local errors = Dict{Tuple{Int, String}, Array{Float32, 2}}()
    local timesteps = Dict{Tuple{Int, String}, Array{Float32, 1}}()
    local cells = Dict{Tuple{Int, String}, Array{Int32, 3}}()

    for ti in 1:args.num_rollouts
        println("Rollout trajectory $ti...")
        data, meta = next_trajectory!(dataset, device; types_noisy = args.types_noisy)

        initial_state = Dict{String, AbstractArray}(
            [typeof(v) <: AbstractArray ? (k, v[:, :, 1]) : (k, v) for (k,v) in data]
        )
        for k in keys(initial_state)
            if endswith(k, ".ev")
                delete!(initial_state, k)
            end
        end

        val_mask = Float32.(map(x -> x in args.types_updated, data["node_type"][:, :, 1]))
        val_mask = repeat(val_mask, sum(size(data[field], 1) for field in meta["target_features"]), 1) |> device

        inflow_mask = repeat(data["node_type"][:, :, 1] .== 1, sum(size(data[field], 1) for field in meta["target_features"]), 1) |> device

        node_type, senders, receivers, edge_features = create_base_graph(data, meta["features"]["node_type"]["data_max"], meta["features"]["node_type"]["data_min"], device)

        fields = deleteat!(copy(dataset.meta["feature_names"]), findall(x -> x == "node_type" || x == "mesh_pos" || x == "cells", dataset.meta["feature_names"]))

        target_dict = Dict{String, Int32}()
        for tf in meta["target_features"]
            target_dict[tf] = meta["features"][tf]["dim"]
        end

        sol_u, sol_t = rollout(solver, mgn, initial_state, fields, meta, dataset.meta["target_features"], target_dict, node_type, edge_features, senders, receivers, val_mask, inflow_mask, data, start, stop, dt, saves)
        
        prediction = cat(sol_u..., dims = 3)
        error = mean((prediction - vcat([data[field][:, :, 1:length(saves)] for field in meta["target_features"]]...)) .^ 2; dims = 2)
        timesteps[(ti, "timesteps")] = sol_t

        clear_log(1)
        @info "Rollout trajectory $ti completed!"

        println("MSE of state prediction:")
        for horizon in mse_steps
            err = mean(error[:, 1, findfirst(x -> x == horizon, saves)])
            cum_err = mean(error[:, 1, 1:findfirst(x -> x == horizon, saves)])
            println("  Trajectory $ti | mse t=$(horizon): $err | cum_mse t=$(horizon): $cum_err | cum_rmse t=$(horizon): $(sqrt(cum_err))")
        end

        traj_ops[(ti, "mesh_pos")] = cpu_device()(data["mesh_pos"])
        traj_ops[(ti, "gt")] = cpu_device()(vcat([data[field] for field in meta["target_features"]]...))
        traj_ops[(ti, "prediction")] = cpu_device()(prediction)
        errors[(ti, "error")] = cpu_device()(error[:, 1, :])
    end
    
    eval_path = joinpath(out_path, isnothing(solver) ? "collocation" : lowercase("$(nameof(typeof(solver)))"))
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