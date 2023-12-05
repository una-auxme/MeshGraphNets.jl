#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module MeshGraphNets

using GraphNetCore

using CUDA
using Lux, LuxCUDA
using Optimisers
using Zygote

import DifferentialEquations:  ODEProblem, OrdinaryDiffEqAlgorithm, Tsit5
import ProgressMeter: Progress, ProgressUnknown

import Base: @kwdef
import ChainRulesCore: @ignore_derivatives
import DifferentialEquations: solve, remake
import HDF5: h5open, create_group, open_group
import ProgressMeter: next!, update!, finish!
import Statistics: mean

include("utils.jl")
include("graph.jl")
include("dataset.jl")

export SingleShooting, MultipleShooting, RandomCollocation, Collocation

export train_network, eval_network

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
    solver_valid::OrdinaryDiffEqAlgorithm = Tsit5()
    solver_valid_dt::Union{Nothing, Float32} = nothing
end

"""
    train_network(noise_stddevs, opt, ds_path, cp_path; kws...)

Starts the training process with the given configuration.

# Arguments
- `noise_stddevs`: Array containing the noise that is added to the specified node types.
- `opt`: Optimiser that is used for training.
- `ds_path`: Path to the dataset folder.
- `cp_path`: Path where checkpoints are being saved to.
- `kws`: Keyword arguments that customize the training process.

# Keyword Arguments
- `mps = 15`: Number of message passing steps.
- `layer_size = 256`: Latent size of the hidden layers inside MLPs.
- `hidden_layers = 2`: Number of hidden layers inside MLPs.
- `batchsize = 1`: Size per batch *(not implemented yet)*.
- `epochs = 1`: Number of epochs.
- `steps = 10e6`: Number of training steps.
- `checkpoint = 10000`: Number of steps after which checkpoints are created.
- `norm_steps = 1000`: Number of steps before training (accumulate normalization stats).
- `types_updated = [0, 5]`: Array containing node types which are updated after each step.
- `types_noisy = [0]`: Array containing node types which noise is added to.
- `training_strategy = Collocation()`: Methods used for training. See 
- `use_cuda = true`: Whether a GPU is used for training or not (if available). Currently only CUDA GPUs are supported.
- `gpu_idx = 0`: Index of GPU. See *nvidia-smi* for reference.
- `cell_idxs = [0]`: Indices of cells that are plotted during validation (if enabled).
- `solver_valid = Tsit5()`: Which solver should be used for validation during training.
- `solver_valid_dt = nothing`: If set, the solver for validation will use fixed timesteps.

# Training Strategies
- `Collocation`
- `RandomCollocation`
- `SingleShooting`
- `MultipleShooting`

See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/cylinder_flow) for reference.

# Returns
- `mgn`: The trained network as a [`GraphNetwork`](@ref) struct.
"""
function train_network(noise_stddevs, opt, ds_path, cp_path; kws...)
    args = Args(;kws...)

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU..."
        CUDA.device!(args.gpu_idx)
        CUDA.allowscalar(true)
        device = gpu_device()
    else
        @info "Training on CPU..."
        device = cpu_device()
    end

    @info "Training with $(typeof(args.training_strategy))..."

    println("Loading training data...")
    dataset = load_dataset(ds_path, true)
    clear_log(1)
    @info "Training data loaded!"

    println("Building model...")

    quantities = 0
    norms = Dict{String, Union{NormaliserOffline, NormaliserOnline}}()

    for feature in dataset.meta["feature_names"]
        if feature == "mesh_pos" || feature == "cells"
            continue
        end
        if getfield(Base, Symbol(uppercasefirst(dataset.meta["features"][feature]["dtype"]))) == Bool
            quantities += 1
            norms[feature] = NormaliserOffline(0.0f0, 1.0f0)
        elseif getfield(Base, Symbol(uppercasefirst(dataset.meta["features"][feature]["dtype"]))) == Int32
            if haskey(dataset.meta["features"][feature], "onehot") && dataset.meta["features"][feature]["onehot"]
                quantities += dataset.meta["features"][feature]["data_max"] - dataset.meta["features"][feature]["data_min"] + 1
                if haskey(dataset.meta["features"][feature], "target_min") && haskey(dataset.meta["features"][feature], "target_max")
                    norms[feature] = NormaliserOffline(0.0f0, 1.0f0, Float32(dataset.meta["features"][feature]["target_min"]), Float32(dataset.meta["features"][feature]["target_max"]))
                else
                    norms[feature] = NormaliserOffline(0.0f0, 1.0f0)
                end
            end
        else
            quantities += dataset.meta["features"][feature]["dim"]
            if haskey(dataset.meta["features"][feature], "data_min") && haskey(dataset.meta["features"][feature], "data_max")
                if haskey(dataset.meta["features"][feature], "target_min") && haskey(dataset.meta["features"][feature], "target_max")
                    norms[feature] = NormaliserOffline(Float32(dataset.meta["features"][feature]["data_min"]), Float32(dataset.meta["features"][feature]["data_max"]), Float32(dataset.meta["features"][feature]["target_min"]), Float32(dataset.meta["features"][feature]["target_max"]))
                else
                    norms[feature] = NormaliserOffline(Float32(dataset.meta["features"][feature]["data_min"]), Float32(dataset.meta["features"][feature]["data_max"]))
                end
            else
                norms[feature] = NormaliserOnline(dataset.meta["features"][feature]["dim"], device; max_acc = Float32(args.norm_steps))
            end
        end
    end

    dims = dataset.meta["dims"]
    outputs = 0
    for tf in dataset.meta["target_features"]
        outputs += dataset.meta["features"][tf]["dim"]
    end

    mgn, opt_state, df_train, df_valid = load(quantities, typeof(dims) <: AbstractArray ? length(dims) : dims, norms, outputs, args.mps, args.layer_size, args.hidden_layers, opt, device, cp_path)

    if isnothing(opt_state)
        if typeof(opt) <: AbstractRule
            opt_state = Optimisers.setup(opt, mgn.ps)
        else 
            opt_state = opt
        end
    end
    Lux.trainmode(mgn.st)

    clear_log(1)
    @info "Model built!"
    print("Compiling code...\r")

    train_mgn!(mgn, opt_state, dataset, noise_stddevs, df_train, df_valid, device, cp_path, args)

    return mgn
end

function train_mgn!(mgn::GraphNetwork, opt_state, dataset::Dataset, noise, df_train, df_valid, device::Function, cp_path, args::Args)
    checkpoint = length(df_train.step) > 0 ? last(df_train.step) : 0
    step = checkpoint
    cp_progress = 0
    min_validation_loss = length(df_valid.loss) > 0 ? last(df_valid.loss) : Inf32

    pr = Progress(args.epochs*args.steps; desc = "Training progress: ", dt=1.0, barlen=50, start=checkpoint, showspeed=true)

    local tmp_loss = 0.0f0
    local avg_loss = 0.0f0
    fields = deleteat!(copy(dataset.meta["feature_names"]), findall(x -> x == "node_type" || x == "mesh_pos" || x == "cells", dataset.meta["feature_names"]))

    delta = get_delta(args.training_strategy, dataset.meta["trajectory_length"])

    callback_plt = args.training_strategy.plot_progress ? get_plt_callback(args.training_strategy, args.cell_idxs, cp_path) : nothing

    train_tuple_additional = prepare_training(args.training_strategy)

    for _ in checkpoint:delta:args.steps*args.epochs
        data, meta = next_trajectory!(dataset, device; types_noisy = args.types_noisy, noise_stddevs=noise, ts = args.training_strategy)
        
        mask = Int32.(findall(x -> x in args.types_updated, data["node_type"][1, :, 1])) |> device
        
        val_mask = Float32.(map(x -> x in args.types_updated, data["node_type"][:, :, 1]))
        val_mask = repeat(val_mask, sum(size(data[field], 1) for field in meta["target_features"]), 1) |> device

        node_type, senders, receivers, edge_features = create_base_graph(data, meta["features"]["node_type"]["data_max"], meta["features"]["node_type"]["data_min"], device)

        for datapoint in 1:delta
            train_tuple = init_train_step(args.training_strategy, (mgn, data, meta, fields, meta["target_features"], node_type, edge_features, senders, receivers, datapoint, mask, val_mask), train_tuple_additional)
            if typeof(opt_state) <: Optimisers.Leaf
                gs, losses = train_step(args.training_strategy, (train_tuple..., nothing))
                
                tmp_loss += sum(losses)

                if step + datapoint > args.norm_steps
                    for i in eachindex(gs)
                        opt_state, ps = Optimisers.update(opt_state, mgn.ps, gs[i])
                        mgn.ps = ps
                    end
                    next!(pr, showvalues=[(:train_step,"$(step + datapoint)/$(args.epochs*args.steps)"), (:train_loss, sum(losses)), (:checkpoint, length(df_train.step) > 0 ? last(df_train.step) : 0), (:data_interval, delta == 1 ? "1:end" : 1:delta), (:min_validation_loss, min_validation_loss)])
                else
                    next!(pr, showvalues=[(:step,"$(step + datapoint)/$(args.epochs*args.steps)"), (:loss,"acc norm stats..."), (:checkpoint, 0)])
                end
            else
                result = train_step(args.training_strategy, (train_tuple..., opt_state))
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

                ve, g, p = validation_step(args.training_strategy, (
                    mgn, data_valid, meta_valid, delta, args.solver_valid, args.solver_valid_dt, fields, node_type_valid, edge_features_valid, senders_valid, receivers_valid, val_mask_valid,
                    opt_state, step, avg_loss, cp_path, tmp_loss, rollout
                ))
                
                valid_error += ve
                if i == 1
                    gt = g
                    prediction = p
                end
                
                next!(pr_valid, showvalues = [(:trajectory, "$i/$(meta["n_trajectories_valid"])"), (:valid_loss, "$((valid_error + ve) / i)")])
            end

            finish!(pr_valid)

            clear_log(3)

            if args.training_strategy.plot_progress
                sim_interval = get_sim_interval(args.training_strategy, (nothing, data, nothing, delta, ntuple(_ -> nothing, 14)...))
                callback_plt(valid_error / meta["n_trajectories_valid"], sim_interval, gt[1, :, :], prediction[1, :, :]; xlims = (sim_interval[1] - data["dt"][2], sim_interval[end] + data["dt"][2]))
            end

            if valid_error / meta["n_trajectories_valid"] < min_validation_loss
                save!(mgn, opt_state, df_train, df_valid, step, valid_error / meta["n_trajectories_valid"], joinpath(cp_path, "valid"); is_training = false)
                min_validation_loss = valid_error / meta["n_trajectories_valid"]
            end

        end

        if cp_progress >= args.checkpoint
            save!(mgn, opt_state, df_train, df_valid, step, avg_loss / Float32(step / delta), cp_path)
            avg_loss = 0.0f0
            cp_progress = 0
        end
    end
    finish!(pr)
end


# Zygote.accum(x::NamedTuple, y::Base.RefValue) = Zygote.accum(x, y.x)

"""
    eval_network(ds_path, cp_path, out_path, solver; start, stop, dt, saves, mse_steps, kws...)

Starts the evaluation process with the given configuration.

# Arguments
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

# Keyword Arguments
- `mps = 15`: Number of message passing steps.
- `layer_size = 256`: Latent size of the hidden layers inside MLPs.
- `hidden_layers = 2`: Number of hidden layers inside MLPs.
- `types_updated = [0, 5]`: Array containing node types which are updated after each step.
- `use_cuda = true`: Whether a GPU is used for training or not (if available). Currently only CUDA GPUs are supported.
- `gpu_idx = 0`: Index of GPU. See *nvidia-smi* for reference.
- `num_rollouts = 10`: Number of trajectories that are simulated (from the test dataset).
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

    clear_log(1)
    @info "Evaluation data loaded!"


    println("Building model...")

    quantities = 0
    norms = Dict{String, Union{NormaliserOffline, NormaliserOnline}}()

    for feature in dataset.meta["feature_names"]
        if feature == "mesh_pos" || feature == "cells"
            continue
        end
        if getfield(Base, Symbol(uppercasefirst(dataset.meta["features"][feature]["dtype"]))) == Bool
            quantities += 1
            norms[feature] = NormaliserOffline(0.0f0, 1.0f0)
        elseif getfield(Base, Symbol(uppercasefirst(dataset.meta["features"][feature]["dtype"]))) == Int32
            if haskey(dataset.meta["features"][feature], "onehot") && dataset.meta["features"][feature]["onehot"]
                quantities += dataset.meta["features"][feature]["data_max"] - dataset.meta["features"][feature]["data_min"] + 1
                if haskey(dataset.meta["features"][feature], "target_min") && haskey(dataset.meta["features"][feature], "target_max")
                    norms[feature] = NormaliserOffline(0.0f0, 1.0f0, Float32(dataset.meta["features"][feature]["target_min"]), Float32(dataset.meta["features"][feature]["target_max"]))
                else
                    norms[feature] = NormaliserOffline(0.0f0, 1.0f0)
                end
            end
        else
            quantities += dataset.meta["features"][feature]["dim"]
            if haskey(dataset.meta["features"][feature], "data_min") && haskey(dataset.meta["features"][feature], "data_max")
                if haskey(dataset.meta["features"][feature], "target_min") && haskey(dataset.meta["features"][feature], "target_max")
                    norms[feature] = NormaliserOffline(Float32(dataset.meta["features"][feature]["data_min"]), Float32(dataset.meta["features"][feature]["data_max"]), Float32(dataset.meta["features"][feature]["target_min"]), Float32(dataset.meta["features"][feature]["target_max"]))
                else
                    norms[feature] = NormaliserOffline(Float32(dataset.meta["features"][feature]["data_min"]), Float32(dataset.meta["features"][feature]["data_max"]))
                end
            else
                norms[feature] = NormaliserOnline(dataset.meta["features"][feature]["dim"], device, max_acc = Float32(args.norm_steps))
            end
        end
    end

    dims = dataset.meta["dims"]
    outputs = 0
    for tf in dataset.meta["target_features"]
        outputs += dataset.meta["features"][tf]["dim"]
    end

    mgn, _, _, _ = load(quantities, typeof(dims) <: AbstractArray ? length(dims) : dims, norms, outputs, args.mps, args.layer_size, args.hidden_layers, nothing, device, cp_path)
    Lux.testmode(mgn.st)

    clear_log(1)
    @info "Model built!"

    eval_network!(solver, mgn, dataset, device, out_path, start, stop, dt, saves, mse_steps, args)
end

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

        node_type, senders, receivers, edge_features = create_base_graph(data, meta["features"]["node_type"]["data_max"], meta["features"]["node_type"]["data_min"], device)

        fields = deleteat!(copy(dataset.meta["feature_names"]), findall(x -> x == "node_type" || x == "mesh_pos" || x == "cells", dataset.meta["feature_names"]))

        target_dict = Dict{String, Int32}()
        for tf in meta["target_features"]
            target_dict[tf] = meta["features"][tf]["dim"]
        end

        solution = rollout(solver, mgn, initial_state, fields, dataset.meta["target_features"], target_dict, node_type, edge_features, senders, receivers, val_mask, start, stop, dt, saves)
        
        prediction = cat(solution.u..., dims = 3)
        error = mean((prediction - vcat([data[field][:, :, 1:length(saves)] for field in meta["target_features"]]...)) .^ 2; dims = 2)
        timesteps[(ti, "timesteps")] = solution.t

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

function rollout(solver, mgn::GraphNetwork, initial_state, fields, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, start, stop, dt, saves; show_progress = true)
    pr = show_progress ? ProgressUnknown(showspeed = true) : nothing

    interval = (start, stop)
    x0 = vcat([initial_state[field] for field in target_fields]...)
    inputs = deepcopy(initial_state)
    for i in keys(target_dict)
        delete!(inputs, i)
    end

    prob = ODEProblem(fast_step, x0, interval, (mgn, mgn.ps, inputs, fields, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, pr))
    if isnothing(dt)
        sol = solve(prob, solver; saveat=saves)
    else
        sol = solve(prob, solver; adaptive=false, dt=dt, saveat=saves)
    end
    
    if show_progress
        finish!(pr)
    end
    
    return sol
end

function fast_step(x, (mgn, ps, inputs, fields, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, pr), t)
    offset = 1
    for k in target_fields
        inputs[k] = x[offset:offset + target_dict[k] - 1, :]
        offset += target_dict[k]
    end

    graph = build_graph(mgn, inputs, fields, node_type, edge_features, senders, receivers, false)
    output, st = mgn.model(graph, ps, mgn.st)
    mgn.st = st

    quan_update = inverse_data(mgn.o_norm, output)
    
    @ignore_derivatives begin
        if !isnothing(pr)
            next!(pr, showvalues=[(:t,"$(t)")])
        end
    end
    
    return quan_update .* val_mask
end

end