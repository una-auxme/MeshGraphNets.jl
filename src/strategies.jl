#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import DifferentialEquations: ODEFunction
import SciMLBase: AbstractSensitivityAlgorithm
import SciMLSensitivity: InterpolatingAdjoint, ZygoteVJP

#######################################################
# Abstract type and functions for training strategies #
#######################################################

abstract type TrainingStrategy end

"""
    prepare_training(strategy)

Function that is executed once before training. Can be overwritten by training strategies if necessary.

## Arguments
- `strategy`: Used training strategy.

## Returns
- Tuple containing the results of the function.
"""
function prepare_training(::TrainingStrategy) 
    return (nothing,)
end

"""
    get_delta(strategy, trajectory_length)

Returns the delta between samples in the training data.

## Arguments
- `strategy`: Used training strategy.
- Trajectory length (used for collocation strategies).

## Returns
- Delta between samples in the training data.
"""
function get_delta(strategy::TrainingStrategy, ::Integer)
    throw(ArgumentError("Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies."))
end

"""
    init_train_step(strategy, t, ta)

Function that is executed before each training sample.

## Arguments
- `strategy`: Used training strategy.
- `t`: Tuple containing the variables necessary for initializing training.
- `ta`: Tuple with additional variables that is returned from [prepare_training](@ref).

## Returns
- Tuple containing variables needed for [train_step](@ref).
"""
function init_train_step(strategy::TrainingStrategy, ::Tuple, ::Tuple)
    throw(ArgumentError("Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies."))
end

"""
    train_step(strategy, t)

Performs a single training step and return the resulting gradients and loss.

## Arguments
- `strategy`: Solver strategy that is used for training.
- `t`: Tuple that is returned from [`init_train_step`](@ref).

## Returns
- Gradients for optimization step.
- Loss for optimization step.
"""
function train_step(strategy::TrainingStrategy, ::Tuple)
    throw(ArgumentError("Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies."))
end

"""
    validation_step(strategy, t)

Performs validation of a single trajectory. Should be overwritten by training strategies to determine simulation and data interval before calling the inner function [_validation_step](@ref).

## Arguments
- `strategy`: Type of training strategy (used for dispatch).
- `t`: Tuple containing the variables necessary for validation.

## Returns
- See [_validation_step](@ref).
"""
function validation_step(strategy::TrainingStrategy, ::Tuple)
    throw(ArgumentError("Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies."))
end

"""
    _validation_step(t, sim_interval, data_interval)

Inner function for validation of a single trajectory.

## Arguments
- `t`: Tuple containing the variables necessary for validation.
- `sim_interval`: Interval that determines the simulated time for the validation.
- `data_interval`: Interval that determines the indices of the timesteps in ground truth and prediction data.

## Returns
- Loss calculated on the difference between ground truth and prediction (via mse).
- Ground truth data with `data_interval` as timesteps.
- Prediction data with `data_interval` as timesteps.
"""
function _validation_step(t::Tuple, sim_interval, data_interval)

    mgn, data, meta, _, solver, solver_dt, fields, node_type, edge_features, senders, receivers, mask, val_mask, inflow_mask, data = t

    initial_state = Dict(
        [typeof(v) <: AbstractArray ? (k, v[:, :, 1]) : (k, v) for (k,v) in data]
    )

    target_dict = Dict{String, Int32}()
    for tf in meta["target_features"]
        target_dict[tf] = meta["features"][tf]["dim"]
    end

    gt = vcat([data[tf] for tf in meta["target_features"]]...)[:, :, data_interval]

    sol_u, _ = rollout(solver, mgn, initial_state, fields, meta, meta["target_features"], target_dict, node_type, edge_features, senders, receivers, val_mask, inflow_mask, data, sim_interval[1], sim_interval[end], solver_dt, sim_interval; show_progress = false)
    prediction = cat(sol_u..., dims = 3)[:, :, data_interval]

    error = mean((prediction - gt) .^ 2; dims = 3)

    return mean(error[mask]), gt, prediction
end

####################################################################
# Abstract type and functions for solver based training strategies #
####################################################################

abstract type SolverStrategy <: TrainingStrategy end

function get_delta(::SolverStrategy, ::Integer)
    return 1
end

function init_train_step(::SolverStrategy, t::Tuple, ta::Tuple)

    mgn, data, meta, fields, target_fields, node_type, edge_features, senders, receivers, _, _, val_mask = t

    target_dict = Dict{String, Int32}()
    for tf in meta["target_features"]
        target_dict[tf] = meta["features"][tf]["dim"]
    end

    initial_state = Dict{String, AbstractArray}(
        [typeof(v) <: AbstractArray ? (k, v[:, :, 1]) : (k, v) for (k,v) in data]
    )
    for k in keys(initial_state)
        if endswith(k, ".ev")
            delete!(initial_state, k)
        end
    end

    inputs = deepcopy(initial_state)
    for i in keys(target_dict)
        delete!(inputs, "target|" * i)
    end

    gt = vcat([data[tf] for tf in meta["target_features"]]...)
    u0 = gt[:, :, 1]

    return (mgn, data, inputs, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, u0, gt)
end

function train_step(strategy::SolverStrategy, t::Tuple)

    mgn, data, inputs, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, u0, gt = t

    inflow_mask = repeat(data["node_type"][:, :, 1] .== 1, sum(size(data[field], 1) for field in meta["target_features"]), 1) |> cpu_device()

    pr = ProgressUnknown(showspeed = true)

    ff = ODEFunction{false}((x, p, t) -> ode_func_train(x, (mgn, p, data, inputs, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, inflow_mask, strategy, pr), t))
    prob = ODEProblem(ff, u0, (strategy.tstart, strategy.tstop), mgn.ps)

    shoot_loss, back = Zygote.pullback(ps -> train_loss(strategy, (prob, ps, u0, nothing, gt, val_mask, mgn.n_norm, target_fields, [meta["features"][tf]["dim"] for tf in target_fields])), mgn.ps)
    shoot_gs = back(one(shoot_loss))
    return shoot_gs, shoot_loss
end

"""
    train_loss(strategy, t)

Inner function for a solver based training step that calculates the loss based on the difference between the ground truth and the predicted solution.

## Arguments
- `strategy`: Solver strategy that is used for training.
- `t`: Tuple containing all variables necessary for loss calculation.

## Returns
- Calculated loss.
"""
function train_loss(strategy::SolverStrategy, ::Tuple)
    throw(ArgumentError("Unknown solver based training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available solver strategies."))
end

function validation_step(strategy::SolverStrategy, t::Tuple)
    sim_interval = strategy.tstart:strategy.dt:strategy.tstop
    data_interval = 1:length(sim_interval)

    return _validation_step(t, sim_interval, data_interval)
end



"""
    SingleShooting(tstart, dt, tstop, solver; sense = InterpolatingAdjoint(autojacvec = ZygoteVJP()), solargs...)

The default solver based training that is normally used for NeuralODEs.
Simulates the system from `tstart` to `tstop` and calculates the loss based on the difference between the prediction and the ground truth at the timesteps `tstart:dt:tstop`.

## Arguments
- `tstart`: Start time of the simulation.
- `dt`: Interval at which the simulation is saved.
- `tstop`: Stop time of the simulation.
- `solver`: Solver that is used for simulating the system.

## Keyword Arguments
- `sense = InterpolatingAdjoint(autojacvec = ZygoteVJP())`: The sensitivity algorithm that is used for caluclating the sensitivities.
- `solargs`: Keyword arguments that are passed on to the solver.
"""
struct SingleShooting <: SolverStrategy
    tstart::Float32
    dt::Float32
    tstop::Float32
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    solargs
end

function SingleShooting(tstart::Float32, dt::Float32, tstop::Float32, solver::OrdinaryDiffEqAlgorithm; sense::AbstractSensitivityAlgorithm = InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true), solargs...)
    SingleShooting(tstart, dt, tstop, solver, sense, solargs)
end

function train_loss(strategy::SingleShooting, t::Tuple)

    prob, ps, u0, callback_solve, gt, val_mask, n_norm, target_fields, target_dims = t

    sol = solve(remake(prob; p = ps), strategy.solver; u0 = u0, saveat = strategy.tstart:strategy.dt:strategy.tstop, tstops = strategy.tstart:strategy.dt:strategy.tstop, sensealg = strategy.sense, callback = callback_solve, strategy.solargs...)

    pred = typeof(gt) <: CuArray ? CuArray(sol) : Array(sol)

    local gt_n
    local pred_n

    for i in eachindex(target_fields)
        gt_n = vcat([n_norm[target_fields[i]](gt[sum(target_dims[1:i-1])+1:sum(target_dims[1:i]), :, 1:size(pred, 3)]) for i in eachindex(target_fields)]...)
        pred_n = vcat([n_norm[target_fields[i]](pred[sum(target_dims[1:i-1])+1:sum(target_dims[1:i]), :, :]) for i in eachindex(target_fields)]...)
    end

    error = (gt_n[:, :, 1:size(pred, 3)] .- pred_n) .^ 2 |> cpu_device()

    err_buf = Zygote.Buffer(error)

    vm = cpu_device()(val_mask)

    err_buf[:, :, :] = error
    for i in axes(err_buf, 3)
        err_buf[:, :, i] = err_buf[:, :, i] .* vm
    end
    loss = mean(copy(err_buf))

    return loss
end



"""
    MultipleShooting(tstart, dt, tstop, solver, interval_size, continuity_term = 100; sense = InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true), solargs...)

Similar to SingleShooting, but splits the trajectory into intervals that are solved independently and then combines them for loss calculation.
Useful if the network tends to get stuck in a local minimum if SingleShooting is used.

## Arguments
- `tstart`: Start time of the simulation.
- `dt`: Interval at which the simulation is saved.
- `tstop`: Stop time of the simulation.
- `solver`: Solver that is used for simulating the system.
- `interval_size`: Size of the intervals (i.e. number of datapoints in one interval).
- `continuity_term = 100`: Factor by which the error between points of concurrent intervals is multiplied.

## Keyword Arguments
- `sense = InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true)`:
- `solargs`: Keyword arguments that are passed on to the solver.
"""
struct MultipleShooting <: SolverStrategy
    tstart::Float32
    dt::Float32
    tstop::Float32
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    interval_size::Integer                  # Number of observations in one interval
    continuity_term::Integer
    solargs
end

function MultipleShooting(tstart::Float32, dt::Float32, tstop::Float32, solver::OrdinaryDiffEqAlgorithm; sense::AbstractSensitivityAlgorithm = InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true), interval_size, continuity_term = 100, solargs...)
    MultipleShooting(tstart, dt, tstop, solver, sense, interval_size, continuity_term, solargs)
end

function train_loss(strategy::MultipleShooting, t::Tuple)
    prob, ps, _, callback_solve, gt, val_mask, _, _, _ = t

    tsteps = strategy.tstart:strategy.dt:strategy.tstop
    ranges = [i:min(length(tsteps), i + strategy.interval_size - 1) for i in 1:strategy.interval_size-1:length(tsteps)-1]

    sols = [
        solve(
            remake(
                prob;
                p = ps,
                tspan = (tsteps[first(rg)], tsteps[last(rg)]),
                u0 = gt[:, :, first(rg)]
            ),
            strategy.solver;
            saveat = tsteps[rg],
            sensealg = strategy.sense,
            callback = callback_solve,
            strategy.solargs...
        ) for rg in ranges
    ]
    group_predictions = typeof(gt) <: CuArray ? CuArray.(sols) : Array.(sols)

    retcodes = [sol.retcode for sol in sols]
    if any(retcodes .!= :Success)
        return Inf
    end

    vm = cpu_device()(val_mask)

    loss = 0
    for (i, rg) in enumerate(ranges)
        error = (gt[:, :, rg] - group_predictions[i]) .^ 2 |> cpu_device()

        err_buf = Zygote.Buffer(error)
        err_buf[:, :, :] = error
        for i in axes(err_buf, 3)
            err_buf[:, :, i] = err_buf[:, :, i] .* vm
        end
        loss += mean(copy(err_buf))
        
        if i > 1
            loss += strategy.continuity_term * sum(abs, group_predictions[i - 1][:, :, end] - gt[:, :, first(rg)])
        end
    end

    return loss
end

#########################################################################
# Abstract type and functions for collocation based training strategies #
#########################################################################

abstract type CollocationStrategy <: TrainingStrategy end

function get_delta(strategy::CollocationStrategy, trajectory_length::Integer)
    return strategy.window_size > 0 ? strategy.window_size : trajectory_length - 1
end

function init_train_step(::CollocationStrategy, t::Tuple, ::Tuple)

    mgn, data, meta, fields, target_fields, node_type, edge_features, senders, receivers, datapoint, mask, _ = t

    if typeof(meta["dt"]) <: AbstractArray
        target_quantities_change = vcat([mgn.o_norm[field]((data["target|" * field][:, :, datapoint] - data[field][:, :, datapoint]) / (meta["dt"][datapoint + 1] - meta["dt"][datapoint])) for field in target_fields]...)
    else
        target_quantities_change = vcat([mgn.o_norm[field]((data["target|" * field][:, :, datapoint] - data[field][:, :, datapoint]) / Float32(meta["dt"])) for field in target_fields]...)
    end
    graph = build_graph(mgn, data, fields, datapoint, node_type, edge_features, senders, receivers)

    return (mgn, graph, target_quantities_change, mask)
end

function train_step(::CollocationStrategy, t::Tuple)

    mgn, graph, target_quantities_change, mask = t

    return step!(mgn, graph, target_quantities_change, mask, mse_reduce)
end

function validation_step(::CollocationStrategy, t::Tuple)
    sim_interval = t[2]["dt"][1]:t[2]["dt"][2]-t[2]["dt"][1]:t[2]["dt"][t[4]]
    data_interval = 1:t[4]

    return _validation_step(t, sim_interval, data_interval)
end



"""
    Collocation(; window_size = 0)

Compares the prediction of the system with the derivative from the data (via finite differences).
Useful for initial training of the system since it it faster than training with a solver.

## Keyword Arguments
- `window_size = 0`: Number of steps from each trajectory (starting at the beginning) that are used for training. If the number is zero then the whole trajectory is used.
"""
struct Collocation <: CollocationStrategy
    window_size::Integer
    random::Bool
end
function Collocation(;window_size::Integer = 0, random = true)
    Collocation(window_size, random)
end
