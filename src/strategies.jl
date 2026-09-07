# Copyright (c) 2023 Julian Trommer
# Copyright (c) 2025 Luca Kahlenberg
# SPDX-License-Identifier: MIT
# See LICENSE for details.

import SciMLBase: AbstractSensitivityAlgorithm, ODEFunction
import SciMLSensitivity: GaussAdjoint, ZygoteVJP

import SciMLBase: isadaptive, successful_retcode

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
- `trajectory_length`: Trajectory length (used for derivative-based strategies).

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
- Loss calculated from the difference between the ground truth and prediction.
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
"""
function _validation_step(t::Tuple, sim_interval, data_interval)
    mgn, data, meta, _, solver, solver_dt, fields, node_type, edge_features,
    senders, receivers, mask, val_mask, inflow_mask, pr = t

    target_dict = Dict{String, Int32}()
    for tf in meta["target_features"]
        target_dict[tf] = meta["features"][tf]["dim"]
    end

    gt = vcat([data[tf] for tf in meta["target_features"]]...)[:, :, data_interval]

    sol_u,
    _ = rollout(
        solver, mgn, data, fields, meta, meta["target_features"], target_dict,
        node_type, edge_features, senders, receivers, val_mask, inflow_mask,
        sim_interval[1], sim_interval[end], solver_dt, sim_interval, pr)
    prediction = cat(sol_u...; dims = 3)[:, :, data_interval]

    error = mean((prediction - gt) .^ 2; dims = 3)

    return mean(error[:, mask])
end

####################################################################
# Abstract type and functions for solver based training strategies #
####################################################################

abstract type SolverStrategy <: TrainingStrategy end

function get_delta(::SolverStrategy, ::Integer)
    return 1
end

function init_train_step(::SolverStrategy, t::Tuple, ta::Tuple)
    mgn, data, meta, fields, target_fields, node_type,
    edge_features, senders, receivers, _, idx_mask, val_mask = t

    target_dict = Dict{String, Int32}()
    for tf in meta["target_features"]
        target_dict[tf] = meta["features"][tf]["dim"]
    end

    inputs = Dict{String, AbstractArray}(
        [typeof(data[field]) <: AbstractArray ? (field, data[field][:, :, 1]) :
         (field, data[field]) for field in fields]
    )

    gt = vcat([data[tf] for tf in meta["target_features"]]...)
    u0 = gt[:, :, 1]

    return (mgn, data, inputs, fields, meta, target_fields, target_dict,
        node_type, edge_features, senders, receivers, idx_mask, val_mask, u0, gt)
end

function train_step(strategy::SolverStrategy, t::Tuple)
    mgn, data, inputs, fields, meta, target_fields, target_dict, node_type,
    edge_features, senders, receivers, idx_mask, val_mask, u0, gt = t

    pr = ProgressUnknown(; desc = "Solver progress: ", dt = 1.0, showspeed = true)
    print("\n\n\n\n\n\n\n") # display solver progress after main progress

    ff = ODEFunction{false}((x,
        p,
        t) -> ode_func_train(x,
        (mgn, p, data, inputs, fields, meta,
            target_fields, target_dict, node_type,
            edge_features, senders, receivers, val_mask, data["inflow_mask"], strategy, pr),
        t))
    prob = ODEProblem(ff, u0, (strategy.tstart, strategy.tstop), mgn.train_state.parameters)

    shoot_loss,
    shoot_gs = Zygote.withgradient(
        ps -> train_loss(strategy,
            (prob, ps, u0, nothing, gt, idx_mask,
                val_mask, mgn.n_norm, target_fields,
                [meta["features"][tf]["dim"] for tf in target_fields])),
        mgn.train_state.parameters)

    clear_log(7, false)
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
function train_loss(strategy::SolverStrategy, t::Tuple)
    prob, ps, u0, callback_solve, gt, idx_mask, val_mask, n_norm, target_fields,
    target_dims = t

    sol = solve(remake(prob; p = ps), strategy.solver; u0 = u0,
        saveat = (first(prob.tspan)):(strategy.dt):(last(prob.tspan)),
        sensealg = strategy.sense, callback = callback_solve, strategy.solargs...)

    pred = typeof(gt) <: CuArray ? CuArray(sol) : Array(sol)
    error = abs2.(gt[:, :, axes(pred, 3)] - pred)

    mask = reshape(val_mask, size(val_mask)..., 1)
    denominator = size(error, 3) * sum(val_mask)

    return sum(error .* mask) / denominator
end

function validation_step(strategy::SolverStrategy, t::Tuple)
    sim_interval = (strategy.tstart):(strategy.dt):(strategy.tstop)
    data_interval = 1:length(sim_interval)

    return _validation_step(t, sim_interval, data_interval)
end

"""
    SolverTraining(tstart, dt, tstop, solver;
                   sense = GaussAdjoint(autojacvec = ZygoteVJP()), solargs...)

The default solver based training that is normally used for NeuralODEs.
Simulates the system from `tstart` to `tstop` and calculates the loss based on the difference between the prediction and the ground truth at the timesteps `tstart:dt:tstop`.

## Arguments
- `tstart`: Start time of the simulation.
- `dt`: Interval at which the simulation is saved.
- `tstop`: Stop time of the simulation.
- `solver`: Solver that is used for simulating the system.

## Keyword Arguments
- `sense = GaussAdjoint(autojacvec = ZygoteVJP())`: The sensitivity algorithm used for calculating sensitivities. Checkpointing is enabled by default for adaptive solvers.
- `solargs`: Keyword arguments that are passed on to the solver.
"""
struct SolverTraining <: SolverStrategy
    tstart::Float32
    dt::Float32
    tstop::Float32
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    solargs::Any
end

function SolverTraining(tstart::Float32,
        dt::Float32,
        tstop::Float32,
        solver::OrdinaryDiffEqAlgorithm;
        sense::AbstractSensitivityAlgorithm = GaussAdjoint(;
            autojacvec = ZygoteVJP(), checkpointing = isadaptive(solver) ? true : false),
        solargs...)
    SolverTraining(tstart, dt, tstop, solver, sense, solargs)
end

"""
    SolverBatchTraining(tstart, dt, tstop, interval_size, solver;
                        sense = GaussAdjoint(autojacvec = ZygoteVJP()), solargs...)

Solver-based training that splits the trajectory into overlapping intervals and performs a
separate training step for each interval. Consecutive intervals share their boundary point.

## Arguments
- `tstart`: Start time of the simulation.
- `dt`: Interval at which the simulation is saved.
- `tstop`: Stop time of the simulation.
- `interval_size`: Number of datapoints in each interval.
- `solver`: Solver that is used for simulating the system.

## Keyword Arguments
- `sense = GaussAdjoint(autojacvec = ZygoteVJP())`: The sensitivity algorithm used for calculating sensitivities. Checkpointing is enabled by default for adaptive solvers.
- `solargs`: Keyword arguments that are passed on to the solver.
"""
struct SolverBatchTraining <: SolverStrategy
    tstart::Float32
    dt::Float32
    tstop::Float32
    interval_size::Integer
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    solargs::Any
end

function SolverBatchTraining(tstart::Float32,
        dt::Float32,
        tstop::Float32,
        interval_size::Integer,
        solver::OrdinaryDiffEqAlgorithm;
        sense::AbstractSensitivityAlgorithm = GaussAdjoint(;
            autojacvec = ZygoteVJP(), checkpointing = isadaptive(solver) ? true : false),
        solargs...)
    SolverBatchTraining(tstart, dt, tstop, interval_size, solver, sense, solargs)
end

function get_delta(strategy::SolverBatchTraining, ::Integer)
    tsteps = (strategy.tstart):(strategy.dt):(strategy.tstop)
    ranges = [i:min(length(tsteps), i + strategy.interval_size - 1)
              for i in 1:(strategy.interval_size - 1):(length(tsteps) - 1)]
    return ranges
end

function init_train_step(strategy::SolverBatchTraining, t::Tuple, ta::Tuple)
    mgn, data, meta, fields, target_fields, node_type, edge_features,
    senders, receivers, datapoint_interval, idx_mask, val_mask = t

    tsteps = (strategy.tstart):(strategy.dt):(strategy.tstop)
    tspan = (tsteps[first(datapoint_interval)], tsteps[last(datapoint_interval)])

    target_dict = Dict{String, Int32}()
    for tf in meta["target_features"]
        target_dict[tf] = meta["features"][tf]["dim"]
    end

    inputs = Dict{String, AbstractArray}(
        [typeof(data[field]) <: AbstractArray ?
         (field, data[field][:, :, min(size(data[field], 3), first(datapoint_interval))]) :
         (field, data[field]) for field in fields]
    )

    gt = vcat([data[tf] for tf in meta["target_features"]]...)[:, :, datapoint_interval]
    u0 = gt[:, :, 1]

    return (mgn, data, inputs, fields, meta, target_fields, target_dict,
        node_type, edge_features, senders, receivers, idx_mask, val_mask, u0, gt, tspan)
end

function train_step(strategy::SolverBatchTraining, t::Tuple)
    mgn, data, inputs, fields, meta, target_fields, target_dict, node_type,
    edge_features, senders, receivers, idx_mask, val_mask, u0, gt, tspan = t

    pr = ProgressUnknown(; desc = "Solver progress: ", dt = 1.0, showspeed = true)
    print("\n\n\n\n\n\n\n") # display solver progress after main progress

    ff = ODEFunction{false}((x,
        p,
        t) -> ode_func_train(x,
        (mgn, p, data, inputs, fields, meta,
            target_fields, target_dict, node_type,
            edge_features, senders, receivers, val_mask, data["inflow_mask"], strategy, pr),
        t))
    prob = ODEProblem(ff, u0, tspan, mgn.train_state.parameters)

    shoot_loss,
    shoot_gs = Zygote.withgradient(
        ps -> train_loss(strategy,
            (prob, ps, u0, nothing, gt, idx_mask,
                val_mask, mgn.n_norm, target_fields,
                [meta["features"][tf]["dim"] for tf in target_fields])),
        mgn.train_state.parameters)

    clear_log(7, false)
    return shoot_gs, shoot_loss
end

"""
    MultipleShooting(tstart, dt, tstop, interval_size, solver;
                     sense = GaussAdjoint(autojacvec = ZygoteVJP()),
                     continuity_term = 100, solargs...)

Similar to SolverTraining, but splits the trajectory into intervals that are solved independently and then combines them for loss calculation.
Useful if the network tends to get stuck in a local minimum if SolverTraining is used.

## Arguments
- `tstart`: Start time of the simulation.
- `dt`: Interval at which the simulation is saved.
- `tstop`: Stop time of the simulation.
- `interval_size`: Number of datapoints in each independently solved interval.
- `solver`: Solver that is used for simulating the system.

## Keyword Arguments
- `sense = GaussAdjoint(autojacvec = ZygoteVJP())`: The sensitivity algorithm used for calculating sensitivities. Checkpointing is enabled by default for adaptive solvers.
- `continuity_term = 100`: Factor by which the error between points of consecutive intervals is multiplied.
- `solargs`: Keyword arguments that are passed on to the solver.
"""
struct MultipleShooting <: SolverStrategy
    tstart::Float32
    dt::Float32
    tstop::Float32
    interval_size::Integer                  # Number of observations in one interval
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    continuity_term::Integer
    solargs::Any
end

function MultipleShooting(tstart::Float32,
        dt::Float32,
        tstop::Float32,
        interval_size::Integer,
        solver::OrdinaryDiffEqAlgorithm;
        sense::AbstractSensitivityAlgorithm = GaussAdjoint(;
            autojacvec = ZygoteVJP(), checkpointing = isadaptive(solver) ? true : false),
        continuity_term = 100,
        solargs...)
    MultipleShooting(
        tstart, dt, tstop, interval_size, solver, sense, continuity_term, solargs)
end

function train_loss(strategy::MultipleShooting, t::Tuple)
    prob, ps, _, callback_solve, gt, _, val_mask, _, _, _ = t

    tsteps = (strategy.tstart):(strategy.dt):(strategy.tstop)
    ranges = [i:min(length(tsteps), i + strategy.interval_size - 1)
              for i in 1:(strategy.interval_size - 1):(length(tsteps) - 1)]

    sols = [solve(
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
            ) for rg in ranges]
    group_predictions = typeof(gt) <: CuArray ? CuArray.(sols) : Array.(sols)

    if !all(successful_retcode.(sols))
        throw(ErrorException("solve call did not succeed."))
    end

    loss = 0
    for (i, rg) in enumerate(ranges)
        error = (gt[:, :, rg] - group_predictions[i]) .^ 2

        err_buf = Zygote.Buffer(error)
        err_buf[:, :, :] = error
        for i in axes(err_buf, 3)
            err_buf[:, :, i] = err_buf[:, :, i] .* val_mask
        end
        loss += mean(copy(err_buf))

        if i > 1
            loss += strategy.continuity_term *
                    sum(abs, group_predictions[i - 1][:, :, end] - gt[:, :, first(rg)])
        end
    end

    return loss
end

########################################################################
# Abstract type and functions for derivative based training strategies #
########################################################################

abstract type DerivativeStrategy <: TrainingStrategy end

function get_delta(strategy::DerivativeStrategy, trajectory_length::Integer)
    return strategy.window_size > 0 ? range(1, strategy.window_size) :
           range(1, trajectory_length - 1)
end

function init_train_step(::DerivativeStrategy, t::Tuple, ::Tuple)
    mgn, data, meta, fields, target_fields, node_type,
    edge_features, senders, receivers, datapoint, mask, val_mask = t

    target_quantities_change = vcat([mgn.o_norm[field]((data["target|" * field][
                                         :, :, datapoint] -
                                                        data[field][:, :, datapoint]) /
                                                       (data["dt"][datapoint + 1] -
                                                        data["dt"][datapoint]))
                                     for field in target_fields]...)

    graph = build_graph(
        mgn, data, fields, datapoint, node_type, edge_features, senders, receivers)

    return (mgn, graph, target_quantities_change, mask, val_mask)
end

function train_step(::DerivativeStrategy, t::Tuple)
    mgn, graph, target_quantities_change, mask, val_mask = t

    return GraphNetCore.step!(mgn, graph, target_quantities_change, mask, val_mask)
end

function validation_step(::DerivativeStrategy, t::Tuple)
    sim_interval = t[2]["dt"][1:(end - 1)]
    data_interval = 1:(length(sim_interval))

    return _validation_step(t, sim_interval, data_interval)
end

"""
    DerivativeTraining(; window_size = 0, random = true)

Compares the prediction of the system with the derivative from the data (via finite differences).
Useful for initial training of the system since it is faster than training with a solver.

## Keyword Arguments
- `window_size = 0`: Number of steps from each trajectory (starting at the beginning) that are used for training. If the number is zero then the whole trajectory is used.
- `random = true`: Whether the derivative samples should be shuffled before training.
"""
struct DerivativeTraining <: DerivativeStrategy
    window_size::Integer
    random::Bool
end
function DerivativeTraining(; window_size::Integer = 0, random = true)
    DerivativeTraining(window_size, random)
end
