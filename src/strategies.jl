#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import DifferentialEquations: ODEFunction
import Optimization: AutoZygote, OptimizationFunction, OptimizationProblem
import SciMLBase: AbstractSensitivityAlgorithm
import SciMLSensitivity: InterpolatingAdjoint, ZygoteVJP

import Optimization: solve

abstract type TrainingStrategy end

function prepare_training(::TrainingStrategy) 
    return (nothing,)
end

function validation_step(::TrainingStrategy, t::Tuple, sim_interval, data_interval)

    mgn, data, meta, delta, solver, solver_dt, fields, node_type, edge_features, senders, receivers, val_mask, opt_state, step, avg_loss, cp_path, tmp_loss, rollout = t

    initial_state = Dict(
        [typeof(v) <: AbstractArray ? (k, v[:, :, 1]) : (k, v) for (k,v) in data]
    )

    target_dict = Dict{String, Int32}()
    for tf in meta["target_features"]
        target_dict[tf] = meta["features"][tf]["dim"]
    end

    gt = vcat([data[tf] for tf in meta["target_features"]]...)[:, :, data_interval]

    solution = rollout(solver, mgn, initial_state, fields, meta["target_features"], target_dict, node_type, edge_features, senders, receivers, val_mask, sim_interval[1], sim_interval[end], solver_dt, sim_interval; show_progress = false)
    prediction = cat(solution.u..., dims = 3)[:, :, data_interval]

    error = mean((prediction - gt) .^ 2; dims = 2)

    return sqrt(mean(error[:, 1, :])), gt, prediction
end

abstract type SolverStrategy <: TrainingStrategy end

function get_delta(::SolverStrategy, ::Integer)
    return 1
end

function init_train_step(::SolverStrategy, t::Tuple, ta::Tuple)

    mgn, data, meta, fields, target_fields, node_type, edge_features, senders, receivers, _, _, val_mask = t
    cb_solve = ta

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

    return (mgn, inputs, fields, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, u0, gt, cb_solve)
end

function train_step(strategy::SolverStrategy, t::Tuple)

    mgn, inputs, fields, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, u0, gt, cb_solve, opt = t

    pr = ProgressUnknown(showspeed = true)

    ff = ODEFunction((x, p, t) -> fast_step(x, (mgn, p, inputs, fields, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, pr), t))
    prob = ODEProblem(ff, u0, (strategy.tstart, strategy.tstop), mgn.ps)

    if !isnothing(opt)
        adtype = AutoZygote()
        optf = OptimizationFunction((x, p) -> train_loss(strategy, (prob, ps, u0, cb_solve, gt)), adtype)
        opt_prob = OptimizationProblem(optf, mgn.ps)
        result = solve(opt_prob, opt; maxiters = 300)

        return result
    else
        shoot_loss, back = Zygote.pullback(ps -> train_loss(strategy, (prob, ps, u0, cb_solve, gt)), mgn.ps)

        shoot_gs = back(one(shoot_loss))[1]

        return shoot_gs, shoot_loss
    end
end

function get_sim_interval(strategy::SolverStrategy, ::Tuple)
    return strategy.tstart:strategy.dt:strategy.tstop
end

function validation_step(strategy::SolverStrategy, t::Tuple)
    sim_interval = get_sim_interval(strategy, (nothing,))
    data_interval = 1:length(sim_interval)

    return validation_step(strategy, t, sim_interval, data_interval)
end

"""
    SingleShooting(tstart, dt, tstop, solver; sense = InterpolatingAdjoint(autojacvec = ZygoteVJP()), plot_progress = false, solargs...)

The default solver based training that is normally used for NeuralODEs.
Simulates the system from `tstart` to `tstop` and calculates the loss based on the difference between the prediction and the ground truth at the timesteps `tstart:dt:tstop`.

# Arguments
- `tstart`: Start time of the simulation.
- `dt`: Interval at which the simulation is saved.
- `tstop`: Stop time of the simulation.
- `solver`: The solver that is used for simulating the system.

# Keyword Arguments
- `sense`: The sensitivity algorithm that is used for caluclating the sensitivities.
- `plot_progress`: Whether the training progress is plotted or not.
- `solargs`: Keyword arguments that are passed on to the solver.
"""
struct SingleShooting <: SolverStrategy
    tstart::Float32
    dt::Float32
    tstop::Float32
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    plot_progress::Bool
    solargs
end

function SingleShooting(tstart::Float32, dt::Float32, tstop::Float32, solver::OrdinaryDiffEqAlgorithm; sense::AbstractSensitivityAlgorithm = InterpolatingAdjoint(autojacvec = ZygoteVJP()), plot_progress = false, solargs...)
    SingleShooting(tstart, dt, tstop, solver, sense, plot_progress, solargs)
end

function train_loss(strategy::SingleShooting, t::Tuple)

    prob, ps, u0, callback_solve, gt = t

    sol = solve(remake(prob; p = ps), strategy.solver; u0 = u0, saveat = strategy.tstart:strategy.dt:strategy.tstop, sensealg = strategy.sense, strategy.solargs...)

    pred = typeof(gt) <: CuArray ? CuArray(sol) : Array(sol)

    error = (gt[:, :, 1:size(pred, 3)] .- pred) .^ 2
    loss = mean(error)

    return loss
end

"""
    MultipleShooting (Prototype, in development)
"""
struct MultipleShooting <: SolverStrategy
    tstart::Float32
    dt::Float32
    tstop::Float32
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    interval_size::Integer                  # Number of observations in one interval
    continuity_term::Integer
    plot_progress::Bool
    solargs
end

function MultipleShooting(tstart::Float32, dt::Float32, tstop::Float32, solver::OrdinaryDiffEqAlgorithm, interval_size, continuity_term = 100; sense::AbstractSensitivityAlgorithm = InterpolatingAdjoint(autojacvec = ZygoteVJP()), plot_progress = false, solargs...)
    MultipleShooting(tstart, dt, tstop, solver, sense, interval_size, continuity_term, plot_progress, solargs)
end

function train_loss(strategy::MultipleShooting, t::Tuple)
    prob, ps, _, callback_solve, gt = t

    tsteps = strategy.tstart:strategy.dt:strategy.tstop
    ranges = [i:min(size(gt, 3), i + strategy.interval_size - 1) for i in 1:strategy.interval_size-1:size(gt, 3)-1]

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
    group_predictions = Array.(sols)

    retcodes = [sol.retcode for sol in sols]
    if any(retcodes .!= :Success)
        return Inf
    end

    loss = 0
    for (i, rg) in enumerate(ranges)
        u = cpu_device()(gt[:, :, rg])
        û = group_predictions[i]
        
        error = (u - û) .^ 2
        loss += mean(error)
        
        if i > 1
            loss += strategy.continuity_term * sum(abs, group_predictions[i - 1][:, :, end] - u[:, :, 1])
        end
    end

    return loss
end



abstract type CollocationStrategy <: TrainingStrategy end

function get_delta(strategy::CollocationStrategy, trajectory_length::Integer)
    return strategy.window_size > 0 ? strategy.window_size : trajectory_length - 1
end

function init_train_step(strategy::CollocationStrategy, t::Tuple, ::Tuple)

    mgn, data, meta, fields, target_fields, node_type, edge_features, senders, receivers, datapoint, mask, _ = t
    
    if strategy.eigeninformed
        ;
    end

    cur_quantities = vcat([data[field][:, :, datapoint] for field in target_fields]...)
    target_quantities = vcat([data["target|" * field][:, :, datapoint] for field in target_fields]...)
    if typeof(meta["dt"]) <: AbstractArray
        target_quantities_change = mgn.o_norm((target_quantities - cur_quantities) / (meta["dt"][datapoint + 1] - meta["dt"][datapoint]))
    else
        target_quantities_change = mgn.o_norm((target_quantities - cur_quantities) / Float32(meta["dt"]))
    end
    graph = build_graph(mgn, data, fields, datapoint, node_type, edge_features, senders, receivers, true)

    return (mgn, graph, target_quantities_change, mask)
end

function train_step(::CollocationStrategy, t::Tuple)

    mgn, graph, target_quantities_change, mask = t

    return step!(mgn, graph, target_quantities_change, mask, mse_reduce)
end

function get_sim_interval(::CollocationStrategy, t::Tuple)
    return t[2]["dt"][1]:t[2]["dt"][2]-t[2]["dt"][1]:t[2]["dt"][t[4]]
end

function validation_step(strategy::CollocationStrategy, t::Tuple)
    sim_interval = get_sim_interval(strategy, t)
    data_interval = 1:t[4]

    return validation_step(strategy, t, sim_interval, data_interval)
end

"""
    Collocation(;window_size = 0, eigeninformed = false, plot_progress = false)

Compares the prediction of the system with the derivative from the data (via finite differences).
Useful for initial training of the system since it it faster than training with a solver.

# Keyword Arguments
- `window_size`: Number of steps from each trajectory (starting at the beginning) that are used for training. If the number is zero then the whole trajectory is used.
- `eigeninformed`: Whether eigeninformed training is used at each training step or not. See [utils.jl](@ref) for reference.
- `plot_progress`: Whether the training progress is plotted or not.

"""
struct Collocation <: CollocationStrategy
    window_size::Integer
    eigeninformed::Bool
    plot_progress::Bool
end

function Collocation(;window_size::Integer = 0, eigeninformed = false, plot_progress = false)
    Collocation(window_size, eigeninformed, plot_progress)
end

"""
    RandomCollocation(;window_size = 0, eigeninformed = false, plot_progress = false)

Similar to Collocation, but timesteps are sampled randomly from the trajectory instead of sequential.

# Keyword Arguments
- `window_size`: Number of steps from each trajectory (starting at the beginning) that are used for training. If the number is zero then the whole trajectory is used.
- `eigeninformed`: Whether eigeninformed training is used at each training step or not. See [utils.jl](@ref) for reference.
- `plot_progress`: Whether the training progress is plotted or not.

"""
struct RandomCollocation <: CollocationStrategy
    window_size::Integer
    eigeninformed::Bool
    plot_progress::Bool
end

function RandomCollocation(;window_size::Integer = 0, eigeninformed = false, plot_progress = false)
    RandomCollocation(window_size, eigeninformed, plot_progress)
end

function prepare_training(strategy::RandomCollocation)
    samples = shuffle(1:strategy.window_size)

    return (samples,)
end

function init_train_step(::RandomCollocation, t::Tuple, ta::Tuple)

    mgn, data, meta, fields, target_fields, node_type, edge_features, senders, receivers, datapoint, mask, _ = t

    sample = ta[1][datapoint]
    
    if strategy.eigeninformed
        ;
    end

    cur_quantities = vcat([data[field][:, :, sample] for field in target_fields]...)
    target_quantities = vcat([data["target|" * field][:, :, sample] for field in target_fields]...)
    if typeof(meta["dt"]) <: AbstractArray
        target_quantities_change = mgn.o_norm((target_quantities - cur_quantities) / (meta["dt"][sample + 1] - meta["dt"][sample]))
    else
        target_quantities_change = mgn.o_norm((target_quantities - cur_quantities) / Float32(meta["dt"]))
    end
    graph = build_graph(mgn, data, fields, sample, node_type, edge_features, senders, receivers, true)

    return (mgn, graph, target_quantities_change, mask)
end
