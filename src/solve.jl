#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import ProgressMeter: ProgressUnknown

import ChainRulesCore: @ignore_derivatives

"""
    rollout(solver, mgn, initial_state, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, inflow_mask, data, start, stop, dt, saves; show_progress = true)

Solves the ODEProblem of the MGN with the given solver.

## Arguments
- `solver`: Solver that is used for evaluating the system.
- `mgn`: [GraphNetwork](@ref) that should be evaluated.
- `initial_state`: Initial state of the system.
- `fields`: Node features of the MGN.
- `meta`: Metadata of the dataset.
- `target_fields`: Output features of the MGN.
- `target_dict`: Dictionary containing the output features and their dimensions as key-value pair.
- `node_type`: Onehot vector of the node types used for node features.
- `edge_features`: Array of edge features for each edge in the graph.
- `senders`: Vector of indices where each edge in the graph starts.
- `receivers`: Vector of indices where each edge in the graph ends.
- `val_mask`: Bitmask specifying which nodes should be updated.
- `inflow_mask`: Vector of indices of nodes that are defined as inflow nodes.
- `data`: Simulation data used for setting the inputs on the inflow nodes.
- `start`: Start time of the simulation.
- `stop`: Stop time of the simulation.
- `dt`: If set, the solver will use fixed timesteps.
- `saves`: Timesteps where the solution is saved at.

## Keyword Arguments
- `show_progress = true`: Whether a progress bar should be displayed.

## Returns
- Solution of the ODEProblem at the specified timesteps.
- Timesteps corresponding to the solution.
"""
function rollout(solver, mgn::GraphNetwork, data, fields, meta, target_fields,
        target_dict, node_type, edge_features, senders, receivers, val_mask,
        inflow_mask, start, stop, dt, saves, pr = nothing)
    interval = (start, stop)
    x0 = vcat([typeof(data[field]) <: AbstractArray ? data[field][:, :, 1] :
               data[field] for field in target_fields]...)
    inputs = Dict{String, AbstractArray}(
        [typeof(data[field]) <: AbstractArray ? (field, data[field][:, :, 1]) :
         (field, data[field]) for field in fields]
    )
    re = nothing
    if typeof(mgn.model) <: Flux.Chain
        mgn.ps, re = Flux.destructure(mgn.model)
    end
    prob = ODEProblem(ode_func_eval, x0, interval,
        (mgn, mgn.ps, re, data, inputs, fields, meta, target_fields,
            target_dict, node_type, edge_features, senders, receivers,
            val_mask, inflow_mask, saves[2] - saves[1], pr))
    if isnothing(dt)
        sol = solve(prob, solver; saveat = saves, tstops = saves)
    else
        sol = solve(prob, solver; adaptive = false, dt = dt, saveat = saves)
    end

    if !isnothing(pr)
        finish!(pr)
    end

    return sol.u, sol.t
end

"""
    ode_func_train(x, (mgn, ps, data, inputs, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, inflow_mask, strategy, pr), t)

Inner function for training the system via solver.

## Arguments
- `x`: Current state of the system.
- Tuple containing variables needed for a step of the ODE.
- `t`: Current timestep of the system.

The parameter tuple contains the following variables:
- `mgn`: [GraphNetwork](@ref) that should be evaluated.
- `ps`: Parameters of the network inside the MGN.
- `data`: Simulation data used for setting the inputs on the inflow nodes.
- `inputs`: Dictionary of the initial state without the target features.
- `fields`: Node features of the MGN.
- `meta`: Metadata of the dataset.
- `target_fields`: Output features of the MGN.
- `target_dict`: Dictionary containing the output features and their dimensions as key-value pair.
- `node_type`: Onehot vector of the node types used for node features.
- `edge_features`: Array of edge features for each edge in the graph.
- `senders`: Vector of indices where each edge in the graph starts.
- `receivers`: Vector of indices where each edge in the graph ends.
- `val_mask`: Bitmask specifying which nodes should be updated.
- `inflow_mask`: Vector of indices of nodes that are defined as inflow nodes.
- `strategy`: Training strategy used for training.
- `pr`: Progress bar for logging.

## Returns
- See [ode_step](@ref).
"""
function ode_func_train(x,
        (mgn, ps, re, data, inputs, fields, meta, target_fields, target_dict, node_type,
            edge_features, senders, receivers, val_mask, inflow_mask, strategy, pr),
        t)
    bx = Zygote.Buffer(x)
    bx[:, :] = x
    bx[inflow_mask] = vcat([data[field][:, :, floor(Int, t / strategy.dt) + 1]
                            for field in target_fields]...)[inflow_mask]

    return ode_step(bx,
        (mgn, ps, re, inputs, fields, meta, target_fields, target_dict,
            node_type, edge_features, senders, receivers, val_mask, pr),
        t)
end

"""
    ode_func_eval(x, (mgn, ps, data, inputs, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, inflow_mask, saves_dt, pr), t)

Inner function for evaluating the ODEProblem.

## Arguments
- `x`: Current state of the system.
- Tuple containing variables needed for a step of the ODE.
- `t`: Current timestep of the system.

The parameter tuple contains the following variables:
- `mgn`: [GraphNetwork](@ref) that should be evaluated.
- `ps`: Parameters of the network inside the MGN.
- `data`: Simulation data used for setting the inputs on the inflow nodes.
- `inputs`: Dictionary of the initial state without the target features.
- `fields`: Node features of the MGN.
- `meta`: Metadata of the dataset.
- `target_fields`: Output features of the MGN.
- `target_dict`: Dictionary containing the output features and their dimensions as key-value pair.
- `node_type`: Onehot vector of the node types used for node features.
- `edge_features`: Array of edge features for each edge in the graph.
- `senders`: Vector of indices where each edge in the graph starts.
- `receivers`: Vector of indices where each edge in the graph ends.
- `val_mask`: Bitmask specifying which nodes should be updated.
- `inflow_mask`: Vector of indices of nodes that are defined as inflow nodes.
- `saves_dt`: Timesteps where the input of the system is updated.
- `pr`: Progress bar for logging.

## Returns
- See [ode_step](@ref).
"""
function ode_func_eval(x,
        (mgn, ps, re, data, inputs, fields, meta, target_fields, target_dict, node_type,
            edge_features, senders, receivers, val_mask, inflow_mask, saves_dt, pr),
        t)
    x[inflow_mask] = vcat([data[field][:, :, floor(Int, t / saves_dt) + 1]
                           for field in target_fields]...)[inflow_mask]

    return ode_step(x,
        (mgn, ps, re, inputs, fields, meta, target_fields, target_dict,
            node_type, edge_features, senders, receivers, val_mask, pr),
        t)
end

"""
    ode_step(x, (mgn, ps, inputs, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, pr), t)

Performs a single step of the ODEProblem (see [ode_func_train](@ref) and [ode_func_eval](@ref)).

## Arguments
- `x`: Current state of the system.
- Tuple containing variables needed for a step of the ODE.
- `t`: Current timestep of the system.

The parameter tuple contains the following variables:
- `mgn`: [GraphNetwork](@ref) that should be evaluated.
- `ps`: Parameters of the network inside the MGN.
- `inputs`: Dictionary of the initial state without the target features.
- `fields`: Node features of the MGN.
- `meta`: Metadata of the dataset.
- `target_fields`: Output features of the MGN.
- `target_dict`: Dictionary containing the output features and their dimensions as key-value pair.
- `node_type`: Onehot vector of the node types used for node features.
- `edge_features`: Array of edge features for each edge in the graph.
- `senders`: Vector of indices where each edge in the graph starts.
- `receivers`: Vector of indices where each edge in the graph ends.
- `val_mask`: Bitmask specifying which nodes should be updated.
- `pr`: Progress bar for logging.

## Returns
- Output of the ODE at the current timestep.
"""
function ode_step(x,
        (mgn, ps, re, inputs, fields, meta, target_fields, target_dict,
            node_type, edge_features, senders, receivers, val_mask, pr),
        t)
    offset = 1
    for k in target_fields
        inputs[k] = x[offset:(offset + target_dict[k] - 1), :]
        offset += target_dict[k]
    end

    graph = build_graph(
        mgn, inputs, fields, 1, node_type, edge_features, senders, receivers)
    if isnothing(re)
        output, st = mgn.model(graph, ps, mgn.st)
        mgn.st = st
    else
        output = re(ps)(graph)
    end

    indices = [meta["features"][tf]["dim"] for tf in target_fields]

    buf = Zygote.Buffer(output)
    for i in eachindex(target_fields)
        buf[(sum(indices[1:(i - 1)]) + 1):sum(indices[1:i]), :] = inverse_data(
            mgn.o_norm[target_fields[i]],
            output[(sum(indices[1:(i - 1)]) + 1):sum(indices[1:i]), :])
    end

    @ignore_derivatives begin
        if !isnothing(pr)
            next!(pr; showvalues = [(:t, "$(t)")])
        end
    end

    return copy(buf) .* val_mask
end
