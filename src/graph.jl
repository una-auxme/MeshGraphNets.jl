#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Statistics: norm

"""
    create_base_graph(data, type_size, type_min, device)

Constructs the parts of the node features and edge features that do not change during one trajectory.

## Arguments
- `data`: Data from the dataset containing one trajectory.
- `type_size`: Depth of the node type matrix.
- `type_min`: Offset of the node type matrix.
- `device`: Device where the normaliser should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).

## Returns
- Onehot vector of the node types used for node features.
- Vector of indices where each edge in the graph starts.
- Vector of indices where each edge in the graph ends.
- Array of edge features for each edge in the graph.
"""
function create_base_graph(data, type_size, type_min, device::Function)
    node_type = one_hot(vec(data["node_type"][:, :, 1]), type_size - type_min + 1, 1 - type_min)

    if haskey(data, "cells")
        senders, receivers = triangles_to_edges(data["cells"][:, :, 1])
        if 0 in senders || 0 in receivers
            senders .+= 1
            receivers .+= 1
        end
        rel_vec = [data["mesh_pos"][:, senders[i], 1] - data["mesh_pos"][:, receivers[i], 1] for i in eachindex(senders)]
    elseif haskey(data, "edges")
        senders, receivers = parse_edges(data["edges"])
        if 0 in senders || 0 in receivers
            senders .+= 1
            receivers .+= 1
        end
        rel_vec = [data["mesh_pos"][:, senders[i], 1] - data["mesh_pos"][:, receivers[i], 1] for i in eachindex(senders)]
    else
        throw(MissingException("Data does not contain cell or edge information!"))
    end

    relative_mesh_pos = hcat(rel_vec...)

    edge_features = vcat(relative_mesh_pos, permutedims(map(norm, eachcol(relative_mesh_pos))))

    return device(node_type), device(senders), device(receivers), device(edge_features)
end

"""
    build_graph(mgn, data, fields, datapoint, node_type, edge_features, senders, receivers)

Constructs a [FeatureGraph](@ref) based on the given arguments.

## Arguments
- `mgn`: MGN from where normalisers for node & edge features are used.
- `data`: Data from the dataset containing one trajectory.
- `fields`: Node features of the MGN.
- `datapoint`: Current index of the data corresponding to the current timestep.
- `node_type`: Onehot vector of the node types used for node features.
- `edge_features`: Array of edge features for each edge in the graph.
- `senders`: Vector of indices where each edge in the graph starts.
- `receivers`: Vector of indices where each edge in the graph ends.

## Returns
- Resulting [FeatureGraph](@ref).
"""
function build_graph(mgn::GraphNetwork, data, fields, datapoint::Integer, node_type, edge_features::AbstractArray{Float32, 2}, senders::AbstractArray{T, 1}, receivers::AbstractArray{T, 1}) where {T <: Integer}
    # Removed generator in favor of removing Zygote.jl piracies (minimal increase of time and allocations)
    # Can be reverted once Enzyme.jl is compatible
    nt = mgn.n_norm["node_type"](node_type)
    nf = similar(nt, 0, size(nt, 2))
    for field in fields
        nf = vcat(nf, mgn.n_norm[field](data[field][:, :, min(size(data[field], 3), datapoint)]))
    end
    nf = vcat(nf, nt)
    return FeatureGraph(
        nf,
        # vcat(
        #     [mgn.n_norm[field](data[field][:, :, min(size(data[field], 3), datapoint)]) for field in fields]...,
        #     mgn.n_norm["node_type"](node_type)
        # ),
        mgn.e_norm(edge_features),
        senders,
        receivers
    )
end
