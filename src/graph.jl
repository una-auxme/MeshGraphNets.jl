#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Statistics: norm

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

function build_graph(mgn::GraphNetwork, data, fields, datapoint::Integer, node_type, edge_features::AbstractArray{Float32, 2}, senders::AbstractArray{T, 1}, receivers::AbstractArray{T, 1}, is_training::Bool) where {T <: Integer}
    return build_graph(mgn.n_norm, mgn.e_norm, data, fields, datapoint, node_type, edge_features, senders, receivers, is_training)
end

function build_graph(mgn::GraphNetwork, data, fields, node_type, edge_features::AbstractArray{Float32, 2}, senders::AbstractArray{T, 1}, receivers::AbstractArray{T, 1}, is_training::Bool) where {T <: Integer}
    return build_graph(mgn.n_norm, mgn.e_norm, data, fields, 1, node_type, edge_features, senders, receivers, is_training)
end

function build_graph(n_norm, e_norm, data, fields, node_type, edge_features::AbstractArray{Float32, 2}, senders::AbstractArray{T, 1}, receivers::AbstractArray{T, 1}, is_training::Bool) where {T <: Integer}
    return build_graph(n_norm, e_norm, data, fields, 1, node_type, edge_features, senders, receivers, is_training)
end

function build_graph(n_norm, e_norm, data, fields, datapoint, node_type, edge_features::AbstractArray{Float32, 2}, senders::AbstractArray{T, 1}, receivers::AbstractArray{T, 1}, is_training::Bool) where {T <: Integer}
    return FeatureGraph(
        vcat(
            [n_norm[field](data[field][:, :, min(size(data[field], 3), datapoint)], is_training) for field in fields]...,
            n_norm["node_type"](node_type, is_training)
        ),
        e_norm(edge_features, is_training),
        senders,
        receivers
    )
end
