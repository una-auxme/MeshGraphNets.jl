#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Distributions: Normal
import HDF5: Group
import Random: MersenneTwister

import HDF5: read_dataset
import JLD2: jldopen
import JSON: parse
import Random: seed!, shuffle

include("strategies.jl")

"""
    Dataset(meta, datafile, lock)

Data structure for the training, evaluation and test data inside a dataset.

## Arguments
- `meta`: Metadata of the dataset.
- `datafile`: Path of datafile.
- `lock`: Used to prevent simultaneous acces to the datafile.
"""
struct Dataset
    meta::Dict{String, Any}
    datafile::String
    lock::ReentrantLock
end

"""
    Dataset(datafile, metafile, args)

Creates a [Dataset](@ref) with the given files.

## Arguments
- `datafile`: Path of datafile.
- `metafile`: Path of metadata file.
- `args`: Arguments of the framework.
"""
function Dataset(datafile::String, metafile::String, args)
    if !isfile(datafile)
        throw(ArgumentError("Invalid datafile: $datafile"))
    elseif !endswith(datafile, ".jld2") || !endswith(datafile, ".h5")
        throw(ArgumentError("Invalid file format for datafile: $datafile. Possible formats are [.jld2, .h5]"))
    end
    if !isfile(metafile)
        throw(ArgumentError("Invalid metafile: $metafile"))
    elseif !endswith(metafile, ".json")
        throw(ArgumentError("Invalid file format for metafile: $metafile. Possible formats are [.json]"))
    end

    meta = parse(Base.read(metafile), String)
    keys_traj = keystraj(datafile)
    meta["n_trajectories"] = length(keys_traj)
    meta["keys_trajectories"] = keys_traj
    merge!(meta, Dict(String(key) => getfield(args, key) for key in propertynames(args)))

    Dataset(meta, datafile, ReentrantLock())
end

"""
    Dataset(split, path, args)

Creates a [Dataset](@ref) with the given split type from the files of the given path.

## Arguments
- `split`: Symbol representing the type of data; possible values are: `[:train, :valid, :test]`.
- `path`: Path to the data and metadata file.
- `args`: Arguments of the framework.
"""
function Dataset(split::Symbol, path::String, args)
    if split != :train && split != :valid && split != :test
        throw(ArgumentError("Invalid symbol for dataset: $split. Possible values are [:train, :valid, :test]"))
    end
    if !isfile(joinpath(path, "meta.json"))
        throw(ArgumentError("Metafile not found in path: $path. Check that your metafile is named \"meta.json\""))
    end

    meta = parse(Base.read(joinpath(path, "meta.json"), String))
    datafile = get_file(split, path)
    keys_traj = keystraj(datafile)
    meta["n_trajectories"] = length(keys_traj)
    meta["keys_trajectories"] = keys_traj
    merge!(meta, Dict(String(key) => getfield(args, key) for key in propertynames(args)))

    Dataset(meta, datafile, ReentrantLock())
end

function get_file(split::Symbol, path::String)
    filename = String(split)
    if isfile(joinpath(path, "$filename.jld2"))
        return joinpath(path, "$filename.jld2")
    elseif isfile(joinpath(path, "$filename.h5"))
        return joinpath(path, "$filename.h5")
    else
        throw(ArgumentError("No datafile for $filename was found at the given path: $path"))
    end
end

function keystraj(datafile::String)
    if endswith(datafile, ".jld2")
        file = jldopen(datafile, "r")
    elseif endswith(datafile, ".h5")
        file = h5open(datafile, "r")
    end
    keys_traj = keys(file)
    close(file)

    return keys_traj
end

MLUtils.numobs(ds::Dataset) = ds.meta["n_trajectories"]

function MLUtils.getobs!(buffer, ds::Dataset, idx)
    key = ds.meta["keys_trajectories"][idx]

    set_meta!(buffer, ds, key)

    for fn in ds.meta["feature_names"]
        alloc_traj!(buffer, ds, fn)

        match_data = match_keys(ds, key, fn)

        set_traj_data!(buffer, match_data, ds, fn)
    end
    set_edges!(buffer, ds, key)

    prepare_trajectory!(buffer, ds.meta, ds.meta["device"])

    buffer["mask"] = Int32.(findall(
        x -> x in ds.meta["types_updated"], buffer["node_type"][1, :, 1])) |>
                     ds.meta["device"]

    buffer["val_mask"] = Float32.(map(
        x -> x in ds.meta["types_updated"], buffer["node_type"][:, :, 1]))
    buffer["val_mask"] = repeat(
        buffer["val_mask"], sum(size(buffer[field], 1)
        for field in ds.meta["target_features"]), 1) |>
                         ds.meta["device"]

    buffer["inflow_mask"] = repeat(buffer["node_type"][:, :, 1] .== 1,
        sum(size(buffer[field], 1) for field in ds.meta["target_features"]), 1) |>
                            ds.meta["device"]

    create_base_graph!(buffer, ds.meta["features"]["node_type"]["data_max"],
        ds.meta["features"]["node_type"]["data_min"], ds.meta["device"])

    return buffer
end

function MLUtils.getobs(ds::Dataset, idx)
    traj_dict = Dict{String, Any}()

    getobs!(traj_dict, ds, idx)

    return traj_dict
end

function set_meta!(traj_dict::Dict{String, Any}, ds::Dataset, key::String)
    dt = ds.meta["dt"]
    tl = ds.meta["trajectory_length"]
    dims = ds.meta["dims"]

    if typeof(dt) <: AbstractFloat
        if tl == -1
            throw(ArgumentError("The metadata \"dt\" was specified as static and \"trajectory_length\" as -1 inside the metafile. You need to specify one of them as a vector with the length equal to the number of steps to infer the other one."))
        elseif typeof(tl) <: Integer
            dt = range(0.0, dt * (tl - 1); step = dt)
        elseif (typeof(tl)) == String
            lock(ds.lock) do
                if endswith(ds.datafile, ".jld2")
                    file = jldopen(ds.datafile, "r")
                    traj = file[key]
                    tl = file[key][tl]
                else
                    file = h5open(ds.datafile, "r")
                    traj = open_group(file, key)
                    tl = Base.read(traj, tl)
                end
                close(file)
            end
            dt = range(0.0, dt * (tl - 1); step = dt)
        else
            throw(ArgumentError("The metadata \"trajectory_length\" is invalid. Possible values are: [-1 (for inferring the length), Integer (for specifying the length), String (as key inside the datafile)]"))
        end
    elseif typeof(dt) == String
        lock(ds.lock) do
            if endswith(ds.datafile, ".jld2")
                file = jldopen(ds.datafile, "r")
                traj = file[key]
                dt = file[key][dt]
            else
                file = h5open(ds.datafile, "r")
                traj = open_group(file, key)
                dt = Base.read(traj, dt)
            end
            close(file)
        end
        if (typeof(tl)) == String
            lock(ds.lock) do
                if endswith(ds.datafile, ".jld2")
                    file = jldopen(ds.datafile, "r")
                    traj = file[key]
                    tl = file[key][tl]
                else
                    file = h5open(ds.datafile, "r")
                    traj = open_group(file, key)
                    tl = Base.read(traj, tl)
                end
                close(file)
            end
        elseif !(typeof(tl) <: Integer)
            throw(ArgumentError("The metadata \"trajectory_length\" is invalid. Possible values are: [-1 (for inferring the length), Integer (for specifying the length), String (as key inside the datafile)]"))
        end
        if length(dt) == 1
            dt = range(0.0, dt * (tl - 1); step = dt)
        end
    else
        throw(ArgumentError("The metadata \"dt\" is invalid. Possible values are: [Float (for specifying the static time delta), String (as key inside the datafile)]"))
    end

    if typeof(dims) == String
        lock(ds.lock) do
            if endswith(ds.datafile, ".jld2")
                file = jldopen(ds.datafile, "r")
                traj = file[key]
                dims = file[key][dims]
            else
                file = h5open(ds.datafile, "r")
                traj = open_group(file, key)
                dims = Base.read(traj, dims)
            end
            close(file)
        end
    end
    if typeof(dims) <: Integer
        if haskey(ds.meta, "n_nodes")
            n_nodes = ds.meta["n_nodes"]
            if typeof(n_nodes) == String
                lock(ds.lock) do
                    if endswith(ds.datafile, ".jld2")
                        file = jldopen(ds.datafile, "r")
                        traj = file[key]
                        n_nodes = file[key][n_nodes]
                    else
                        file = h5open(ds.datafile, "r")
                        traj = open_group(file, key)
                        n_nodes = Base.read(traj, n_nodes)
                    end
                    close(file)
                end
            elseif !(typeof(n_nodes) <: Integer)
                throw(ArgumentError("The metadata \"n_nodes\" is invalid. Possible values are: [Integer (for specifying the number of nodes), String (as key inside the datafile)]"))
            end
        else
            throw(ArgumentError("The metadata \"dims\" is specified as Integer but no metadata \"n_nodes\" was provided. The number of nodes can only be inferred from a vector of dimensions. Either provide the number of nodes or use a vector of static dimensions."))
        end
    elseif typeof(dims) <: AbstractArray && all(x -> typeof(x) <: Integer, dims)
        if any(x -> x == -1, dims)
            if haskey(ds.meta, "n_nodes")
                n_nodes = ds.meta["n_nodes"]
                if typeof(n_nodes) == String
                    lock(ds.lock) do
                        if endswith(ds.datafile, ".jld2")
                            file = jldopen(ds.datafile, "r")
                            traj = file[key]
                            n_nodes = file[key][n_nodes]
                        else
                            file = h5open(ds.datafile, "r")
                            traj = open_group(file, key)
                            n_nodes = Base.read(traj, n_nodes)
                        end
                        close(file)
                    end
                elseif !(typeof(n_nodes) <: Integer)
                    throw(ArgumentError("The metadata \"n_nodes\" is invalid. Possible values are: [Integer (for specifying the number of nodes), String (as key inside the datafile)]"))
                end
            else
                throw(ArgumentError("The metadata \"dims\" contains -1 (for inferring dimensions) but no metadata \"n_nodes\" was provided. The number of nodes can only be inferred from a vector of dimensions with positive values. Either provide the number of nodes or use a vector of static positive dimensions."))
            end
            if haskey(ds.meta, "dims_key")
                lock(ds.lock) do
                    if endswith(ds.datafile, ".jld2")
                        file = jldopen(ds.datafile, "r")
                        traj = file[key]
                        dims_file = file[key]["dims_key"]
                    else
                        file = h5open(ds.datafile, "r")
                        traj = open_group(file, key)
                        dims_file = Base.read(traj, "dims_key")
                    end
                    close(file)
                end
                if length(dims_file) != length(dims)
                    throw(ArgumentError("The size of the metadata \"dims\" vector is not equal the size of the dims inside the datafile: size(dims_meta) = $dims, size(dims_file) = $dims_file"))
                else
                    dims = dims_file
                end
            else
                throw(ArgumentError("The metadata \"dims\" contains -1 (for inferring dimensions) but no metadata \"dims_key\" for reading the dimensions from the datafile was provided."))
            end
        else
            n_nodes = prod(dims)
        end
    else
        throw(ArgumentError("The metadata \"dims\" is invalid. Possible values are: [Integer (for specifying the dimensions), Vector{Integer} (for specifying nodes in each dimension)]"))
    end

    traj_dict["dt"] = Float32.(dt)
    traj_dict["trajectory_length"] = tl
    traj_dict["n_nodes"] = n_nodes
    traj_dict["dims"] = dims
end

function alloc_traj!(traj_dict::Dict{String, Any}, ds::Dataset, fn::String)
    dim = haskey(ds.meta["features"][fn], "dim") ? ds.meta["features"][fn]["dim"] : 1
    if ds.meta["features"][fn]["type"] == "static"
        tl = 1
    elseif ds.meta["features"][fn]["type"] == "dynamic"
        tl = traj_dict["trajectory_length"]
    else
        throw(ArgumentError("feature type of feature \"$fn\" must be static or dynamic"))
    end
    if !haskey(traj_dict, fn)
        traj_dict[fn] = zeros(
            getfield(Base, Symbol(uppercasefirst(ds.meta["features"][fn]["dtype"]))),
            dim, traj_dict["n_nodes"], tl)
    end
    if haskey(ds.meta["features"][fn], "has_ev") && ds.meta["features"][fn]["has_ev"]
        if !haskey(traj_dict, fn * ".ev")
            traj_dict[fn * ".ev"] = zeros(
                eltype(traj_dict[fn]), 2, traj_dict["n_nodes"], tl)
        end
    end
end

function match_keys(ds::Dataset, key::String, fn::String)
    if haskey(ds.meta["features"][fn], "split") && ds.meta["features"][fn]["split"]
        rx = Regex(replace(
            replace(replace(ds.meta["features"][fn]["key"], "[" => "\\["),
                "]" => "\\]"),
            "%d" => "\\d+") * "\\[\\d+\\]")
    else
        rx = Regex(replace(
            replace(replace(ds.meta["features"][fn]["key"], "[" => "\\["),
                "]" => "\\]"),
            "%d" => "\\d+"))
    end

    match_data = Dict()

    lock(ds.lock) do
        if endswith(ds.datafile, ".jld2")
            file = jldopen(ds.datafile, "r")
            traj = file[key]
            rx_match = eachmatch.(rx, keys(traj))
            deleteat!(rx_match, findall(x -> length(collect(x)) == 0, rx_match))
            matches = unique(getfield.(getindex.(collect.(rx_match), 1), :match))
            for m in matches
                match_data[m] = traj[m]
                if haskey(ds.meta["features"][fn], "has_ev") &&
                   ds.meta["features"][fn]["has_ev"]
                    match_data[m * ".ev"] = traj[m * ".ev"]
                end
            end
        else
            file = h5open(ds.datafile, "r")
            traj = open_group(file, key)
            rx_match = eachmatch.(rx, keys(traj))
            deleteat!(rx_match, findall(x -> length(collect(x)) == 0, rx_match))
            matches = unique(getfield.(collect.(rx_match)[1], :match))
            for m in matches
                match_data[m] = Base.read(traj, m)
                if haskey(ds.meta["features"][fn], "has_ev") &&
                   ds.meta["features"][fn]["has_ev"]
                    match_data[m * ".ev"] = Base.read(traj, m * ".ev")
                end
            end
        end
        close(file)
    end

    return match_data
end

function set_traj_data!(traj_dict::Dict{String, Any}, match_data, ds::Dataset, fn::String)
    for (m, data) in match_data
        if !occursin("]", m[1:(end - 1)])
            idx = Colon()
            if haskey(ds.meta["features"][fn], "split") &&
               ds.meta["features"][fn]["split"]
                coord = Base.parse.(Int, split(split(m, r"(\[|\])")[2], ","))
            else
                coord = Colon()
            end

            fn_k = occursin(".ev", m) ? "$fn.ev" : fn

            if ds.meta["features"][fn]["type"] == "dynamic"
                if ndims(data) == 2
                    traj_dict[fn_k][coord, :, :] = data[
                        coord, 1:traj_dict["trajectory_length"]]
                else
                    traj_dict[fn_k][coord, :, :] = data[1:traj_dict["trajectory_length"]]
                end
            else
                traj_dict[fn_k][coord, :, :] .= data
            end

        else
            idx = Base.parse.(Int, split(split(m, r"(\[|\])")[2], ","))
            if haskey(ds.meta["features"][fn], "split") &&
               ds.meta["features"][fn]["split"]
                coord = Base.parse.(Int, split(split(m, r"(\[|\])")[4], ","))
            else
                coord = Colon()
            end

            fn_k = occursin(".ev", m) ? "$fn.ev" : fn

            if typeof(idx) <: AbstractArray
                if length(idx) > 1
                    idx_node = dims_to_li(traj_dict["dims"], idx)
                else
                    idx_node = idx
                end
            else
                idx_node = idx
            end
            if ds.meta["features"][fn]["type"] == "dynamic"
                if ndims(data) == 2
                    traj_dict[fn_k][coord, idx_node, :] = data[
                        coord, 1:traj_dict["trajectory_length"]]
                else
                    traj_dict[fn_k][coord, idx_node, :] = data[
                        1:traj_dict["trajectory_length"]]
                end
            else
                traj_dict[fn_k][coord, idx_node, :] .= data
            end
        end
    end
end

function set_edges!(traj_dict::Dict{String, Any}, ds::Dataset, key::String)
    lock(ds.lock) do
        if endswith(ds.datafile, ".jld2")
            file = jldopen(ds.datafile, "r")
            traj = file[key]
        elseif endswith(ds.datafile, ".h5")
            file = h5open(ds.datafile, "r")
            traj = open_group(file, key)
        end

        if haskey(ds.meta, "edges")
            if haskey(ds.meta["edges"], "type")
                edge_type = ds.meta["edges"]["type"]
                if edge_type == "cells"
                    if !haskey(ds.meta["edges"], "key")
                        throw(ArgumentError("The metadata \"type\" for metadata \"edges\" was defined as \"cells\", but no metadata \"key\" for the datafile was given."))
                    end
                    edge_key = ds.meta["edges"]["key"]
                    if !haskey(traj, edge_key)
                        throw(ArgumentError("The metadata \"key\" for metadata \"edges\", defined as \"$edge_key\", was not found in the datafile."))
                    end
                    if endswith(ds.datafile, ".jld2")
                        traj_dict["cells"] = traj[edge_key]
                    else
                        traj_dict["cells"] = Base.read(traj, edge_key)
                    end
                elseif edge_type == "dims"
                    traj_dict["edges"] = create_edges(
                        traj_dict["dims"], traj_dict["node_type"],
                        haskey(ds.meta, "no_edges_node_types") ?
                        ds.meta["no_edges_node_types"] : [])
                elseif edge_type == "custom"
                    if !haskey(ds.meta["edges"], "key")
                        throw(ArgumentError("The metadata \"type\" for metadata \"edges\" was defined as \"custom\", but no metadata \"key\" for the datafile was given."))
                    end
                    edge_key = ds.meta["edges"]["key"]
                    if !haskey(traj, edge_key)
                        throw(ArgumentError("The metadata \"key\" for metadata \"edges\", defined as \"$edge_key\", was not found in the datafile."))
                    end
                    if endswith(ds.datafile, ".jld2")
                        edges = traj[edge_key]
                    else
                        edges = Base.read(traj, edge_key)
                    end
                    traj_dict["edges"] = parse_custom_edges(edges, traj_dict["node_type"],
                        haskey(ds.meta, "no_edges_node_types") ?
                        ds.meta["no_edges_node_types"] : [],
                        haskey(ds.meta, "exclude_node_indices") ?
                        ds.meta["exclude_node_indices"] : [])
                else
                    throw(ArgumentError("The metadata \"type\" of metadata \"edges\" is invalid. Possible values are: [\"cells\" for cell-type edge structures, \"dims\" for fixed edges along the dimensions, \"custom\" for custom edges]"))
                end
            else
                throw(ArgumentError("The metadata \"edges\" does not specify an edge type with the metadata \"type\"."))
            end
        else
            throw(ArgumentError("The metadata \"edges\" was not provided."))
        end
        close(file)
    end
end

"""
    create_edges(dims, node_type, excluded_node_types)

Creates a two-dimensional array with the first dimension as the two connected nodes and the second dimensions as the number of edges.
Depending on the length of the given dimensions the edges are created differently:

## 1D

Nodes are connected via node IDs in ascending order, i.e. for four nodes with indices `[1, 2, 3, 4]`:

```
2×3 Matrix{Int64}:
 1  2  3
 2  3  4

1 --- 2 --- 3 --- 4
 ```

## 2D

2D is not supported yet.

## 3D

Nodes are connected along the dimensions based on the [LinearIndex](@ref) of the dimensional index, i.e.:

```
dims = [2, 2, 2]
3-element Vector{Int64}:
 2
 2
 2

li = LinearIndices(Tuple(dims))
2×2×2 LinearIndices{3, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}, Base.OneTo{Int64}}}:
[:, :, 1] =
 1  3
 2  4

[:, :, 2] =
 5  7
 6  8

  7 -------- 8
 /|         /|
3 -------- 4 |
| |        | |
| 5 - - - -| 6
|/         |/
1 -------- 2
```


## Arguments
- `dims`: Array with the dimensions of the mesh.
- `node_type`: Array of node types from the datafile.
- `excluded_node_types`: Vector of node types that should not be connected with edges.

## Returns
- Two-dimensional array of edges as pairs of node indices.
"""
function create_edges(dims, node_type, excluded_node_types)
    li = LinearIndices(Tuple(dims))
    edges = Vector{Vector{Int32}}()

    if length(dims) == 1
        for i in 1:(dims[1] - 1)
            push!(edges, [i, i + 1])
        end
    elseif length(dims) == 2
        throw(ArgumentError("2D-Meshes are not supported yet"))
    elseif length(dims) == 3
        dim_x, dim_y, dim_z = dims

        function add_edge!(edges, x, y, z, cond, shift)
            if cond
                if node_type[1, li[x + shift[1], y + shift[2], z + shift[3]], 1] ∉
                   excluded_node_types
                    push!(
                        edges, [li[x, y, z], li[x + shift[1], y + shift[2], z + shift[3]]])
                end
            end
        end

        for x in 1:dim_x
            for y in 1:dim_y
                for z in 1:dim_z
                    if node_type[1, li[x, y, z], 1] ∉ excluded_node_types
                        add_edge!(edges, x, y, z, x != dim_x, [1, 0, 0])
                        add_edge!(edges, x, y, z, y != dim_y, [0, 1, 0])
                        add_edge!(edges, x, y, z, z != dim_z, [0, 0, 1])
                    else
                        if [li[x, y, z], li[x, y, z]] ∉ edges
                            push!(edges, [li[x, y, z], li[x, y, z]])
                        end
                    end
                end
            end
        end
    end

    return hcat(sort(edges)...)
end

"""
    parse_custom_edges(edges, node_type, no_edges_node_types, exclude_node_indices)

Parses the edges that were read from the datafile. The format is a vector of pairs of node indices that represent edges.

## Arguments
- `edges`: Vector of pairs of node indices.
- `node_type`: Array of node types from the datafile.
- `excluded_node_types`: Vector of node types that should not be connected with edges.
- `exclude_node_indices`: Vector of node indices that should not be connected with edges.

## Returns
- Two-dimensional array of edges as pairs of node indices.
"""
function parse_custom_edges(edges, node_type, no_edges_node_types, exclude_node_indices)
    exclude_indices = findall(x -> x ∈ no_edges_node_types, node_type)
    exclude_indices = vcat(exclude_indices, exclude_node_indices)
    filtered_edges = filter(x -> x[1] ∉ exclude_indices && x[2] ∉ exclude_indices, edges)
    edge_vec = Vector{Vector{Int32}}()
    for edge in filtered_edges
        push!(edge_vec, [edge[1], edge[2]])
    end

    return hcat(sort(edge_vec)...)
end

"""
    add_targets!(data, fields, device)

Shifts the datapoints beginning from second index back in order to use them as ground truth data (used for derivative based strategies).

## Arguments
- `data`: Data from the dataset containing one trajectory.
- `fields`: Node features of the MGN.
- `device`: Device where the data should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).
"""
function add_targets!(data, fields, device)
    new_data = deepcopy(data)
    for (key, value) in data
        if startswith(key, "target|") || key == "dt"
            continue
        end
        if ndims(value) > 2 && size(value)[end] > 1
            if key == "mesh_pos" || key == "node_type" || key == "cells"
                new_data[key] = value[:, :, 1:(end - 1)]
            else
                new_data[key] = device(value[:, :, 1:(end - 1)])
            end
            if key in fields
                new_data["target|" * key] = device(value[:, :, 2:end])
            end
        end
    end
    for (key, value) in new_data
        data[key] = value
    end
end

"""
    preprocess!(data, noise_fields, noise_stddevs, types_noisy, ts, device)

Adds noise to the given features and shuffles the datapoints if a derivative based strategy is used.

## Arguments
- `data`: Data from the dataset containing one trajectory.
- `noise_fields`: Node features to which noise is added.
- `noise_stddevs`: Array of standard deviations of the noise, where the length is either one if broadcasted or equal to the length of features.
- `types_noisy`: Node types to which noise is added.
- `ts`: Training strategy that is used.
- `device`: Device where the data should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).
"""
function preprocess!(data, noise_fields, noise_stddevs, types_noisy, ts, device)
    if length(noise_stddevs) != 1 && length(noise_stddevs) != length(noise_fields)
        throw(DimensionMismatch("dimension of noise must be 1 or match noise fields: noise has dim $(size(noise_stddevs)), noise fields has dim $(size(noise_fields))"))
    end
    for (i, nf) in enumerate(noise_fields)
        d = Normal(0.0f0, length(noise_stddevs) > 1 ? noise_stddevs[i] : noise_stddevs[1])
        noise = rand(d, size(data[nf])) |> device

        mask = findall(x -> x ∉ types_noisy, data["node_type"][1, :, 1])
        noise[:, mask, :] .= 0
        data[nf] += noise
    end

    rng = MersenneTwister(1234)

    for key in keys(data)
        if key == "edges" || length(data[key]) == 1 || size(data[key])[end] == 1
            continue
        end
        if typeof(ts) <: DerivativeStrategy && ts.random
            if key != "dt"
                data[key] = data[key][repeat([:], ndims(data[key]) - 1)...,
                    shuffle(rng,
                        ts.window_size == 0 ? collect(1:end) : collect(1:(ts.window_size)))]
            end
        end
        seed!(rng, 1234)
    end
end

"""
    prepare_trajectory!(data, meta, device; types_noisy, noise_stddevs, ts)

Transfers the data to the given device and configures the data if a derivative based strategy is used.

## Arguments
- `data`: Data from the dataset containing one trajectory.
- `meta`: Metadata of the dataset.
- `device`: Device where the data should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).

## Keyword Arguments
- `types_noisy`: Node types to which noise is added.
- `noise_stddevs`: Array of standard deviations of the noise, where the length is either one if broadcasted or equal to the length of features.
- `ts`: Training strategy that is used.

## Returns
- Transfered data.
- Metadata of the dataset.
"""
function prepare_trajectory!(data, meta, device::Function)
    if !isnothing(meta["training_strategy"]) &&
       (typeof(meta["training_strategy"]) <: DerivativeStrategy)
        add_targets!(data, meta["target_features"], device)
        preprocess!(data, meta["target_features"], meta["noise_stddevs"],
            meta["types_noisy"], meta["training_strategy"], device)
        for field in meta["feature_names"]
            if field == "mesh_pos" || field == "node_type" || field == "cells" ||
               field in meta["target_features"]
                continue
            end
            data[field] = device(data[field])
        end
    else
        for field in meta["feature_names"]
            if field == "mesh_pos" || field == "node_type" || field == "cells"
                continue
            end
            data[field] = device(data[field])
        end
    end
    return data, meta
end
