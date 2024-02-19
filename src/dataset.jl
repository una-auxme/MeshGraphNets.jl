#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Distributions: Normal
import Random: MersenneTwister
import TFRecord: Example

import HDF5: h5open
import JLD2: jldopen
import JSON: parse
import Random: seed!, make_seed, shuffle
import TFRecord: read

include("strategies.jl")

mutable struct Dataset
    file::String
    file_valid::String
    meta::Dict{String, Any}
    ch::Union{Channel{Example}, Channel{Dict}}
    ch_valid::Union{Nothing, Channel{Example}, Channel{Dict}}
    data::Union{Nothing, Dict{Int, Dict}}
    data_valid::Union{Nothing, Dict{Int, Dict}}
    cs::Integer
    current::Integer
    current_valid::Integer
end

function parse_data(data::Example, meta::Dict{String, Any})
    out = Dict{String, AbstractArray}()
    for (key, value) in meta["features"]
        d = reinterpret(getfield(Base, Symbol(uppercasefirst(value["dtype"]))), data.features.feature[key].kind.value.value[])
        dims = Tuple(reverse(replace(value["shape"], -1 => abs(reduce(div, value["shape"], init=length(d))))))
        d = reshape(d, dims)
        if value["type"] == "static"
            d = repeat(d, 1, 1, meta["trajectory_length"])
        end
        out[key] = d
    end
    return out
end

function load_dataset(path::String, is_training::Bool)
    seed!(1234)
    
    filename = is_training ? "train" : "test"

    if isfile(joinpath(path, filename * ".tfrecord"))
        file = filename * ".tfrecord"
    elseif isfile(joinpath(path, filename * ".jld2"))
        file = filename * ".jld2"
    else
        file = filename * ".h5"
    end

    if endswith(file, "tfrecord")
        meta = parse(Base.read(joinpath(path, "meta.json"), String))

        ds = Dataset(
            joinpath(path, file),
            joinpath(path, "valid.tfrecord"),
            meta,
            read(joinpath(path, file); channel_size = 10),
            is_training ? read(joinpath(path, "valid.tfrecord"); channel_size = 10) : nothing,
            nothing,
            nothing,
            10,
            0,
            0
        )
    elseif endswith(file, "jld2") || endswith(file, "h5")
        meta = parse(Base.read(joinpath(path, "meta.json"), String))

        is_jld = endswith(file, "jld2")

        if is_jld
            sl = Base.SimpleLogger(Base.CoreLogging.Error)
            Base.with_logger(sl) do
                datafile = jldopen(joinpath(path, file), "r")
            end
        else
            datafile = h5open(joinpath(path, file), "r")
        end
        
        meta["n_trajectories"] = length(keys(datafile))
        data_keys = keys(datafile)
        close(datafile)
        if is_training
            data_keys_valid = nothing
            if is_jld
                jldopen(joinpath(path, "valid.jld2"), "r") do fv
                    meta["n_trajectories_valid"] = length(keys(fv))
                    data_keys_valid = keys(fv)
                end
            else
                h5open(joinpath(path, "valid.h5"), "r") do fv
                    meta["n_trajectories_valid"] = length(keys(fv))
                    data_keys_valid = keys(fv)
                end
            end
            ch_valid = read_h5!(joinpath(path, is_jld ? "valid.jld2" : "valid.h5"), data_keys_valid, meta, is_jld)
        else
            ch_valid = nothing
        end

        ch = read_h5!(joinpath(path, file), data_keys, meta, is_jld)
        

        ds = Dataset(
            joinpath(path, file),
            joinpath(path, is_jld ? "valid.jld2" : "valid.h5"),
            meta,
            ch,
            ch_valid,
            Dict{Int, Dict}(),
            Dict{Int, Dict}(),
            meta["trajectory_length"],
            0,
            0
        )
    else
        throw(ErrorException(path * " does not contain a $filename.tfrecord or a $filename.h5 file"))
    end

    return ds
end

function read_h5!(filename, data_keys, meta, is_jld)
    feature_names = meta["feature_names"]
    dims = meta["dims"]
    trajectory_length = meta["trajectory_length"]

    li = LinearIndices(Tuple(dims))
    global l = ReentrantLock()

    function get_traj(ch)
        for k in data_keys
            traj_dict = Dict{String, Any}()
            for fn in feature_names
                dim = haskey(meta["features"][fn], "dim") ? meta["features"][fn]["dim"] : 1
                if meta["features"][fn]["type"] == "static"
                    tl = 1
                elseif meta["features"][fn]["type"] == "dynamic"
                    tl = trajectory_length
                else
                    throw(ErrorException("feature type must be static or dynamic"))
                end
                traj_dict[fn] = zeros(getfield(Base, Symbol(uppercasefirst(meta["features"][fn]["dtype"]))), dim, prod(dims), tl)
                if haskey(meta["features"][fn], "has_ev") && meta["features"][fn]["has_ev"]
                    traj_dict[fn * ".ev"] = zeros(eltype(traj_dict[fn]), 2, prod(dims), tl)
                end
                
                if haskey(meta["features"][fn], "split") && meta["features"][fn]["split"]
                    rx = Regex(replace(replace(replace(meta["features"][fn]["key"], "[" => "\\["), "]" => "\\]"), "%d" => "\\d+") * "\\[\\d+\\]")
                else
                    rx = Regex(replace(replace(replace(meta["features"][fn]["key"], "[" => "\\["), "]" => "\\]"), "%d" => "\\d+"))
                end

                match_data = Dict()
                lock(l) do
                    if is_jld
                        file = jldopen(filename, "r")
                        traj = file[k]
                        rx_match = eachmatch.(rx, keys(traj))
                        deleteat!(rx_match, findall(isnothing, rx_match))
                        matches = unique(getfield.(rx_match, :match))
                        for m in matches
                            match_data[m] = traj[m]
                            if haskey(meta["features"][fn], "has_ev") && meta["features"][fn]["has_ev"]
                                match_data[m * ".ev"] = traj[m * ".ev"]
                            end
                        end
                    else
                        file = h5open(filename, "r")
                        traj = open_group(file, k)
                        rx_match = match.(rx, keys(traj))
                        deleteat!(rx_match, findall(isnothing, rx_match))
                        matches = unique(getfield.(rx_match, :match))
                        for m in matches
                            match_data[m] = Base.read(traj, m)
                            if haskey(meta["features"][fn], "has_ev") && meta["features"][fn]["has_ev"]
                                match_data[m * ".ev"] = Base.read(traj, m * ".ev")
                            end
                        end
                    end
                    close(file)
                end

                for (m, data) in match_data
                    if !occursin("]", m[1:end-1])
                        idx = Colon()
                        if haskey(meta["features"][fn], "split") && meta["features"][fn]["split"]
                            coord = Base.parse.(Int, split(split(m, r"(\[|\])")[2], ","))
                        else
                            coord = Colon()
                        end

                        fn_k = occursin(".ev", m) ? "$fn.ev" : fn

                        if meta["features"][fn]["type"] == "dynamic"
                            if ndims(data) == 2
                                traj_dict[fn_k][coord, :, :] = data[coord, 1:trajectory_length]
                            else
                                traj_dict[fn_k][coord, :, :] = data[1:trajectory_length]
                            end
                        else
                            traj_dict[fn_k][coord, :, :] .= data
                        end

                    else
                        idx = Base.parse.(Int, split(split(m, r"(\[|\])")[2], ","))
                        if haskey(meta["features"][fn], "split") && meta["features"][fn]["split"]
                            coord = Base.parse.(Int, split(split(m, r"(\[|\])")[4], ","))
                        else
                            coord = Colon()
                        end

                        fn_k = occursin(".ev", m) ? "$fn.ev" : fn

                        if meta["features"][fn]["type"] == "dynamic"
                            if ndims(data) == 2
                                traj_dict[fn_k][coord, dims_to_li(dims, idx), :] = data[coord, 1:trajectory_length]
                            else
                                traj_dict[fn_k][coord, dims_to_li(dims, idx), :] = data[1:trajectory_length]
                            end
                        else
                            traj_dict[fn_k][coord, dims_to_li(dims, idx), :] .= data
                        end
                    end
                end
            end
            
            lock(l) do
                if is_jld
                    file = jldopen(filename, "r")
                    traj_dict["dt"] = Float32.(file[k][meta["dt"]])
                else
                    file = h5open(filename, "r")
                    traj_dict["dt"] = Float32.(Base.read(file[k], meta["dt"]))
                end
                close(file)
            end

            edges = Vector{Vector{Int32}}() 
            if haskey(meta, "dims")
                ################################################
                # 1D-Meshes are connected in order by their id #
                ################################################
                if length(dims) == 1
                    for i in 1:dims[1]-1
                        push!(edges, [i, i+1])
                    end
                elseif length(dims) == 2
                    throw(ErrorException("2D-Meshes are not supported yet"))
                #################################################
                # 3D-Meshes are connected in order by their id, #
                # starting the count from z then y and then x   #
                #################################################
                elseif length(dims) == 3
                    dim_x, dim_y, dim_z = dims

                    function add_edge!(edges, x, y, z, cond, shift)
                        if cond
                            if traj_dict["node_type"][1, li[x + shift[1], y + shift[2], z + shift[3]], 1] != 1
                                push!(edges, [li[x, y, z], li[x + shift[1], y + shift[2], z + shift[3]]])
                            end
                        end
                    end

                    for x in 1:dim_x
                        for y in 1:dim_y
                            for z in 1:dim_z
                                if traj_dict["node_type"][1, li[x, y, z], 1] != 1
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
            else
                throw(ErrorException("Edges for custom meshes without specifying domain dimensions is not supported yet"))
            end
            traj_dict["edges"] = hcat(sort(edges)...)

            put!(ch, traj_dict)
        end
    end

    return Channel{Dict}(get_traj, 100; spawn = true)
end

function add_targets!(data, fields, device)
    new_data = deepcopy(data)
    for (key, value) in data
        if startswith(key, "target|")
            continue
        end
        if ndims(value) > 2 && size(value)[end] > 1
            if key == "mesh_pos" || key == "node_type" || key == "cells"
                new_data[key] = value[:, :, 1:end-1]
            else
                new_data[key] = device(value[:, :, 1:end-1])
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

    seed = make_seed(1234)
    rng = MersenneTwister(seed)

    for key in keys(data)
        if key == "edges" || length(data[key]) == 1 || size(data[key])[end] == 1
            continue
        end
        if typeof(ts) == Collocation && ts.window_size <= 0
            data[key] = data[key][:, :, shuffle(rng, 1:end)]
        end
        seed!(rng, seed)
    end
end

function take_trajectory!(dataset::Dataset, is_training::Bool)
    if typeof(dataset.ch) == Channel{Example}
        if is_training
            if !isready(dataset.ch)
                dataset.ch = read(dataset.file; channel_size = dataset.cs)
                dataset.current = 0
            end
            dataset.current += 1
            return take!(dataset.ch)
        else
            if !isready(dataset.ch_valid)
                dataset.ch_valid = read(dataset.file_valid; channel_size = dataset.cs)
                dataset.current_valid = 0
            end
            dataset.current_valid += 1
            return take!(dataset.ch_valid)
        end
    elseif typeof(dataset.ch) <: Channel{Dict}
        if is_training
            if length(keys(dataset.data)) < dataset.meta["n_trajectories"]
                traj = take!(dataset.ch)
                dataset.current += 1
                dataset.data[dataset.current] = deepcopy(traj)
                return traj
            else
                if dataset.current == length(keys(dataset.data))
                    dataset.current = 0
                end
                dataset.current += 1
                return deepcopy(dataset.data[dataset.current])
            end
        else
            if length(keys(dataset.data_valid)) < dataset.meta["n_trajectories_valid"]
                traj = take!(dataset.ch_valid)
                dataset.current_valid += 1
                dataset.data_valid[dataset.current_valid] = deepcopy(traj)
                return traj
            else
                if dataset.current_valid == length(keys(dataset.data_valid))
                    dataset.current_valid = 0
                end
                dataset.current_valid += 1
                return deepcopy(dataset.data_valid[dataset.current_valid])
            end
        end
    else
        throw(ErrorException("Wrong type of dataset: $(typeof(dataset.ch)); must be Channel{TFRecord.Example} or Channel{Dict}"))
    end
end

function next_trajectory!(dataset::Dataset, device::Function; types_noisy, noise_stddevs = nothing, ts = nothing, is_training = true)
    if typeof(dataset.ch) == Channel{Example}
        data = parse_data(take_trajectory!(dataset, is_training), dataset.meta)
        data["dt"] = [i * Float32(dataset.meta["dt"]) for i in 1:dataset.meta["trajectory_length"]]
        return prepare_trajectory!(data, dataset.meta, device; types_noisy, noise_stddevs, ts)
    elseif typeof(dataset.ch) <: Channel{Dict}
        data = take_trajectory!(dataset, is_training)
        meta = copy(dataset.meta)
        meta["dt"] = data["dt"]
        return prepare_trajectory!(data, meta, device; types_noisy, noise_stddevs, ts)
    else
        throw(ErrorException("The type of data is not supported: $(typeof(dataset.ch))"))
    end
end

function prepare_trajectory!(data, meta, device::Function; types_noisy, noise_stddevs, ts)
    if !isnothing(ts) && (typeof(ts) <: CollocationStrategy)
        add_targets!(data, meta["target_features"], device)
        preprocess!(data, meta["target_features"], noise_stddevs, types_noisy, ts, device)
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
