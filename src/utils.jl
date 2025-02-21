#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Printf: @sprintf
import Statistics: stdm

function isnumber(meta, f)
    return getfield(Base, Symbol(uppercasefirst(meta["features"][f]["dtype"]))) ==
           Int32 ||
           getfield(Base, Symbol(uppercasefirst(meta["features"][f]["dtype"]))) ==
           Float32
end

"""
    data_minmax(path)

Calculates the minimum and maximum for each feature in the given part of the dataset.

## Arguments
- `path`: Path to the dataset files.

## Returns
- Minimum and maximum in training, validation and test set
"""
function data_minmax(path)
    args = Args()
    ds_train = Dataset(:train, path, args)
    ds_train.meta["types_updated"] = args.types_updated
    ds_train.meta["types_noisy"] = args.types_noisy
    ds_train.meta["noise_stddevs"] = args.noise_stddevs
    ds_train.meta["device"] = cpu_device()
    ds_train.meta["training_strategy"] = nothing
    train_loader = DataLoader(
        ds_train; batchsize = -1, buffer = false, parallel = true, shuffle = true)

    ds_valid = Dataset(:valid, path, args)
    ds_valid.meta["types_updated"] = args.types_updated
    ds_valid.meta["types_noisy"] = args.types_noisy
    ds_valid.meta["noise_stddevs"] = args.noise_stddevs
    ds_valid.meta["device"] = cpu_device()
    ds_valid.meta["training_strategy"] = nothing
    valid_loader = DataLoader(ds_valid; batchsize = -1, buffer = false, parallel = true)

    ds_test = Dataset(:test, path, args)
    ds_test.meta["device"] = cpu_device()
    ds_test.meta["training_strategy"] = nothing
    test_loader = DataLoader(ds_test; batchsize = -1, buffer = false, parallel = true)

    features = ds_train.meta["feature_names"]
    target_features = ds_train.meta["target_features"]

    result = Dict{String, Vector{Float32}}()
    for f in features
        if !haskey(ds_train.meta["features"][f], "onehot") && isnumber(ds_train.meta, f)
            result[f] = [Inf32, -Inf32]
        end
    end
    for tf in target_features
        if !haskey(ds_train.meta["features"][tf], "onehot") && isnumber(ds_train.meta, tf)
            result["target|$tf"] = [Inf32, -Inf32]
        end
    end

    function add_to_result(data)
        for f in features
            if !haskey(ds_train.meta["features"][f], "onehot") && isnumber(ds_train.meta, f)
                data_min = minimum(data[f])
                data_max = maximum(data[f])
                if data_min < result[f][1]
                    result[f][1] = data_min
                end
                if data_max > result[f][2]
                    result[f][2] = data_max
                end
            end
        end

        for tf in target_features
            if !haskey(ds_train.meta["features"][tf], "onehot") &&
               isnumber(ds_train.meta, tf)
                ddiff = data[tf][:, :, 2:end] - data[tf][:, :, 1:(end - 1)]
                dts = Float32.(data["dt"][2:end] - data["dt"][1:(end - 1)])
                for i in eachindex(dts)
                    ddiff[:, :, i] ./= dts[i]
                end
                ddiff_min = minimum(ddiff)
                ddiff_max = maximum(ddiff)
                if ddiff_min < result["target|$tf"][1]
                    result["target|$tf"][1] = ddiff_min
                end
                if ddiff_max > result[tf][2]
                    result["target|$tf"][2] = ddiff_max
                end
            end
        end
    end
    p = Progress(length(train_loader) + length(valid_loader) + length(test_loader);
        desc = "Calculating minmax-norm: ", dt = 1.0, barlen = 50)
    for data in train_loader
        add_to_result(data)
        next!(p)
    end

    for data in valid_loader
        add_to_result(data)
        next!(p)
    end

    for data in test_loader
        add_to_result(data)
        next!(p)
    end
    finish!(p)

    return result
end

"""
    data_meanstd(path)

Calculates the mean and standard deviation for each feature in the given part of the dataset.

## Arguments
- `path`: Path to the dataset files.

## Returns
- Mean and standard deviation in training, validation and test set
"""
function data_meanstd(path)
    args = Args()
    ds_train = Dataset(:train, path, args)
    ds_train.meta["types_updated"] = args.types_updated
    ds_train.meta["types_noisy"] = args.types_noisy
    ds_train.meta["noise_stddevs"] = args.noise_stddevs
    ds_train.meta["device"] = cpu_device()
    ds_train.meta["training_strategy"] = nothing
    train_loader = DataLoader(
        ds_train; batchsize = -1, buffer = false, parallel = true, shuffle = true)

    ds_valid = Dataset(:valid, path, args)
    ds_valid.meta["types_updated"] = args.types_updated
    ds_valid.meta["types_noisy"] = args.types_noisy
    ds_valid.meta["noise_stddevs"] = args.noise_stddevs
    ds_valid.meta["device"] = cpu_device()
    ds_valid.meta["training_strategy"] = nothing
    valid_loader = DataLoader(ds_valid; batchsize = -1, buffer = false, parallel = true)

    ds_test = Dataset(:test, path, args)
    ds_test.meta["device"] = cpu_device()
    ds_test.meta["training_strategy"] = nothing
    test_loader = DataLoader(ds_test; batchsize = -1, buffer = false, parallel = true)

    features = ds_train.meta["feature_names"]
    target_features = ds_train.meta["target_features"]

    result = Dict{String, Array{Float32, 2}}()
    for f in features
        if !haskey(ds_train.meta["features"][f], "onehot") && isnumber(ds_train.meta, f)
            result[f] = zeros(Float32, ds_train.meta["features"][f]["dim"], 0)
        end
    end
    for tf in target_features
        if isnumber(ds_train.meta, tf)
            result["target|$tf"] = zeros(Float32, ds_train.meta["features"][tf]["dim"], 0)
        end
    end

    function add_to_result(data)
        for f in features
            if !haskey(ds_train.meta["features"][f], "onehot") && isnumber(ds_train.meta, f)
                result[f] = cat(
                    result[f], [data[f][:, :, i] for i in axes(data[f], 3)]...; dims = 2)
            end
        end

        for tf in target_features
            if isnumber(ds_train.meta, tf)
                tf_data = data[tf][:, :, 2:end] - data[tf][:, :, 1:(end - 1)]
                result["target|$tf"] = cat(result["target|$tf"],
                    [tf_data[:, :, i] ./ Float32(data["dt"][i + 1] - data["dt"][i])
                     for i in axes(tf_data, 3)]...;
                    dims = 2)
            end
        end
    end

    p = Progress(length(train_loader) + length(valid_loader) + length(test_loader);
        desc = "Calculating meanstd-norm: ", dt = 1.0, barlen = 50)
    for data in train_loader
        add_to_result(data)
        next!(p)
    end
    for data in valid_loader
        add_to_result(data)
        next!(p)
    end
    for data in test_loader
        add_to_result(data)
        next!(p)
    end
    finish!(p)

    meanstd_dict = Dict{String, Any}()

    for k in keys(result)
        m = mean(result[k])
        s = stdm(result[k], m)
        meanstd_dict[k] = (m, s)
    end

    return meanstd_dict
end

"""
    li_to_ci(dims, li)

Helper function to project a LinearIndex onto the given dimensions as a CartesianIndex.

## Arguments
- `dims`: Dimensions to where the LinearIndex is projected onto.
- `li`: LinearIndex to convert.

## Returns
- Converted CartesianIndex.
"""
function li_to_ci(dims, li)
    ci = CartesianIndices(Tuple(dims))
    return ci[li]
end

"""
    ci_to_li(dims, ci)

Helper function to project a CartesianIndex onto the given dimension as a LinearIndex.

## Arguments
- `dims`: Dimensions to where the CartesianIndex is projected onto.
- `li`: CartesianIndex to convert.

## Returns
- Converted LinearIndex.
"""
function ci_to_li(dims, ci)
    li = LinearIndices(Tuple(dims))
    return li[ci]
end

"""
    li_to_ci(dims, li)

Helper function to proejct the given indices onto the given dimensions as a LinearIndex.

## Arguments
- `dims`: Dimensions to where the LinearIndex is projected onto.
- `idxs`: Indices to convert.

## Returns
- Converted LinearIndex.
"""
function dims_to_li(dims, idxs)
    li = LinearIndices(Tuple(dims))
    return li[idxs...]
end

"""
    clear_line(move_up = true)

Deletes the content of the current line in the terminal.

## Arguments
- `move_up`: Whether the cursor should be moved up one line before deleting.
"""
function clear_line(move_up = true)
    if move_up
        print("\u1b[1F")    # Move cursor up one line and to start of line
    end
    print("\u1b[2K")    # Delete text in current line without moving cursor back
    print("\u1b[1G")    # Move cursor to start of line
end

"""
    clear_log(lines, move_up = true)

Deletes the content of the given number of lines in the terminal.

## Arguments
- `lines`: Number of lines to be deleted.
- `move_up`: Whether the cursor should be moved up one line before deleting.
"""
function clear_log(lines::Integer, move_up = true)
    if lines <= 0
        throw(ArgumentError("Expected positive number of lines to clear, got: lines == $lines"))
    end
    clear_line(move_up)
    for _ in 1:lines
        clear_line()
    end
end
