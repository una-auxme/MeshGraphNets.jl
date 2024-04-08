#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Printf: @sprintf

"""
    der_minmax(path)

Calculates the minimum and maximum derivative for each target feature in the given dataset.

## Arguments
- `path`: Path to the dataset files.

## Returns
- Minimum and maximum derivative in training, validation and test set.
"""
function der_minmax(path)
    result = der_minmax(path, true)
    result_test = der_minmax(path, false)

    for (k, v) in result_test
        if v[1] < result[k][1]
            result[k][1] = v[1]
        end
        if v[2] > result[k][2]
            result[k][2] = v[2]
        end
    end
    return result
end

"""
    der_minmax(path, is_training)

Calculates the minimum and maximum derivative for each target feature in the given part of the dataset.

## Arguments
- `path`: Path to the dataset files.
- `is_training`: Determines for which dataset the calculation should be done. True for train and validation set, false for test set.

## Returns
- Minimum and maximum derivative in the specified part of the dataset.
"""
function der_minmax(path, is_training)
    dataset = load_dataset(path, is_training)

    target_features = dataset.meta["target_features"]

    result = Dict(tf => [Inf32, -Inf32] for tf in target_features)

    n_traj = dataset.meta["n_trajectories"]
    
    for _ in 1:n_traj
        data, meta = next_trajectory!(dataset, cpu_device(); types_noisy = [], noise_stddevs = [], ts = nothing)
        dt =  Float32(meta["dt"][2] -  meta["dt"][1])
        for tf in target_features
            for i in 2:size(data[tf], 3)
                ddiff = (data[tf][:, :, i] - data[tf][:, :, i - 1]) ./ dt
                ddiff_min = minimum(ddiff)
                ddiff_max = maximum(ddiff)
                if ddiff_min < result[tf][1]
                    result[tf][1] = ddiff_min
                end
                if ddiff_max > result[tf][2]
                    result[tf][2] = ddiff_max
                end
            end
        end
    end

    if is_training
        n_traj_valid = dataset.meta["n_trajectories_valid"]
        for _ in 1:n_traj_valid
            data, meta = next_trajectory!(dataset, cpu_device(); types_noisy = [], noise_stddevs = [], ts = nothing, is_training = false)
            dt =  Float32(meta["dt"][2] -  meta["dt"][1])
            for tf in target_features
                for i in 2:size(data[tf], 3)
                    ddiff = (data[tf][:, :, i] - data[tf][:, :, i - 1]) ./ dt
                    ddiff_min = minimum(ddiff)
                    ddiff_max = maximum(ddiff)
                    if ddiff_min < result[tf][1]
                        result[tf][1] = ddiff_min
                    end
                    if ddiff_max > result[tf][2]
                        result[tf][2] = ddiff_max
                    end
                end
            end
        end
    end

    return result
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
    @assert lines > 0 "The number of lines in the terminal to be cleared must be greater than 0 : lines == $lines"
    clear_line(move_up)
    for _ in 1:lines
        clear_line()
    end
end

####################################################################
# Overwritten for differentiation over GraphNetwork and normaliser #
####################################################################

Zygote.accum(x::Base.RefValue{Any}, y::NamedTuple{(:model, :ps, :st, :e_norm, :n_norm, :o_norm)}) = Zygote.accum(x[], y)

Zygote.accum(x::NamedTuple{(:model, :ps, :st, :e_norm, :n_norm, :o_norm)}, y::Base.RefValue{Any}) = Zygote.accum(x, y[])

Zygote.accum(x::Base.RefValue{Any}, y::NamedTuple{(:data_min, :data_max, :target_min, :target_max)}) = Zygote.accum(x[], y)

Base.:+(x::Base.RefValue{Any}, y::NamedTuple{(:model, :ps, :st, :e_norm, :n_norm, :o_norm)}) = Zygote.accum(x[], y)

Base.:+(x::NamedTuple{(:model, :ps, :st, :e_norm, :n_norm, :o_norm)}, y::Base.RefValue{Any}) = Zygote.accum(x, y[])

Base.:+(x::Base.RefValue{Any}, y::NamedTuple{(:data_min, :data_max, :target_min, :target_max)}) = Zygote.accum(x[], y)