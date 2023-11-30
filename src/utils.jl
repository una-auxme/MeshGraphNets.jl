#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Printf: @sprintf

"""
    li_to_ci(dims, li)

Helper function to project a LinearIndex onto the given dimensions as a CartesianIndex.

# Arguments
- `dims`: Dimensions to where the LinearIndex is projected onto.
- `li`: The LinearIndex to convert.

# Returns
- The converted CartesianIndex.
"""
function li_to_ci(dims, li)
    ci = CartesianIndices(Tuple(dims))
    return ci[li]
end

"""
    ci_to_li(dims, ci)

Helper function to project a CartesianIndex onto the given dimension as a LinearIndex.

# Arguments
- `dims`: Dimensions to where the CartesianIndex is projected onto.
- `li`: The CartesianIndex to convert.

# Returns
- The converted LinearIndex.
"""
function ci_to_li(dims, ci)
    li = LinearIndices(Tuple(dims))
    return li[ci]
end

"""
    li_to_ci(dims, li)

Helper function to proejct the given indices onto the given dimensions as a LinearIndex.

# Arguments
- `dims`: Dimensions to where the LinearIndex is projected onto.
- `idxs`: The indices to convert.

# Returns
- The converted LinearIndex.
"""
function dims_to_li(dims, idxs)
    li = LinearIndices(Tuple(dims))
    return li[idxs...]
end

function clear_line(move_up = true)
    if move_up
        print("\u1b[1F")
        print("\u1b[2K")
    else
        print("\u1b[1G")
        print("\u1b[2K")
    end
end

function clear_log(lines::Integer, move_up = true)
    for _ in 1:lines
        clear_line(move_up)
    end
end
