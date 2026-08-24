# Copyright (c) 2026 Julian Trommer
# SPDX-License-Identifier: MIT
# See LICENSE for details.

@testset "utils.jl" begin
    @test MeshGraphNets.li_to_ci([2, 3, 4], 17) == CartesianIndex(1, 3, 3)
    @test MeshGraphNets.ci_to_li([2, 3, 4], CartesianIndex(1, 3, 3)) == 17
    @test MeshGraphNets.dims_to_li([2, 3, 4], [1, 3, 3]) == 17
    for dims in ([5], [2, 3], [2, 2, 2])
        for linear_index in 1:prod(dims)
            cartesian_index = MeshGraphNets.li_to_ci(dims, linear_index)
            @test MeshGraphNets.ci_to_li(dims, cartesian_index) == linear_index
            @test MeshGraphNets.dims_to_li(dims, Tuple(cartesian_index)) == linear_index
        end
    end

    numeric_meta = Dict("features" => Dict(
        "integer" => Dict("dtype" => "int32"),
        "float" => Dict("dtype" => "float32"),
        "boolean" => Dict("dtype" => "Bool")))
    @test MeshGraphNets.isnumber(numeric_meta, "integer")
    @test MeshGraphNets.isnumber(numeric_meta, "float")
    @test !MeshGraphNets.isnumber(numeric_meta, "boolean")

    output = mktemp() do _, io
        redirect_stdout(io) do
            MeshGraphNets.clear_line(false)
            MeshGraphNets.clear_log(2, false)
        end
        seekstart(io)
        read(io, String)
    end
    @test occursin("\e[2K", output)
    @test length(findall("\e[2K", output)) == 4
    @test_throws ArgumentError MeshGraphNets.clear_log(0)
    @test_throws ArgumentError MeshGraphNets.clear_log(-1)

    mktempdir() do dir
        write_fixture_dir(dir; all_splits = true)
        minmax = data_minmax(dir)
        @test minmax["velocity"] == Float32[-3, 8]
        @test minmax["target|velocity"] == Float32[-1, 2]
        @test !haskey(minmax, "node_type")
        @test !haskey(minmax, "flag")

        meanstd = data_meanstd(dir)
        all_velocity = reduce(hcat,
            [trajectory_values(node, 1).velocity for _ in 1:3 for node in 1:3])
        @test meanstd["velocity"][1] ≈ mean(all_velocity)
        @test meanstd["velocity"][2] ≈ std(all_velocity)
        @test meanstd["target|velocity"][1] ≈ 0.5f0
    end
end
