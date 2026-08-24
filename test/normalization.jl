# Copyright (c) 2026 Julian Trommer
# SPDX-License-Identifier: MIT
# See LICENSE for details.

@testset "normalization" begin
    meta = Dict{String, Any}(
        "dims" => [3],
        "edges" => Dict{String, Any}("data_min" => -2, "data_max" => 2),
        "feature_names" => ["mesh_pos", "cells", "flag", "node_type", "minimum",
            "meanstd", "online"],
        "target_features" => ["minimum", "meanstd", "online"],
        "features" => Dict{String, Any}(
            "flag" => Dict("dtype" => "Bool", "dim" => 1),
            "node_type" => Dict("dtype" => "int32", "dim" => 1, "onehot" => true,
                "data_min" => -1, "data_max" => 1, "target_min" => -2,
                "target_max" => 2),
            "minimum" => Dict("dtype" => "float32", "dim" => 2,
                "data_min" => [-2, -1], "data_max" => [2, 3],
                "target_min" => -1, "target_max" => 1, "output_min" => [-4, -3],
                "output_max" => [4, 3], "output_target_min" => -2,
                "output_target_max" => 2),
            "meanstd" => Dict("dtype" => "float32", "dim" => 1,
                "data_mean" => [2], "data_std" => [0.5],
                "output_mean" => [1], "output_std" => [2]),
            "online" => Dict("dtype" => "float32", "dim" => 2)))
    quantities, edge_norm,
    node_norms,
    output_norms = MeshGraphNets.calc_norms((; meta), cpu, test_args(;
        max_norm_steps = 7.0f0))
    @test quantities == 9
    @test edge_norm isa NormaliserOfflineMinMax
    @test node_norms["flag"] isa NormaliserOfflineMinMax
    @test node_norms["node_type"] isa NormaliserOfflineMinMax
    @test node_norms["minimum"] isa NormaliserOfflineMinMax
    @test node_norms["meanstd"] isa NormaliserOfflineMeanStd
    @test node_norms["online"] isa NormaliserOnline
    @test output_norms["minimum"] isa NormaliserOfflineMinMax
    @test output_norms["meanstd"] isa NormaliserOfflineMeanStd
    @test output_norms["online"].max_accumulations == 7.0f0

    mean_meta = deepcopy(meta)
    mean_meta["edges"] = Dict{String, Any}("data_mean" => 0, "data_std" => 2)
    @test MeshGraphNets.calc_norms((; meta = mean_meta), cpu, test_args())[2] isa
          NormaliserOfflineMeanStd
    online_meta = deepcopy(meta)
    delete!(online_meta, "edges")
    @test MeshGraphNets.calc_norms((; meta = online_meta), cpu, test_args())[2] isa
          NormaliserOnline
end
