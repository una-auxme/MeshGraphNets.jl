# Copyright (c) 2026 Julian Trommer
# SPDX-License-Identifier: MIT
# See LICENSE for details.

@testset "graph.jl" begin
    explicit = Dict{String, Any}(
        "node_type" => reshape(Int32[0, 1, 0], 1, 3, 1),
        "mesh_pos" => reshape(Float32[0, 2, 5], 1, 3, 1),
        "edges" => Int32[0 1; 1 2])
    MeshGraphNets.create_base_graph!(explicit, 1, 0, cpu)
    @test size(explicit["node_type"]) == (2, 3)
    @test all(explicit["senders"] .>= 1)
    @test all(explicit["receivers"] .>= 1)
    @test size(explicit["edge_features"]) == (2, 4)
    @test explicit["edge_features"][2, :] == abs.(explicit["edge_features"][1, :])

    cells = Dict{String, Any}(
        "node_type" => reshape(Int32[0, 0, 0], 1, 3, 1),
        "mesh_pos" => reshape(Float32[0, 1, 2], 1, 3, 1),
        "cells" => reshape(Int32[0, 1, 2], 3, 1, 1))
    MeshGraphNets.create_base_graph!(cells, 0, 0, cpu)
    @test length(cells["senders"]) == 6
    @test_throws KeyError MeshGraphNets.create_base_graph!(
        Dict{String, Any}(
            "node_type" => reshape(Int32[0], 1, 1, 1),
            "mesh_pos" => reshape(Float32[0], 1, 1, 1)),
        0,
        0,
        cpu)

    mgn = test_mgn(cpu)
    graph_data = Dict{String, Any}(
        "velocity" => reshape(Float32.(1:12), 2, 3, 2))
    node_type = Float32[1 0 1; 0 1 0]
    edge_features = Float32[1 2; 1 2]
    graph = MeshGraphNets.build_graph(
        mgn, graph_data, ["velocity"], 99, node_type, edge_features,
        Int32[1, 2], Int32[2, 3])
    @test graph.nf[1:2, :] == graph_data["velocity"][:, :, 2]
    @test graph.nf[3:4, :] == node_type
    @test graph.ef == edge_features
    @test graph.senders == Int32[1, 2]
    @test graph.receivers == Int32[2, 3]

    if hascuda
        gpu_data = Dict{String, Any}(
            "node_type" => reshape(Int32[0, 1, 0], 1, 3, 1),
            "mesh_pos" => reshape(Float32[0, 2, 5], 1, 3, 1),
            "edges" => Int32[0 1; 1 2])
        MeshGraphNets.create_base_graph!(gpu_data, 1, 0, gpu)
        @test cpu(gpu_data["node_type"]) == explicit["node_type"]
        @test cpu(gpu_data["senders"]) == explicit["senders"]
        @test cpu(gpu_data["edge_features"]) == explicit["edge_features"]
    end
end
