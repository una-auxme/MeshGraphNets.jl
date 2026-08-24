# Copyright (c) 2026 Julian Trommer
# SPDX-License-Identifier: MIT
# See LICENSE for details.

@testset "solve.jl" begin
    mgn = test_mgn(cpu)
    x = Float32[1 2 3; 4 5 6]
    inputs = Dict{String, AbstractArray}("velocity" => copy(x))
    meta = Dict("features" => Dict("velocity" => Dict("dim" => 2)))
    node_type = Float32[1 0 1; 0 1 0]
    edge_features = Float32[1 2; 1 2]
    parameters = (mgn, mgn.train_state.parameters, inputs, ["velocity"], meta,
        ["velocity"], Dict("velocity" => Int32(2)), node_type, edge_features,
        Int32[1, 2], Int32[2, 3], Float32[1 0 1; 1 0 1], nothing)
    derivative = MeshGraphNets.ode_step(x, parameters, 0.0f0)
    @test derivative == Float32[1 0 3; 4 0 6]

    time_data = cat(x, x .+ 10, x .+ 20; dims = 3)
    eval_data = Dict{String, Any}("velocity" => time_data)
    eval_parameters = (mgn, mgn.train_state.parameters, eval_data,
        Dict{String, AbstractArray}("velocity" => copy(x)), ["velocity"], meta,
        ["velocity"], Dict("velocity" => Int32(2)), node_type, edge_features,
        Int32[1, 2], Int32[2, 3], ones(Float32, 2, 3), Int32[2], 1.0f0, nothing)
    original_x = copy(x)
    eval_derivative = MeshGraphNets.ode_func_eval(x, eval_parameters, 1.0f0)
    @test x == original_x
    @test eval_derivative[:, 2] == time_data[:, 2, 2]

    rollout_data = Dict{String, Any}("velocity" => reshape(copy(x), 2, 3, 1))
    saves = Float32[0, 0.1, 0.2]
    solution,
    solution_times = MeshGraphNets.rollout(Tsit5(), mgn, rollout_data,
        ["velocity"], meta, ["velocity"], Dict("velocity" => Int32(2)), node_type,
        edge_features, Int32[1, 2], Int32[2, 3], ones(Float32, 2, 3), Int32[],
        0.0f0, 0.2f0, nothing, saves)
    @test solution_times ≈ saves
    @test solution[1] == x
    @test solution[end] ≈ exp(0.2f0) .* x rtol = 1.0f-4

    fixed_solution,
    fixed_times = MeshGraphNets.rollout(Tsit5(), mgn, rollout_data,
        ["velocity"], meta, ["velocity"], Dict("velocity" => Int32(2)), node_type,
        edge_features, Int32[1, 2], Int32[2, 3], ones(Float32, 2, 3), Int32[],
        0.0f0, 0.2f0, 0.01f0, saves)
    @test fixed_times ≈ saves
    @test fixed_solution[end] ≈ solution[end] rtol = 1.0f-4

    if hascuda
        gpu_mgn = test_mgn(gpu)
        gpu_solution,
        gpu_times = MeshGraphNets.rollout(Tsit5(), gpu_mgn,
            Dict{String, Any}("velocity" => gpu(reshape(copy(x), 2, 3, 1))),
            ["velocity"], meta, ["velocity"], Dict("velocity" => Int32(2)),
            gpu(node_type), gpu(edge_features), gpu(Int32[1, 2]), gpu(Int32[2, 3]),
            gpu(ones(Float32, 2, 3)), gpu(Int32[]), 0.0f0, 0.2f0, nothing, saves)
        @test gpu_times ≈ saves
        @test cpu(gpu_solution[end]) ≈ solution[end] rtol = 1.0f-4

        gpu_fixed_solution,
        gpu_fixed_times = MeshGraphNets.rollout(Tsit5(), gpu_mgn,
            Dict{String, Any}("velocity" => gpu(reshape(copy(x), 2, 3, 1))),
            ["velocity"], meta, ["velocity"], Dict("velocity" => Int32(2)),
            gpu(node_type), gpu(edge_features), gpu(Int32[1, 2]), gpu(Int32[2, 3]),
            gpu(ones(Float32, 2, 3)), gpu(Int32[]), 0.0f0, 0.2f0, 0.01f0, saves)
        @test gpu_fixed_times ≈ fixed_times
        @test cpu(gpu_fixed_solution[end]) ≈ fixed_solution[end] rtol = 1.0f-4
    end
end
