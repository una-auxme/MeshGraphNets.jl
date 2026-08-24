# Copyright (c) 2026 Julian Trommer
# SPDX-License-Identifier: MIT
# See LICENSE for details.

@testset "strategies.jl" begin
    unknown = UnknownTrainingStrategy()
    @test MeshGraphNets.prepare_training(unknown) == (nothing,)
    @test_throws ArgumentError MeshGraphNets.get_delta(unknown, 4)
    @test_throws ArgumentError MeshGraphNets.init_train_step(unknown, (), ())
    @test_throws ArgumentError MeshGraphNets.train_step(unknown, ())
    @test_throws ArgumentError MeshGraphNets.validation_step(unknown, ())

    derivative = DerivativeTraining(; window_size = 2, random = false)
    @test collect(MeshGraphNets.get_delta(derivative, 6)) == [1, 2]
    @test collect(MeshGraphNets.get_delta(DerivativeTraining(), 4)) == [1, 2, 3]

    solver = Tsit5()
    solver_strategy = SolverTraining(0.0f0, 0.1f0, 0.4f0, solver; abstol = 1.0f-6)
    batch_strategy = SolverBatchTraining(0.0f0, 0.1f0, 0.4f0, 3, solver)
    shooting_strategy = MultipleShooting(0.0f0, 0.1f0, 0.4f0, 3, solver;
        continuity_term = 12)
    @test MeshGraphNets.get_delta(solver_strategy, 99) == 1
    @test MeshGraphNets.get_delta(batch_strategy, 99) == [1:3, 3:5]
    @test solver_strategy.solargs[:abstol] == 1.0f-6
    @test shooting_strategy.continuity_term == 12

    mgn = test_mgn(cpu)
    data = Dict{String, Any}(
        "velocity" => reshape(Float32.(1:12), 2, 3, 2),
        "target|velocity" => reshape(Float32.(3:14), 2, 3, 2),
        "dt" => Float32[0, 0.5, 1.5])
    node_type = Float32[1 0 1; 0 1 0]
    tuple = (mgn, data,
        Dict("target_features" => ["velocity"],
            "features" => Dict("velocity" => Dict("dim" => 2))),
        ["velocity"], ["velocity"], node_type, Float32[1 2; 1 2],
        Int32[1, 2], Int32[2, 3], 1, Int32[1, 3], ones(Float32, 2, 3))
    initialized = MeshGraphNets.init_train_step(derivative, tuple, (nothing,))
    @test initialized[3] ==
          (data["target|velocity"][:, :, 1] - data["velocity"][:, :, 1]) / 0.5f0

    solver_initialized = MeshGraphNets.init_train_step(solver_strategy, tuple, ())
    @test solver_initialized[14] == data["velocity"][:, :, 1]
    @test solver_initialized[15] == data["velocity"]
    batch_tuple = Base.setindex(tuple, 1:2, 10)
    batch_initialized = MeshGraphNets.init_train_step(batch_strategy, batch_tuple, ())
    @test batch_initialized[end] == (0.0f0, 0.1f0)

    validation_data = Dict{String, Any}(
        "velocity" => cat(Float32[1 2 3; 4 5 6],
            exp(0.1f0) .* Float32[1 2 3; 4 5 6],
            exp(0.2f0) .* Float32[1 2 3; 4 5 6]; dims = 3),
        "dt" => Float32[0, 0.1, 0.2])
    validation_times = copy(validation_data["dt"])
    validation_tuple = (mgn, validation_data,
        Dict("target_features" => ["velocity"],
            "features" => Dict("velocity" => Dict("dim" => 2))),
        2, Tsit5(), nothing, ["velocity"], node_type, Float32[1 2; 1 2],
        Int32[1, 2], Int32[2, 3], Int32[1, 2, 3], ones(Float32, 2, 3),
        Int32[], nothing)
    @test isfinite(MeshGraphNets.validation_step(derivative, validation_tuple))
    @test validation_data["dt"] == validation_times

    @testset "loss, gradient, optimizer, and validation" begin
        adaptive_strategies = (
            DerivativeTraining(; window_size = 2, random = false),
            SolverTraining(0.0f0, 0.1f0, 0.4f0, Tsit5();
                abstol = 1.0f-6, reltol = 1.0f-6),
            SolverBatchTraining(0.0f0, 0.1f0, 0.4f0, 3, Tsit5();
                abstol = 1.0f-6, reltol = 1.0f-6),
            MultipleShooting(0.0f0, 0.1f0, 0.4f0, 3, Tsit5();
                continuity_term = 12, abstol = 1.0f-6, reltol = 1.0f-6))

        for strategy in adaptive_strategies
            @testset "$(nameof(typeof(strategy)))" begin
                result = strategy_step_result(strategy, cpu)
                test_strategy_step(result)
                if strategy isa SolverTraining || strategy isa MultipleShooting
                    @test result.initialized[14] ==
                          result.fixture.data["velocity"][:, :, 1]
                    @test result.initialized[15] == result.fixture.data["velocity"]
                elseif strategy isa SolverBatchTraining
                    @test result.initialized[end] == (0.0f0, 0.2f0)
                    @test size(result.initialized[15], 3) == 3
                end
            end
        end

        fixed_strategy = SolverBatchTraining(
            0.0f0, 0.1f0, 0.4f0, 3, Euler(); adaptive = false, dt = 0.01f0)
        fixed_result = strategy_step_result(fixed_strategy, cpu)
        test_strategy_step(fixed_result)

        no_continuity = MultipleShooting(
            0.0f0, 0.1f0, 0.4f0, 3, Tsit5(); continuity_term = 0)
        with_continuity = MultipleShooting(
            0.0f0, 0.1f0, 0.4f0, 3, Tsit5(); continuity_term = 12)
        fixture = strategy_fixture(cpu)
        initialized = MeshGraphNets.init_train_step(
            no_continuity, fixture.tuple, MeshGraphNets.prepare_training(no_continuity))
        _,
        loss_without_continuity = MeshGraphNets.train_step(no_continuity, initialized)
        _, loss_with_continuity = MeshGraphNets.train_step(with_continuity, initialized)
        @test isfinite(loss_without_continuity)
        @test isfinite(loss_with_continuity)
        @test loss_with_continuity > loss_without_continuity
    end
end
