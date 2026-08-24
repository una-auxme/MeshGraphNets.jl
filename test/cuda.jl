# Copyright (c) 2026 Julian Trommer
# SPDX-License-Identifier: MIT
# See LICENSE for details.

if hascuda
    @testset "CUDA integration and parity" begin
        CUDA.allowscalar(false)

        @testset "dataset, normalization, and graph placement" begin
            mktempdir() do dir
                write_fixture_dir(dir; times = REGULAR_TIMES)
                args = test_args(; use_cuda = true)
                dataset = configure_dataset!(
                    MeshGraphNets.Dataset(:train, dir, args); device = gpu)
                data = MLUtils.getobs(dataset, 1)
                @test data["velocity"] isa CuArray
                @test data["node_type"] isa CuArray
                @test data["senders"] isa CuArray
                @test data["receivers"] isa CuArray
                @test data["edge_features"] isa CuArray

                derivative_args = test_args(; use_cuda = true,
                    training_strategy = DerivativeTraining(; random = false))
                derivative_dataset = configure_dataset!(
                    MeshGraphNets.Dataset(:train, dir, derivative_args);
                    training_strategy = derivative_args.training_strategy, device = gpu)
                derivative_data = MLUtils.getobs(derivative_dataset, 1)
                @test derivative_data["velocity"] isa CuArray
                @test derivative_data["target|velocity"] isa CuArray
                @test size(derivative_data["velocity"], 3) ==
                      length(REGULAR_TIMES) - 1
                @test size(derivative_data["target|velocity"]) ==
                      size(derivative_data["velocity"])

                quantities, edge_norm,
                node_norms,
                output_norms = MeshGraphNets.calc_norms(dataset, gpu, args)
                @test quantities > 0
                @test edge_norm isa Union{NormaliserOffline, NormaliserOnline}
                @test all(norm -> norm isa Union{NormaliserOffline, NormaliserOnline},
                    values(node_norms))
                @test all(norm -> norm isa Union{NormaliserOffline, NormaliserOnline},
                    values(output_norms))
                @test edge_norm.acc_sum isa CuArray
                @test node_norms["velocity"].acc_sum isa CuArray
                @test output_norms["velocity"].acc_sum isa CuArray

                graph = MeshGraphNets.build_graph(test_mgn(gpu), data, ["velocity"],
                    1, data["node_type"], data["edge_features"], data["senders"],
                    data["receivers"])
                @test graph.nf isa CuArray
                @test graph.ef isa CuArray
            end
        end

        @testset "strategy loss and optimizer parity" begin
            gpu_strategies = (
                DerivativeTraining(; window_size = 2, random = false),
                SolverTraining(0.0f0, 0.1f0, 0.4f0, Tsit5();
                    abstol = 1.0f-6, reltol = 1.0f-6),
                SolverBatchTraining(0.0f0, 0.1f0, 0.4f0, 3, Tsit5();
                    abstol = 1.0f-6, reltol = 1.0f-6),
                MultipleShooting(0.0f0, 0.1f0, 0.4f0, 3, Tsit5();
                    continuity_term = 12, abstol = 1.0f-6, reltol = 1.0f-6))

            for strategy in gpu_strategies
                @testset "$(nameof(typeof(strategy)))" begin
                    cpu_result = strategy_step_result(strategy, cpu)
                    gpu_result = strategy_step_result(strategy, gpu)
                    test_strategy_step(gpu_result)

                    @test flat_native(gpu_result.gradient) isa CuArray
                    @test flat_native(gpu_result.train_state.parameters) isa CuArray
                    @test isapprox(gpu_result.loss, cpu_result.loss;
                        rtol = 5.0f-3, atol = 1.0f-5)
                    @test isapprox(flat_cpu(gpu_result.gradient),
                        flat_cpu(cpu_result.gradient); rtol = 1.0f-2, atol = 1.0f-4)
                    @test isapprox(gpu_result.parameters_after,
                        cpu_result.parameters_after; rtol = 1.0f-2, atol = 1.0f-4)
                    @test isapprox(gpu_result.validation_loss,
                        cpu_result.validation_loss; rtol = 5.0f-3, atol = 1.0f-5)
                end
            end
        end

        @testset "full GPU training and evaluation" begin
            run_training_integration(
                DerivativeTraining(; window_size = 1, random = false);
                use_cuda = true, evaluation_trajectories = 2)
            run_training_integration(
                SolverTraining(0.0f0, 0.1f0, 0.4f0, Tsit5();
                    abstol = 1.0f-5, reltol = 1.0f-5);
                use_cuda = true, evaluation_trajectories = 2)
        end
    end
end
