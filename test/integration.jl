# Copyright (c) 2026 Julian Trommer
# SPDX-License-Identifier: MIT
# See LICENSE for details.

@testset "CPU training and evaluation integration" begin
    strategy_cases = (
        "DerivativeTraining" =>
            DerivativeTraining(; window_size = 1, random = false),
        "SolverTraining" =>
            SolverTraining(0.0f0, 0.1f0, 0.4f0, Tsit5();
                abstol = 1.0f-5, reltol = 1.0f-5),
        "SolverBatchTraining" =>
            SolverBatchTraining(0.0f0, 0.1f0, 0.4f0, 3, Tsit5();
                abstol = 1.0f-5, reltol = 1.0f-5),
        "MultipleShooting" =>
            MultipleShooting(0.0f0, 0.1f0, 0.4f0, 3, Tsit5();
                continuity_term = 12, abstol = 1.0f-5, reltol = 1.0f-5))

    for (name, strategy) in strategy_cases
        @testset "$name" begin
            run_training_integration(strategy;
                evaluation_trajectories = name == "DerivativeTraining" ? 11 : 0,
                additional_evaluation_paths = name == "DerivativeTraining")
        end
    end
end
