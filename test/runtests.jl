# Copyright (c) 2023 Julian Trommer
# SPDX-License-Identifier: MIT
# See LICENSE for details.

using MeshGraphNets
using Test
using Aqua

@testset "MeshGraphNets.jl" begin
    # TODO
    @testset "Aqua.jl" begin
        # Ambiguities in external packages
        @testset "Method ambiguity" begin
            Aqua.test_ambiguities([MeshGraphNets])
        end
        Aqua.test_all(MeshGraphNets; ambiguities = false)
    end
end
