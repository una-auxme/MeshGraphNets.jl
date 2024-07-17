#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

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
        Aqua.test_all(MeshGraphNets; ambiguities = false, piracies = false)

        # Piracy due to Zygote pullback
        # @testset "Piracy" begin
        #     Aqua.test_piracies(MeshGraphNets)
        # end
    end
end
