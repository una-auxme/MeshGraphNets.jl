# Copyright (c) 2023 Julian Trommer
# SPDX-License-Identifier: MIT
# See LICENSE for details.

using MeshGraphNets
using Test
using Aqua

using CUDA
using GraphNetCore
using HDF5
using JLD2
using JSON
using Lux
using MLUtils
using Optimisers
using OrdinaryDiffEq
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5
using Random
using Statistics

include("helpers.jl")

const hascuda = CUDA.has_cuda() && CUDA.functional()
const cpu = cpu_device()
const gpu = hascuda ? gpu_device() : nothing

if !hascuda
    @warn "CUDA is not functional; skipping GPU tests. Run Pkg.test() on a CUDA-enabled host for GPU verification."
end

@testset "MeshGraphNets.jl" begin
    include("aqua.jl")
    include("utils.jl")
    include("dataset.jl")
    include("graph.jl")
    include("normalization.jl")
    include("strategies.jl")
    include("solve.jl")
    include("integration.jl")
    include("cuda.jl")
end
