#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Pkg; Pkg.develop(path = joinpath(@__DIR__, "../../MeshGraphNets.jl"))
using Documenter, MeshGraphNets, GraphNetCore

makedocs(sitename = "MeshGraphNets.jl",
        format = Documenter.HTML(
            sidebar_sitename = false,
            edit_link = nothing
        ),
        authors = "Julian Trommer, and contributors.",
        modules = [GraphNetCore],
        checkdocs = :exports,
        linkcheck = true,
        pages = [
            "Home" => "index.md",
            "Overview" => "overview.md",
            "Getting Started" => [
                "Training Data" => "training_data.md",
                "Training & Evaluation" => "train_eval.md"
            ],
            "Examples" => [
                "CylinderFlow" => "cylinder_flow.md"
            ],
            "Reference" => [
                "Training Strategies" => "strategies.md"
                "GraphNetCore.jl" => "graph_net_core.md"
            ]
        ]
)

deploydocs(repo = "github.com/una-auxme/MeshGraphNets.jl.git", devbranch = "main")