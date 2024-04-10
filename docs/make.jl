#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Pkg; Pkg.develop(path = joinpath(@__DIR__, "../../MeshGraphNets.jl"))
using Documenter, MeshGraphNets, GraphNetCore

import Documenter: GitHubActions

makedocs(sitename = "MeshGraphNets.jl",
        format = Documenter.HTML(
            sidebar_sitename = false,
            edit_link = nothing
        ),
        authors = "Julian Trommer, and contributors.",
        modules = [MeshGraphNets, GraphNetCore],
        checkdocs = :exports,
        linkcheck = false,
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
                "Training Strategies" => "strategies.md",
                "Utilities" => "util.md",
                "GraphNetCore.jl" => "graph_net_core.md"
            ]
        ]
)

function deployConfig()
    github_repository = get(ENV, "GITHUB_REPOSITORY", "")
    github_event_name = get(ENV, "GITHUB_EVENT_NAME", "")
    if github_event_name == "workflow_run"
        github_event_name = "push"
    end
    github_ref = get(ENV, "GITHUB_REF", "")
    return GitHubActions(github_repository, github_event_name, github_ref)
end

deploydocs(repo = "github.com/una-auxme/MeshGraphNets.jl.git", devbranch = "main", deploy_config = deployConfig())