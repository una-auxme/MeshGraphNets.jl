![MeshGraphNets.jl Logo](https://github.com/una-auxme/MeshGraphNets.jl/blob/main/logo/meshgraphnetsjl_logo.png?raw=true "MeshGraphNets.jl Logo")

# MeshGraphNets.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://una-auxme.github.io/MeshGraphNets.jl/dev)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![Coverage](https://codecov.io/gh/una-auxme/MeshGraphNets.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/una-auxme/MeshGraphNets.jl)

[*MeshGraphNets.jl*](https://github.com/una-auxme/MeshGraphNets.jl) is a software package for the Julia programming language that provides an implementation of the [MeshGraphNets](https://arxiv.org/abs/2010.03409) framework by [Google DeepMind](https://deepmind.google/) for simulating mesh-based physical systems via graph neural networks:

> Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W. Battaglia. 2021. **Learning Mesh-Based Simulation with Graph Networks**. In International Conference on Learning Representations.

You can find the original implementation of MeshGraphNets in their GitHub repository [here](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets).

## Overview

[*MeshGraphNets.jl*](https://github.com/una-auxme/MeshGraphNets.jl) is designed to be part of the [SciML](https://sciml.ai/) ecosystem. The original framework was remodeled into a NeuralODE so that solvers from the [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) can be used to evaluate the system.

Base functionality for the `Encode-Process-Decode` architecture of `DeepMind` that *MeshGraphNets* is based on is provided in the core package [*GraphNetCore.jl*](https://github.com/una-auxme/GraphNetCore.jl).

## How to use MeshGraphNets.jl

Examples from the original paper are implemented in the [examples folder](https://github.com/una-auxme/MeshGraphNets.jl/tree/main/examples). You can also refer to the [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/overview) if you want to model your own system.

## Currently Supported

- Customizable input & output quantities
- 1D & 3D meshes
- Node features & Edge features
- Different strategies for training (see [here](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies))
- Evaluation of system with [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) solvers

## Citation

Coming soon!

## License

This repository uses a mixed-license structure. The Julia adaptations of
Google DeepMind's MeshGraphNets implementation and the CylinderFlow example
are licensed under Apache-2.0; project-specific code, documentation, tests,
configuration, and artwork are licensed under MIT. See the
[license documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/license/)
for the complete file-level boundary and the repository
[`NOTICE`](https://github.com/una-auxme/MeshGraphNets.jl/blob/main/NOTICE) for
upstream attribution and modification information.
