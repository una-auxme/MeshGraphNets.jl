# License

MeshGraphNets.jl uses a mixed-license structure because parts of the package
are adaptations of Google DeepMind's
[MeshGraphNets implementation](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets).

## Apache-2.0 files

The following files are adaptations of, or are derived from, the original
MeshGraphNets implementation:

- `src/MeshGraphNets.jl`
- `src/dataset.jl`
- `src/graph.jl`
- `src/solve.jl`
- `examples/cylinder_flow/cylinder_flow.jl`
- `examples/cylinder_flow/meta.json`

These files retain the applicable Google DeepMind copyright and attribution
notices, identify that they were modified for this Julia project, and are
licensed under the [Apache License,
Version 2.0](https://github.com/una-auxme/MeshGraphNets.jl/blob/main/LICENSE-APACHE).

## MIT files

All other files in the repository are licensed under the [MIT
License](https://github.com/una-auxme/MeshGraphNets.jl/blob/main/LICENSE),
unless a file states otherwise. This includes the project-specific training
strategies and utilities, documentation, tests, package and CI configuration,
and artwork.

## Attribution and modifications

The adapted files were translated to Julia and substantially modified for
MeshGraphNets.jl beginning in 2023. The repository
[`NOTICE`](https://github.com/una-auxme/MeshGraphNets.jl/blob/main/NOTICE)
contains the upstream attribution, modification information, and authoritative
file-level license boundary.

MeshGraphNets.jl is an independent project and is not endorsed by or
affiliated with Google DeepMind.
