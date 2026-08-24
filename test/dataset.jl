# Copyright (c) 2026 Julian Trommer
# SPDX-License-Identifier: MIT
# See LICENSE for details.

@testset "dataset.jl" begin
    @testset "constructors and metadata" begin
        for format in (:jld2, :h5)
            mktempdir() do dir
                write_fixture_dir(dir; format)
                extension = format == :jld2 ? "jld2" : "h5"
                datafile = joinpath(dir, "train.$extension")
                metafile = joinpath(dir, "meta.json")
                args = test_args()
                explicit = MeshGraphNets.Dataset(datafile, metafile, args)
                split = MeshGraphNets.Dataset(:train, dir, args)
                @test explicit.datafile == datafile
                @test split.datafile == datafile
                @test MLUtils.numobs(explicit) == 1
                @test collect(MeshGraphNets.keystraj(datafile)) == ["trajectory_1"]

                configure_dataset!(explicit)
                trajectory = MeshGraphNets.load_traj(explicit, "trajectory_1")
                @test trajectory isa Dict{String, Any}
                @test trajectory["time"] == Float32[0, 0.5, 1.5, 3]
                @test trajectory["node[2].velocity"] == trajectory_values(2, 1).velocity
                velocity_matches = MeshGraphNets.match_keys(
                    explicit, trajectory, "velocity")
                @test length(velocity_matches) == 3
                @test velocity_matches["node[3].velocity"] ==
                      trajectory_values(3, 1).velocity
                metadata = Dict{String, Any}()
                MeshGraphNets.set_meta!(metadata, explicit, trajectory)
                @test metadata["dt"] == Float32[0, 0.5, 1.5, 3]
                @test metadata["trajectory_length"] == 4
                @test metadata["n_nodes"] == 3
                @test metadata["dims"] == [3]

                inferred_meta = deepcopy(explicit.meta)
                inferred_meta["dims"] = [-1]
                inferred_meta["n_nodes"] = "n_nodes"
                inferred_meta["dims_key"] = "actual_dims"
                inferred = MeshGraphNets.Dataset(
                    inferred_meta, datafile, ReentrantLock())
                inferred_values = Dict{String, Any}()
                MeshGraphNets.set_meta!(inferred_values, inferred, trajectory)
                @test inferred_values["dims"] == Int32[3]
                @test inferred_values["n_nodes"] == 3

                static_meta = deepcopy(explicit.meta)
                static_meta["dt"] = 0.25
                static_meta["trajectory_length"] = "trajectory_length"
                static_ds = MeshGraphNets.Dataset(static_meta, datafile, ReentrantLock())
                static_values = Dict{String, Any}()
                MeshGraphNets.set_meta!(static_values, static_ds, trajectory)
                @test static_values["dt"] == Float32[0, 0.25, 0.5, 0.75]

                integer_dims_meta = deepcopy(explicit.meta)
                integer_dims_meta["dims"] = 1
                integer_dims_meta["n_nodes"] = "n_nodes"
                integer_dims_ds = MeshGraphNets.Dataset(
                    integer_dims_meta, datafile, ReentrantLock())
                integer_dims_values = Dict{String, Any}()
                MeshGraphNets.set_meta!(
                    integer_dims_values, integer_dims_ds, trajectory)
                @test integer_dims_values["dims"] == 1
                @test integer_dims_values["n_nodes"] == 3

                string_dims_meta = deepcopy(explicit.meta)
                string_dims_meta["dims"] = "actual_dims"
                string_dims_ds = MeshGraphNets.Dataset(
                    string_dims_meta, datafile, ReentrantLock())
                string_dims_values = Dict{String, Any}()
                MeshGraphNets.set_meta!(
                    string_dims_values, string_dims_ds, trajectory)
                @test string_dims_values["dims"] == Int32[3]

                invalid_meta = deepcopy(explicit.meta)
                invalid_meta["dt"] = 0.1
                invalid_meta["trajectory_length"] = -1
                @test_throws ArgumentError MeshGraphNets.set_meta!(Dict{String, Any}(),
                    MeshGraphNets.Dataset(invalid_meta, datafile, ReentrantLock()),
                    trajectory)
                invalid_meta = deepcopy(explicit.meta)
                invalid_meta["dt"] = true
                @test_throws ArgumentError MeshGraphNets.set_meta!(Dict{String, Any}(),
                    MeshGraphNets.Dataset(invalid_meta, datafile, ReentrantLock()),
                    trajectory)
                invalid_meta = deepcopy(explicit.meta)
                invalid_meta["dims"] = 1
                delete!(invalid_meta, "n_nodes")
                @test_throws ArgumentError MeshGraphNets.set_meta!(Dict{String, Any}(),
                    MeshGraphNets.Dataset(invalid_meta, datafile, ReentrantLock()),
                    trajectory)
                invalid_meta = deepcopy(explicit.meta)
                invalid_meta["dims"] = [-1]
                delete!(invalid_meta, "n_nodes")
                @test_throws ArgumentError MeshGraphNets.set_meta!(Dict{String, Any}(),
                    MeshGraphNets.Dataset(invalid_meta, datafile, ReentrantLock()),
                    trajectory)
                invalid_meta = deepcopy(explicit.meta)
                invalid_meta["dims"] = [-1]
                invalid_meta["n_nodes"] = 3
                delete!(invalid_meta, "dims_key")
                @test_throws ArgumentError MeshGraphNets.set_meta!(Dict{String, Any}(),
                    MeshGraphNets.Dataset(invalid_meta, datafile, ReentrantLock()),
                    trajectory)
            end
        end

        mktempdir() do dir
            meta = fixture_meta()
            open(joinpath(dir, "meta.json"), "w") do io
                JSON.print(io, meta)
            end
            @test_throws ArgumentError MeshGraphNets.Dataset(:other, dir, test_args())
            @test_throws ArgumentError MeshGraphNets.Dataset(:train, dir, test_args())
            @test_throws ArgumentError MeshGraphNets.Dataset(
                joinpath(dir, "missing.jld2"), joinpath(dir, "meta.json"), test_args())
            text_path = joinpath(dir, "train.txt")
            open(text_path, "w") do io
                write(io, "not a dataset")
            end
            @test_throws ArgumentError MeshGraphNets.Dataset(
                text_path, joinpath(dir, "meta.json"), test_args())
            @test_throws ArgumentError MeshGraphNets.keystraj(text_path)
        end
    end

    @testset "trajectory loading" begin
        for format in (:jld2, :h5)
            mktempdir() do dir
                write_fixture_dir(dir; format)
                dataset = configure_dataset!(
                    MeshGraphNets.Dataset(:train, dir, test_args()))
                data = MLUtils.getobs(dataset, 1)
                @test data["trajectory_length"] == 4
                @test size(data["mesh_pos"]) == (1, 3, 1)
                @test size(data["velocity"]) == (2, 3, 4)
                @test size(data["split_feature"]) == (2, 3, 4)
                @test eltype(data["flag"]) == Bool
                @test data["velocity"][:, 2, :] == trajectory_values(2, 1).velocity
                @test data["pressure"][:, 3, :] == trajectory_values(3, 1).pressure
                @test data["mask"] == Int32[1, 3]
                @test data["inflow_mask"] == Int32[2]
                @test data["val_mask"] == Float32[1 0 1; 1 0 1]
                @test size(data["node_type"]) == (5, 3)
                @test all(data["senders"] .>= 1)
                @test all(data["receivers"] .>= 1)
                @test size(data["edge_features"]) == (2, 4)
            end
        end

        mktempdir() do dir
            cells_meta = fixture_meta(;
                edges = Dict{String, Any}("type" => "cells", "key" => "cells"))
            write_fixture_dir(dir; meta = cells_meta)
            cells_data = MLUtils.getobs(
                configure_dataset!(
                    MeshGraphNets.Dataset(:train, dir, test_args())), 1)
            @test length(cells_data["senders"]) == 6
        end
        mktempdir() do dir
            dims_meta = fixture_meta(; edges = Dict{String, Any}("type" => "dims"))
            write_fixture_dir(dir; meta = dims_meta)
            dims_data = MLUtils.getobs(
                configure_dataset!(
                    MeshGraphNets.Dataset(:train, dir, test_args())), 1)
            @test size(dims_data["edge_features"], 2) == 4
        end
        mktempdir() do dir
            malformed_meta = fixture_meta(; edges = Dict{String, Any}())
            write_fixture_dir(dir; meta = malformed_meta)
            malformed = configure_dataset!(
                MeshGraphNets.Dataset(:train, dir, test_args()))
            @test_throws ArgumentError MLUtils.getobs(malformed, 1)
        end
    end

    @testset "edges" begin
        node_type_1d = reshape(Int32[0, 4, 0, 0], 1, 4, 1)
        @test MeshGraphNets.create_edges([4], node_type_1d, Int[]) ==
              Int32[1 2 3; 2 3 4]
        @test MeshGraphNets.create_edges([4], node_type_1d, [4]) ==
              Int32[2 3; 2 4]
        @test_throws ArgumentError MeshGraphNets.create_edges([2, 2],
            reshape(zeros(Int32, 4), 1, 4, 1), Int[])
        @test_throws ArgumentError MeshGraphNets.create_edges([2, 2, 1, 1],
            reshape(zeros(Int32, 4), 1, 4, 1), Int[])

        node_type_3d = reshape(zeros(Int32, 8), 1, 8, 1)
        edges_3d = MeshGraphNets.create_edges([2, 2, 2], node_type_3d, Int[])
        @test size(edges_3d) == (2, 12)
        @test all(edges_3d[1, :] .< edges_3d[2, :])

        custom = Int32[1 2 1 3; 2 3 3 4]
        custom_types = reshape(Int32[0, 4, 0, 0], 1, 4, 1)
        @test MeshGraphNets.parse_custom_edges(custom, custom_types, [4], [4]) ==
              reshape(Int32[1, 3], 2, 1)
        @test MeshGraphNets.parse_custom_edges(permutedims(custom), custom_types,
            [4], [4]) == reshape(Int32[1, 3], 2, 1)
        @test_throws DimensionMismatch MeshGraphNets.parse_custom_edges(
            zeros(Int32, 3, 3), custom_types, Int[], Int[])
    end

    @testset "preprocessing" begin
        data = Dict{String, Any}(
            "dt" => Float32[0, 1, 2, 3],
            "node_type" => reshape(Int32[0, 4, 0], 1, 3, 1),
            "mesh_pos" => reshape(Float32[0, 1, 2], 1, 3, 1),
            "velocity" => reshape(Float32.(1:24), 2, 3, 4),
            "pressure" => reshape(Float32.(1:12), 1, 3, 4),
            "edges" => Int32[1 2; 2 3])
        original_velocity = copy(data["velocity"])
        MeshGraphNets.add_targets!(data, ["velocity"], cpu)
        @test data["velocity"] == original_velocity[:, :, 1:3]
        @test data["target|velocity"] == original_velocity[:, :, 2:4]
        @test size(data["mesh_pos"], 3) == 1
        @test data["dt"] == Float32[0, 1, 2, 3]

        shuffled = deepcopy(data)
        strategy = DerivativeTraining(; window_size = 2, random = true)
        MeshGraphNets.preprocess!(
            shuffled, ["velocity"], [0.0f0], [0], strategy, cpu)
        @test size(shuffled["velocity"], 3) == 2
        @test shuffled["target|velocity"] - shuffled["velocity"] ==
              data["target|velocity"][:, :, [1, 2]] - data["velocity"][:, :, [1, 2]]

        Random.seed!(1234)
        noisy = deepcopy(data)
        MeshGraphNets.preprocess!(noisy, ["velocity"], [0.25f0], [0],
            DerivativeTraining(; random = false), cpu)
        @test noisy["velocity"][:, 2, :] == data["velocity"][:, 2, :]
        @test noisy["velocity"][:, [1, 3], :] != data["velocity"][:, [1, 3], :]
        @test_throws DimensionMismatch MeshGraphNets.preprocess!(deepcopy(data),
            ["velocity", "pressure"], [0.1f0, 0.2f0, 0.3f0], [0], strategy, cpu)
    end
end
