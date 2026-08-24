# Copyright (c) 2026 Julian Trommer
# SPDX-License-Identifier: MIT
# See LICENSE for details.

using Statistics

struct UnknownTrainingStrategy <: MeshGraphNets.TrainingStrategy end

struct TestGraphLinearLayer <: Lux.AbstractLuxLayer end

const IRREGULAR_TIMES = Float32[0, 0.5, 1.5, 3]
const REGULAR_TIMES = Float32[0, 0.1, 0.2, 0.3, 0.4]

function (::TestGraphLinearLayer)(graph::FeatureGraph, ps, st)
    return ps.scale .* graph.nf[1:2, :], st
end

function fixture_meta(; edges = Dict{String, Any}("type" => "custom", "key" => "edges"))
    return Dict{String, Any}(
        "dt" => "time",
        "trajectory_length" => -1,
        "dims" => [3],
        "edges" => edges,
        "feature_names" =>
            ["mesh_pos", "node_type", "velocity", "pressure", "split_feature", "flag"],
        "target_features" => ["velocity"],
        "features" => Dict{String, Any}(
            "mesh_pos" => Dict{String, Any}("key" => "node[%d].mesh_pos", "dim" => 1,
                "type" => "static", "dtype" => "float32"),
            "node_type" =>
                Dict{String, Any}("key" => "node[%d].node_type", "dim" => 1,
                    "type" => "static", "dtype" => "int32", "onehot" => true,
                    "data_min" => 0, "data_max" => 4),
            "velocity" => Dict{String, Any}("key" => "node[%d].velocity", "dim" => 2,
                "type" => "dynamic", "dtype" => "float32"),
            "pressure" => Dict{String, Any}("key" => "node[%d].pressure", "dim" => 1,
                "type" => "dynamic", "dtype" => "float32"),
            "split_feature" =>
                Dict{String, Any}("key" => "node[%d].split", "dim" => 2,
                    "type" => "dynamic", "dtype" => "float32", "split" => true),
            "flag" => Dict{String, Any}("key" => "node[%d].flag", "dim" => 1,
                "type" => "static", "dtype" => "Bool")
        )
    )
end

function trajectory_values(node, trajectory; times = IRREGULAR_TIMES)
    times = Float32.(times)
    offset = Float32(node - 1 + trajectory - 1)
    velocity = vcat(reshape(2.0f0 .* times .+ offset, 1, :),
        reshape(-times .+ offset, 1, :))
    pressure = reshape(times .+ offset, 1, :)
    split_feature = vcat(reshape(times .+ node, 1, :), reshape(2.0f0 .* times, 1, :))
    node_types = Int32[0, 4, 1]
    return (; times, velocity, pressure, split_feature,
        mesh_pos = Float32[node - 1], node_type = Int32[node_types[node]], flag = Bool[node == 1])
end

function write_jld2_fixture(path, trajectories; times = IRREGULAR_TIMES)
    jldopen(path, "w") do file
        for trajectory in 1:trajectories
            group = JLD2.Group(file, "trajectory_$trajectory")
            values = trajectory_values(1, trajectory; times)
            group["time"] = values.times
            group["trajectory_length"] = Int32(length(values.times))
            group["n_nodes"] = Int32(3)
            group["actual_dims"] = Int32[3]
            group["edges"] = Int32[1 2; 2 3]
            group["cells"] = reshape(Int32[0, 1, 2], 3, 1)
            for node in 1:3
                values = trajectory_values(node, trajectory; times)
                group["node[$node].mesh_pos"] = values.mesh_pos
                group["node[$node].node_type"] = values.node_type
                group["node[$node].velocity"] = values.velocity
                group["node[$node].pressure"] = values.pressure
                group["node[$node].split[1]"] = vec(values.split_feature[1, :])
                group["node[$node].split[2]"] = vec(values.split_feature[2, :])
                group["node[$node].flag"] = values.flag
            end
        end
    end
end

function write_hdf5_fixture(path, trajectories; times = IRREGULAR_TIMES)
    h5open(path, "w") do file
        for trajectory in 1:trajectories
            group = create_group(file, "trajectory_$trajectory")
            values = trajectory_values(1, trajectory; times)
            group["time"] = values.times
            group["trajectory_length"] = Int32(length(values.times))
            group["n_nodes"] = Int32(3)
            group["actual_dims"] = Int32[3]
            group["edges"] = Int32[1 2; 2 3]
            group["cells"] = reshape(Int32[0, 1, 2], 3, 1)
            for node in 1:3
                values = trajectory_values(node, trajectory; times)
                group["node[$node].mesh_pos"] = values.mesh_pos
                group["node[$node].node_type"] = values.node_type
                group["node[$node].velocity"] = values.velocity
                group["node[$node].pressure"] = values.pressure
                group["node[$node].split[1]"] = vec(values.split_feature[1, :])
                group["node[$node].split[2]"] = vec(values.split_feature[2, :])
                group["node[$node].flag"] = values.flag
            end
        end
    end
end

function write_fixture_dir(dir; format = :jld2, trajectories = 1, all_splits = false,
        meta = fixture_meta(), times = IRREGULAR_TIMES)
    open(joinpath(dir, "meta.json"), "w") do io
        JSON.print(io, meta)
    end
    splits = all_splits ? (:train, :valid, :test) : (:train,)
    for split in splits
        if format == :jld2
            write_jld2_fixture(joinpath(dir, "$(split).jld2"), trajectories; times)
        else
            write_hdf5_fixture(joinpath(dir, "$(split).h5"), trajectories; times)
        end
    end
    return dir
end

function test_args(; use_cuda = false,
        training_strategy = DerivativeTraining(; random = false), kws...)
    return MeshGraphNets.Args(; use_cuda, training_strategy,
        types_inflow = [4], types_updated = [0, 1], types_noisy = [0],
        noise_stddevs = [0.0f0], kws...)
end

function configure_dataset!(dataset; training_strategy = nothing, device = cpu_device())
    dataset.meta["device"] = device
    dataset.meta["training_strategy"] = training_strategy
    return dataset
end

function test_mgn(device = cpu_device())
    model = TestGraphLinearLayer()
    parameters = (; scale = device(ones(Float32, 2, 1)))
    states = (;)
    train_state = Lux.Training.TrainState(
        model, parameters, states, Optimisers.Descent(1.0f-2))
    edge_norm = NormaliserOfflineMeanStd(Float32[0, 0], Float32[1, 1], device)
    node_norms = Dict{String, Union{NormaliserOffline, NormaliserOnline}}(
        "velocity" => NormaliserOfflineMeanStd(Float32[0, 0], Float32[1, 1], device),
        "node_type" => NormaliserOfflineMinMax(0.0f0, 1.0f0, device))
    output_norms = Dict{String, Union{NormaliserOffline, NormaliserOnline}}(
        "velocity" => NormaliserOfflineMeanStd(Float32[0, 0], Float32[1, 1], device))
    return GraphNetwork(train_state, edge_norm, node_norms, output_norms)
end

flat_native(tree) = first(Optimisers.destructure(tree))

function flat_cpu(tree)
    flat = flat_native(tree)
    return collect(cpu_device()(flat))
end

tree_isfinite(tree) = all(isfinite, flat_cpu(tree))
tree_norm(tree) = sqrt(sum(abs2, flat_cpu(tree)))

function strategy_fixture(device; derivative = false)
    base = Float32[1 2 3; 4 5 6]
    velocity = cat([(1.0f0 + 0.25f0 * time) .* base for time in REGULAR_TIMES]...;
        dims = 3)
    data = Dict{String, Any}(
        "dt" => copy(REGULAR_TIMES),
        "inflow_mask" => device(Int32[]))
    if derivative
        data["velocity"] = device(velocity[:, :, 1:(end - 1)])
        data["target|velocity"] = device(velocity[:, :, 2:end])
    else
        data["velocity"] = device(velocity)
    end

    mgn = test_mgn(device)
    meta = Dict(
        "target_features" => ["velocity"],
        "features" => Dict("velocity" => Dict("dim" => 2)))
    fields = ["velocity"]
    target_fields = ["velocity"]
    node_type = device(Float32[1 0 1; 0 1 0])
    edge_features = device(Float32[1 2; 1 2])
    senders = device(Int32[1, 2])
    receivers = device(Int32[2, 3])
    mask = device(Int32[1, 2, 3])
    val_mask = device(ones(Float32, 2, 3))
    tuple = (mgn, data, meta, fields, target_fields, node_type, edge_features,
        senders, receivers, 1, mask, val_mask)
    return (; mgn, data, meta, fields, target_fields, node_type, edge_features,
        senders, receivers, mask, val_mask, tuple)
end

function strategy_step_result(strategy, device)
    fixture = strategy_fixture(device; derivative = strategy isa DerivativeTraining)
    datapoint = strategy isa SolverBatchTraining ?
                first(MeshGraphNets.get_delta(strategy, length(REGULAR_TIMES))) : 1
    step_tuple = Base.setindex(fixture.tuple, datapoint, 10)
    initialized = MeshGraphNets.init_train_step(
        strategy, step_tuple, MeshGraphNets.prepare_training(strategy))
    gradients, loss = MeshGraphNets.train_step(strategy, initialized)
    gradient = only(gradients)
    parameters_before = flat_cpu(fixture.mgn.train_state.parameters)
    train_state = Lux.Training.apply_gradients!(fixture.mgn.train_state, gradient)
    fixture.mgn.train_state = train_state
    parameters_after = flat_cpu(train_state.parameters)

    validation_tuple = (fixture.mgn, fixture.data, fixture.meta,
        length(REGULAR_TIMES) - 1, Tsit5(), nothing, fixture.fields,
        fixture.node_type, fixture.edge_features, fixture.senders, fixture.receivers,
        fixture.mask, fixture.val_mask, fixture.data["inflow_mask"], nothing)
    validation_loss = MeshGraphNets.validation_step(strategy, validation_tuple)

    return (; fixture, initialized, gradient, loss, train_state, validation_loss,
        parameters_before, parameters_after)
end

function test_strategy_step(result)
    @test isfinite(result.loss)
    @test result.loss > 0
    @test tree_isfinite(result.gradient)
    @test tree_norm(result.gradient) > 0
    @test result.train_state.step == 1
    @test all(isfinite, result.parameters_after)
    @test tree_norm(result.train_state.parameters) > 0
    @test result.parameters_after != result.parameters_before
    @test isfinite(result.validation_loss)
end

function assert_checkpoint(checkpoint_dir, mgn)
    @test mgn.train_state.step > 0
    @test tree_isfinite(mgn.train_state.parameters)
    @test tree_norm(mgn.train_state.parameters) > 0
    @test isfile(joinpath(checkpoint_dir, "checkpoints"))
    @test isfile(joinpath(checkpoint_dir, "valid", "checkpoints"))

    checkpoint_step = parse(Int, last(readlines(joinpath(checkpoint_dir, "checkpoints"))))
    checkpoint = JLD2.load(joinpath(checkpoint_dir, "checkpoint_$checkpoint_step.jld2"))
    @test checkpoint["df_train"].step[end] == checkpoint_step
    @test isfinite(checkpoint["df_train"].loss[end])
    @test checkpoint["df_train"].loss[end] > 0
end

function assert_evaluation_file(output_file, trajectories, saves)
    @test isfile(output_file)
    jldopen(output_file, "r") do file
        @test length(keys(file)) == trajectories
        for trajectory in unique((1, trajectories))
            group = file["trajectory_$trajectory"]
            @test all(haskey(group, key)
            for key in ("mesh_pos", "gt", "prediction", "error", "timesteps", "edges"))
            @test size(group["gt"], 3) == length(saves)
            @test size(group["prediction"], 3) == length(saves)
            @test length(group["error"]) == length(saves)
            @test all(isfinite, group["prediction"])
            @test all(isfinite, group["error"])
            @test group["timesteps"] ≈ saves
        end
    end
end

function run_evaluation_integration(checkpoint_dir; use_cuda = false,
        trajectories = 1, solver = Tsit5(), dt = nothing)
    mktempdir() do evaluation_dir
        write_fixture_dir(
            evaluation_dir; trajectories, all_splits = true, times = REGULAR_TIMES)
        output_dir = joinpath(evaluation_dir, "output")
        device_id = use_cuda ? CUDA.device() : nothing
        saves = Float32[0, 0.1]
        eval_network(evaluation_dir, checkpoint_dir, output_dir, solver;
            start = 0.0f0, stop = 0.1f0, dt, saves,
            mse_steps = Float32[0.1], mps = 1, layer_size = 4,
            hidden_layers = 0, use_cuda, gpu_device = device_id,
            types_inflow = [4], types_updated = [0, 1], use_valid = false)
        solver_dir = isnothing(solver) ?
                     "derivative_training" : lowercase("$(nameof(typeof(solver)))")
        output_file = joinpath(output_dir, solver_dir, "trajectories.jld2")
        assert_evaluation_file(output_file, trajectories, saves)
    end
end

function run_training_integration(strategy; use_cuda = false,
        evaluation_trajectories = 0, additional_evaluation_paths = false)
    mktempdir() do training_dir
        write_fixture_dir(training_dir; all_splits = true, times = REGULAR_TIMES)
        checkpoint_dir = joinpath(training_dir, "checkpoints")
        mkpath(joinpath(checkpoint_dir, "valid"))
        device_id = use_cuda ? CUDA.device() : nothing

        Random.seed!(1234)
        mgn,
        minimum_validation_loss = train_network(
            Optimisers.Adam(1.0f-3), training_dir, checkpoint_dir;
            mps = 1, layer_size = 4, hidden_layers = 0, steps = 1,
            checkpoint = 1, norm_steps = 0, use_cuda, gpu_device = device_id,
            types_inflow = [4], types_updated = [0, 1], types_noisy = [0],
            noise_stddevs = [0.0f0], training_strategy = strategy,
            solver_valid = Tsit5(), solver_valid_dt = 0.02f0)

        @test isfinite(minimum_validation_loss)
        if use_cuda
            @test parent(mgn.train_state.parameters) isa CuArray
        end
        assert_checkpoint(checkpoint_dir, mgn)

        if evaluation_trajectories > 0
            run_evaluation_integration(checkpoint_dir;
                use_cuda, trajectories = evaluation_trajectories, solver = Tsit5())
        end
        if additional_evaluation_paths
            run_evaluation_integration(checkpoint_dir;
                use_cuda, trajectories = 1, solver = Tsit5(), dt = 0.02f0)
            run_evaluation_integration(checkpoint_dir;
                use_cuda, trajectories = 1, solver = nothing)
        end
    end
end
