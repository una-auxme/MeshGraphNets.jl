using TFRecord, JLD2

trajectory_length = 600
features = Dict{String, Dict}(
    "cells" => Dict{String, Any}(
        "type" => "static",
        "shape" => [1, -1, 3],
        "dtype" => "int32"
    ),
    "mesh_pos" => Dict{String, Any}(
        "type" => "static",
        "shape" => [1, -1, 2],
        "dtype" => "float32"
    ),
    "node_type" => Dict{String, Any}(
        "type" => "static",
        "shape" => [1, -1, 1],
        "dtype" => "int32"
    ),
    "velocity" => Dict{String, Any}(
        "type" => "dynamic",
        "shape" => [trajectory_length, -1, 2],
        "dtype" => "float32"
    ),
    "pressure" => Dict{String, Any}(
        "type" => "dynamic",
        "shape" => [trajectory_length, -1, 1],
        "dtype" => "float32"
    )
)

function parse_data(data::TFRecord.Example)
    out = Dict{String, AbstractArray}()
    for (key, value) in features
        d = reinterpret(getfield(Base, Symbol(uppercasefirst(value["dtype"]))),
            data.features.feature[key].kind.value.value[])
        dims = Tuple(reverse(replace(
            value["shape"], -1 => abs(reduce(div, value["shape"]; init = length(d))))))
        d = reshape(d, dims)
        if value["type"] == "static"
            d = repeat(d, 1, 1, trajectory_length)
        end
        out[key] = d
    end
    return out
end

for file in ["train", "valid", "test"]
    i = 1
    jld2_file = jldopen("$file.jld2", "w")
    for traj in TFRecord.read("$file.tfrecord")
        traj_dict = parse_data(traj)
        traj_group = JLD2.Group(jld2_file, "trajectory_$i")

        cells = traj_dict["cells"]
        mesh_pos = traj_dict["mesh_pos"]
        node_type = traj_dict["node_type"]
        pressure = traj_dict["pressure"]
        velocity = traj_dict["velocity"]

        traj_group["cells"] = cells[:, :, 1]
        for idx in axes(mesh_pos, 2)
            traj_group["node[$idx].mesh_pos"] = mesh_pos[:, idx, 1]
            traj_group["node[$idx].node_type"] = node_type[:, idx, 1]
            traj_group["node[$idx].p"] = pressure[:, idx, :]
            traj_group["node[$idx].V"] = velocity[:, idx, :]
        end
        traj_group["n_nodes"] = size(mesh_pos, 2)
        i += 1
    end
    close(jld2_file)
end
