# Training Data

If you want to train your own network via *MeshGraphNets.jl* you have to provide your data files and a corresponding metadata file in a specific manner. This section describes the structure of those files.

## Folder Structure

Your files have to be placed inside the same folder so that *MeshGraphNets.jl* can find them. The structure of your files should for example look like this:

    - data
        - datasets
            - meta.json
            - test.h5 (or test.tfrecord)
            - train.h5 (or train.tfrecord)
            - valid.h5 (or valid.tfrecord)

## Files for Training, Evaluation and Testing

For each of the steps a separate file has to be provided (see [Folder Structure](#folder-structure)). It is recommended to use `HDF5` files since they are easier to work with in Julia. You can also use `TFRecord` files, however an implementation is only provided for handling files that are exactly like the ones from the [CylinderFlow](https://una-auxme.github.io/MeshGraphNets.jl/dev/cylinder_flow) example.

> ⚠️ The following sections only contain explanations for using `HDF5` files.

### Data File Structure (`train.h5`, `valid.h5`, `test.h5`)

Your data files should have the following structure:

- *trajectory_1*
  - *feature_key_1*
  - *feature_key_2*
  - ...
- *trajectory_2*
  - *feature_key_1*
  - *feature_key_2*
  - ...
- ...

## File for Metadata (`meta.json`)

Your metadata file also has to follow a defined structure. Since the metadata file for the [CylinderFlow](https://una-auxme.github.io/MeshGraphNets.jl/dev/cylinder_flow) example is handled differently, two files are explained in the following.

### Default Metadata

The default structure that you should use for your metadata is the following (example derived from the [CylinderFlow metadata](#cylinderflow-metadata), not an actual metadata file):

```json
{
    "dt": "time",                               # key inside the HDF5 file for timesteps
    "trajectory_length": 600,                   # length of trajectories (i.e. number of steps) inside data files
    "dims": [                                   # dimensions of the mesh (here a mesh of dimensions (5, 3))
        5,
        3
    ],
    "feature_names": [                          # names of all features, mesh_pos and node_type are required
        "mesh_pos",
        "node_type",
        "velocity"
    ],
    "target_features": [                        # names of target features (i.e. quantities of interest) as output of the network
        "velocity"
    ],
    "features": {                               # detailed information of the features given above
        "mesh_pos": {                           # name of the feature given above
            "key": "cl_mesh[%d,%d].pos",        # key inside the HDF5 file for the feature, see below
            "split": true,                      # true if your feature is split between multiple keys, false otherwise
            "dim": 2,                           # dimension of the feature
            "type": "static",                   # "static" if the feature does not change over time, "dynamic" otherwise
            "dtype": "float32"                  # data type of the feature
        },
        "node_type": {
            "key": "cl_mesh[%d,%d].cellType",
            "dim": 1,
            "type": "static",
            "dtype": "int32",
            "onehot": true,                     # should the feature be represented as a onehot vector (optional)
            "data_min": 0,                      # minimum value of the feature (optional, required if "data_max" specified)
            "data_max": 6                       # maximum value of the feature (optional, required if "data_min" specified)
        },
        "velocity": {
            "key": "cl_mesh[%d,%d].velocity",
            "dim": 2,
            "type": "dynamic",
            "dtype": "float32"
        }
    }
}
```

Here is a detailed description of each possible metadata:

| Metadata            | Data type       | Description                                                                                    |
|---------------------|-----------------|------------------------------------------------------------------------------------------------|
| `"dt"`                | String          | each trajectory needs to have an entry for timesteps with the given key                      |
| `"trajectory_length"` | Integer         | each trajectory needs to have the same length i.e. the same amount of steps                  |
| `"dims"`              | Vector{Integer} | dimensions can be 1-, 2- or 3-dimensional                                                    |
| `"feature_names"`     | Vector{String}  | list all features that are also used as an input of the network                              |
| `"target_features"`   | Vector{String}  | list all features that the network should predict, they have to be part of `"feature_names"` |

Each feature has its own metadata:

| Feature Metadata | Data Type | Description                                                                                          |
|------------------|-----------|------------------------------------------------------------------------------------------------------|
| `"key"`            | String  | further description of `HDF5` key structure  are below                                               |
| `"split"`          | Bool    | keys are split at the end (e.g. `"cl_mesh[%d,%d].pos[1]"` and `"cl_mesh[%d,%d].pos[2]"`)             |
| `"dim"`            | Integer | dimension of the feature                                                                             |
| `"type"`           | String  | `"static"` if the feature does not change over time, `"dynamic"` otherwise                           |
| `"dtype"`          | String  | possible datatypes: `"int32"`, `"float32"`, `"Bool"`                                                 |
| `"onehot"`         | Bool    | can be used if you want to convert types represented as Integer to a onehot vector                   |
| `"data_min"`       | Float   | if you specify `"data_min"` and `"data_max"`, offline normalization is used, online otherwise        |
| `"data_max"`       | Float   | see `"data_min"`                                                                                     |
| `"target_min"`     | Float   | equivalent to `"data_min"` and `"data_max"`, specifies interval for normalization target             |
| `"target_max"`     | Float   | see `"target_min"`                                                                                   |

The structure for the `HDF5` key has to follow one rule:

> ⚠️ Square brackets are exlusively used
>   - once for the index of the mesh point (e.g. `"cl_mesh[%d,%d].cellType"`) and
>   - once at the end of the key if the feature `"split"` is set to `true`.

### CylinderFlow Metadata

The [metadata file](https://github.com/una-auxme/MeshGraphNets.jl/blob/main/examples/cylinder_flow/meta.json) (taken from the from [CylinderFlow](https://una-auxme.github.io/MeshGraphNets.jl/dev/cylinder_flow) example) has the following structure:

```json
{
    "dt": 0.01,                     # time delta between steps in the data 
    "trajectory_length": 600,       # length of trajectories (i.e. number of steps) inside data files
    "n_trajectories": 1000,         # number of trajectories inside train.h5
    "n_trajectories_valid": 100,    # number of trajectories inside valid.h5
    "dims": 2,                      # dimension of the mesh
    "feature_names": [              # names of all features, mesh_pos and node_type are required
        "cells",
        "mesh_pos",
        "node_type",
        "velocity"
    ],
    "target_features": [            # names of target features (i.e. quantities of interest) as output of the network
        "velocity"
    ],
    "features": {                   # detailed information of the features given above
      "cells": {                    # name of the feature given above
        "type": "static",           # "static" if the feature does not change over time, "dynamic" otherwise
        "dim": 3,                   # dimension of the feature
        "shape": [                  # individual dimensions of the feature, one dimension can be inferred from data via -1
          1,
          -1,
          3
        ],
        "dtype": "int32"            # data type of the feature
      },
      "mesh_pos": {
        "type": "static",
        "dim": 2,
        "shape": [
          1,
          -1,
          2
        ],
        "dtype": "float32"
      },
      "node_type": {
        "type": "static",
        "dim": 1,
        "shape": [
          1,
          -1,
          1
        ],
        "dtype": "int32",
        "onehot": true,             # should the feature be represented as a onehot vector (optional)
        "data_min": 0,              # minimum value of the feature (optional, required if "data_max" specified)
        "data_max": 6               # maximum value of the feature (optional, required if "data_min" specified)
      },
      "velocity": {
        "type": "dynamic",
        "dim": 2,
        "shape": [
          600,
          -1,
          2
        ],
        "dtype": "float32"
      }
    }
  }
```