# CylinderFlow by Google DeepMind

This examples provides information on preparing the data for the CylinderFlow example provided by [Google DeepMind](https://deepmind.google/) in their corresponding [repository](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) an how you can train and evaluate the resulting network.

## Data preparation

First you need to download the provided datasets for training, evaluation and testing. An explanation on how you can download the files is provided in the repository:

> https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets#datasets

If you execute the file [download_dataset.sh](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/download_dataset.sh) with the argument `cylinder_flow` the following files should download:

- meta.json
- train.tfrecord
- valid.tfrecord
- test.tfrecord

You can keep the `.tfrecord` files as is. You only need to change the `meta.json` file to be compatible with [*MeshGraphNets.jl*](https://github.com/una-auxme/MeshGraphNets.jl). The correct file is provided in the [examples folder](https://github.com/una-auxme/MeshGraphNets.jl/tree/main/examples/cylinder_flow) and you only need to copy it to the same folder as the `.tfrecord` files.

> If you want to understand the structure of the `meta.json` file take a look at the section [Training Data](https://una-auxme.github.io/MeshGraphNets.jl/dev/training_data).

The default path for the data folder that is specified in the [example script](https://github.com/una-auxme/MeshGraphNets.jl/blob/main/examples/cylinder_flow/cylinder_flow.jl) is:

    `{path_to_cylinder_flow.jl}/data/datasets/`

## Training the network

In order to train the system you can simply comment in/out the lines of code provided in the script:

```julia
#################
# Train network #
#################

# with Collocation

train_network(
    noise, opt, ds_path, chk_path; mps = message_steps, layer_size = layer_size, hidden_layers = hidden_layers, batchsize = batch,
    epochs = epo, steps = Int(ns), use_cuda = cuda, checkpoint = cp, norm_steps = 1000, types_updated = types_updated,
    types_noisy = types_noisy, training_strategy = Collocation(), solver_valid = Euler(), solver_valid_dt = 0.01f0
)

# with SingleShooting

train_network(
    noise, opt, ds_path, chk_path; mps = message_steps, layer_size = layer_size, hidden_layers = hidden_layers, batchsize = batch, epochs = epo,
    steps = Int(ns), use_cuda = cuda, checkpoint = 10, norm_steps = 1000, types_updated = types_updated, types_noisy = types_noisy,
    training_strategy = SingleShooting(0.0f0, 0.01f0, 5.99f0, Euler(); adaptive = false, tstops = 0.0f0:0.01f0:5.99f0)
)
```

## Evaluating the system

The same applies to evaluating the system. Simply comment in/out the desired lines of code:

```julia
####################
# Evaluate network #
####################

# with Euler (fixed timestep)

eval_network(
    ds_path, chk_path, eval_path, Euler(); start = 0.0f0, stop = 5.99f0, dt = 0.01f0, saves = 0.0f0:0.01f0:5.99f0,
    mse_steps = collect(0.0f0:1.0f0:5.99f0), mps = message_steps, layer_size = layer_size, hidden_layers = hidden_layers, use_cuda=cuda
)

# with Tsit5 (adaptive timestep)

eval_network(
    ds_path, chk_path, eval_path, Tsit5(); start = 0.0f0, stop = 5.99f0, saves = 0.0f0:0.01f0:5.99f0,
    mse_steps = collect(0.0f0:1.0f0:5.99f0), mps = message_steps, layer_size = layer_size, hidden_layers = hidden_layers, use_cuda=cuda
)

```

## Addition: Arguments for training & evaluation

The arguments provided at the top of the example script correspond to the default values that were used by DeepMind. You can change them to see how that affects runtime and accuracy of the network.

The arguments inside the function calls can also be modified. An explanation can be found in the [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/train_eval).