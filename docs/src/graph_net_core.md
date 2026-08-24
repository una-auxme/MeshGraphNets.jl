# GraphNetCore.jl Documentation

## GraphNetwork

```@docs
GraphNetwork
build_model
step!
set_training!
save_checkpoint!
load_checkpoint
```

## FeatureGraph

```@docs
FeatureGraph
```

## Normaliser

```@docs
NormaliserOfflineMinMax
NormaliserOfflineMeanStd
NormaliserOnline
inverse_data
```

## Utilities

```@docs
triangles_to_edges
parse_edges
one_hot
minmaxnorm
```