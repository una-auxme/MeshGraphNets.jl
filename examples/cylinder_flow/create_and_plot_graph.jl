#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Makie, GraphMakie
using LinearAlgebra
using Graphs, MetaGraphsNext
using JLD2, JSON

is_pluto_notebook() = @isdefined(PlutoRunner)

if is_pluto_notebook()
    using WGLMakie
    WGLMakie.activate!()
else
    using GLMakie
end

"""
    plot_graph(g::MetaGraph; pos_key::Symbol = :mesh_pos, node_color = :red, 
        edge_color = :green, node_size = 10, edge_width = 3, height = 600, 
        width = 700, selected_keys::Vector{Symbol} = Symbol[])

Creates an interactive visualization of a MetaGraph with hover functionality.

## Arguments
- `g`: The MetaGraph to be visualized.
- `pos_key`: The key for position data in node attributes (default: `:mesh_pos`).
- `node_color`: Color of nodes (default: `:red`).
- `edge_color`: Color of edges (default: `:green`).
- `node_size`: Size of nodes (default: 10).
- `edge_width`: Width of edges (default: 3).
- `height`: Height of the figure in pixels (default: 600).
- `width`: Width of the figure in pixels (default: 700).
- `selected_keys`: List of attribute keys to display in the tooltip. If empty, all attributes are shown (default: empty vector).

## Returns
- A Makie Figure with the interactive graph visualization.
"""
function plot_graph(
        g::MetaGraph; pos_key::Symbol = :mesh_pos, node_color = :red,
        edge_color = :green, node_size = 10, edge_width = 3, height = 600,
        width = 700, selected_keys::Vector{Symbol} = Symbol[])
    # Define a custom layout function that extracts node positions from the graph
    function mylayout(g::MetaGraph)
        # Get labels for all vertices in the graph
        labels = [label_for(g, i) for i in vertices(g)]

        # Extract position data and convert to Point objects
        # Handles both 2D and 3D coordinates automatically
        [let pos = g[lbl][pos_key]
             length(pos) == 3 ? Point(pos[1], pos[2], pos[3]) : Point(pos[1], pos[2])
         end
         for lbl in labels]
    end

    # Determine if we're working with 3D data by checking the first node
    is_3D = length(g[label_for(g, 1)][pos_key]) == 3

    # Create the figure with specified dimensions
    fig = Figure(; size = (width, height))

    # Choose appropriate axis type based on data dimensionality
    ax = is_3D ? Axis3(fig[1, 1]) : Axis(fig[1, 1])

    # Initialize uniform sizes for all nodes and edges
    node_size_init = fill(node_size, nv(g))
    edge_width_init = fill(edge_width, ne(g))

    # Create the graph plot with our custom layout
    p = graphplot!(ax, g;
        layout = mylayout,
        edge_width = edge_width_init,
        node_size = node_size_init,
        node_color = node_color,
        edge_color = edge_color)

    # Create a grid layout for the tooltip box
    gl = GridLayout(
        fig[1, 1]; tellwidth = false, tellheight = false, halign = :right, valign = :top)

    # Observable text for label - will be updated when hovering
    label_text = Observable("Hover over a \n node or edge")

    # Create a white box for the tooltip background
    Box(gl[1, 1]; color = :white, strokecolor = :black, strokewidth = 2)

    # Tooltip label
    Label(gl[1, 1], label_text; fontsize = 14, color = :black, padding = (15, 15, 15, 15))

    # Store all edges for quick lookup during hover events
    edge_list = collect(edges(g))

    # Define the action to take when hovering over a node
    function node_action(state::Bool, idx::Int, event, axis)
        if state  # Mouse is over the node
            # Get the node's label and data
            node_label = label_for(g, idx)
            node_data = g[node_label]  # Get node attributes (Dict)

            # Increase the size of the hovered node for visual feedback
            p.node_size[][idx] = 2 * minimum(p.node_size[])

            # If no selected_keys are given, use all keys from node_data
            display_keys = isempty(selected_keys) ? keys(node_data) : selected_keys

            # Generate tooltip text in the given order
            tooltip = "Node: $node_label\n"
            for key in display_keys
                if haskey(node_data, key)
                    value = node_data[key]

                    # Determine max absolute value for this key to set appropriate precision
                    if value isa Tuple
                        max_value = maximum(abs.(value))
                    elseif value isa Number
                        max_value = abs(value)
                    else
                        max_value = 1  # Default for non-numeric types
                    end

                    # Determine number of decimal places dynamically based on value magnitude
                    # Small values get more decimal places for precision
                    decimal_places = max_value ≥ 1 ? 2 :
                                     min(6, ceil(Int, abs(log10(max_value))) + 2)

                    # Format the tooltip text based on the type of data
                    tooltip *= if key == pos_key  # If the key represents position
                        if length(value) == 3
                            "x: $(round(value[1]; digits=decimal_places)), y: $(round(value[2]; digits=decimal_places)), z: $(round(value[3]; digits=decimal_places))\n"
                        else
                            "x: $(round(value[1]; digits=decimal_places)), y: $(round(value[2]; digits=decimal_places))\n"
                        end
                    elseif key == :velocity  # If the key represents velocity
                        if length(value) == 3
                            "v_x: $(round(value[1]; digits=decimal_places)), v_y: $(round(value[2]; digits=decimal_places)), v_z: $(round(value[3]; digits=decimal_places))\n"
                        else
                            "v_x: $(round(value[1]; digits=decimal_places)), v_y: $(round(value[2]; digits=decimal_places))\n"
                        end
                    elseif key == :pressure  # Convert Pressure to p
                        "p: $(round(value; digits=decimal_places))\n"
                    elseif value isa Tuple  # If the value is another tuple
                        "$key: " *
                        string(tuple(round.(value; digits = decimal_places)...)) * "\n"
                    elseif value isa Number  # If the value is a single number
                        "$key: " * string(round(value; digits = decimal_places)) * "\n"
                    else  # Default case for other data types
                        "$key: " * string(value) * "\n"
                    end
                end
            end

            # Update the tooltip text
            label_text[] = tooltip
        else  # Mouse is no longer over the node
            # Reset tooltip and node size
            label_text[] = "Hover over a \n node or edge"
            p.node_size[][idx] = minimum(p.node_size[])
        end
        # Trigger update of node sizes
        p.node_size[] = p.node_size[]
    end

    # Define the action to take when hovering over an edge
    function edge_action(state::Bool, idx::Int, event, axis)
        if state  # Mouse is over the edge
            # Increase the width of the hovered edge for visual feedback
            p.edge_width[][idx] = 2 * minimum(p.edge_width[])

            # Get edge information
            e = edge_list[idx]
            i, j = src(e), dst(e)  # Source and destination vertices

            # Get the weight matrix and extract this edge's weight
            weight_matrix = Graphs.weights(g)
            weight_tuple = weight_matrix[i, j]  # NamedTuple with dx, dy, dz, distance

            # Exit if no valid weight exists
            if !(weight_tuple isa NamedTuple)
                return
            end

            # Collect all valid distances to determine appropriate rounding precision
            valid_weights = [edge.distance
                             for edge in values(weight_matrix) if edge isa NamedTuple]

            # Find the largest absolute value in the weight matrix to determine rounding precision
            max_value = !isempty(valid_weights) ? maximum(abs.(valid_weights)) : 1.0

            # Determine number of decimal places based on max_value
            decimal_places = max_value ≥ 1 ? 2 :
                             min(6, ceil(Int, abs(log10(max_value))) + 2)

            # Create edge label with adaptive rounding
            tooltip = "Edge idx: $(idx)\n"
            tooltip *= "dx: $(round(weight_tuple.dx; digits=decimal_places)), "
            tooltip *= "dy: $(round(weight_tuple.dy; digits=decimal_places))\n"

            # Add z-coordinate for 3D graphs
            if is_3D
                tooltip *= "dz: $(round(weight_tuple.dz; digits=decimal_places))\n"
            end

            # Add the Euclidean distance
            tooltip *= "distance: $(round(weight_tuple.distance; digits=decimal_places))"

            # Update the tooltip text
            label_text[] = tooltip
        else  # Mouse is no longer over the edge
            # Reset tooltip and edge width
            label_text[] = "Hover over a \n node or edge"
            p.edge_width[][idx] = minimum(p.edge_width[])
        end
        # Trigger update of edge widths
        p.edge_width[] = p.edge_width[]
    end

    # Register the interaction handlers for node and edge hovering
    register_interaction!(ax, :nodehover, NodeHoverHandler(node_action))
    register_interaction!(ax, :edgehover, EdgeHoverHandler(edge_action))

    # Ensure the figure layout is properly sized
    resize_to_layout!(fig)
    fig
end

"""
    triangles_to_edges(faces)

Converts the given faces of a mesh to edges.

## Arguments
- `faces`: Two-dimensional array with the node indices in the first dimension.

## Returns
- Tuple containing the edge pairs. (See [`parse_edges`](@ref))
"""
function triangles_to_edges(faces::AbstractArray{T, 2} where {T <: Integer})
    edges = hcat(faces[1:2, :], faces[2:3, :], permutedims(hcat(faces[3, :], faces[1, :])))

    return parse_edges(edges)
end

"""
    parse_edges(edges)

Converts the given edges to unique pairs of senders and receivers (in both directions).

## Arguments
- `edges`: Two-dimensional Array containing the edges. The first dimension represents a sender-receiver pair.

## Returns
- Tuple containing the bi-directional sender-receiver pairs. The first index is one direction, the second index the other one.
"""
function parse_edges(edges)
    receivers = minimum(edges; dims = 1)
    senders = maximum(edges; dims = 1)
    packed_edges = vcat(senders, receivers)
    unique_edges = unique(packed_edges; dims = 2)
    senders = unique_edges[1, :]
    receivers = unique_edges[2, :]

    return vcat(senders, receivers), vcat(receivers, senders)
end

"""
    create_and_plot_graph(datafile::JLD2.JLDFile, trajectory::String, meta::Dict{String, Any}, 
        time_step::Int, selected_keys::Vector{Symbol} = Symbol[], pos_key::Symbol = :mesh_pos)

Creates and visualizes a graph from simulation data at a specific time step.

## Arguments
- `datafile`: The opened JLD2 file containing simulation data.
- `trajectory`: The name of the trajectory to analyze in the file.
- `meta`: A dictionary with metadata about the simulation.
- `time_step`: The time step for which to create the graph.
- `selected_keys`: List of attribute keys to display in the tooltip. If empty, all attributes are shown (default: empty vector).
- `pos_key`: The key for position data in node attributes (default: `:mesh_pos`).

## Functionality
1. Checks if the specified time step and position key are valid.
2. Extracts node data from the trajectory.
3. Creates a MetaGraph with the extracted data.
4. Adds edges based on the cell data.
5. Visualizes the graph using the `plot_graph` function.

## Returns
- A tuple containing:
    - `Figure`: The Makie Figure with the interactive graph visualization, or `nothing` in case of errors.
    - `MetaGraphsNext`:The created MetaGraph object for further analysis, or `nothing` in case of errors.
"""
function create_and_plot_graph(
        datafile::JLD2.JLDFile, trajectory::String, meta::Dict{String, Any}, time_step::Int,
        selected_keys::Vector{Symbol} = Symbol[], pos_key::Symbol = :mesh_pos)
    # Validate that the requested time step is within bounds
    if time_step > meta["trajectory_length"]
        println("Time step out of bounds.")
        close(datafile)
        return nothing, nothing
    end

    # Ensure the position key exists in the metadata
    if String(pos_key) ∉ meta["feature_names"]
        println("Position key not found in meta.")
        close(datafile)
        return nothing, nothing
    end

    # Access the trajectory data from the JLD2 file
    data = datafile[trajectory]

    # Extract unique node names using regex pattern matching
    # This finds all keys that match the pattern "node[digit]"
    nodes = unique(match(r"node\[\d+\]", x).match
    for x in keys(data) if occursin(r"node\[\d+\]", x))

    # Define a function to calculate edge properties between two nodes
    function calculate_delta_and_distance(edge_data)
        # This function computes both the Euclidean distance and the deltas (dx, dy, dz)
        graph, (v1, v2) = edge_data
        v1 = Symbol(v1)
        v2 = Symbol(v2)

        # Get the coordinates of both nodes
        coord_1 = graph[v1][pos_key]
        coord_2 = graph[v2][pos_key]

        # Compute the coordinate differences (deltas)
        dx = coord_2[1] - coord_1[1]
        dy = coord_2[2] - coord_1[2]
        # Handle both 2D and 3D cases for z-coordinate
        dz = length(coord_1) == 3 ? coord_2[3] - coord_1[3] : 0.0

        # Compute the Euclidean distance between the nodes
        distance = norm((dx, dy, dz))

        # Return all computed values as a named tuple
        return (dx = dx, dy = dy, dz = dz, distance = distance)
    end

    # Initialize a MetaGraph to represent the simulation data
    # This graph will store nodes with their attributes and edges with weights
    graph = MetaGraph(
        Graph();
        label_type = Symbol,
        vertex_data_type = Dict{Symbol, Any},  # Store all node attributes in a Dict
        edge_data_type = NamedTuple{
            (:graph, :vertexes), Tuple{MetaGraph, Tuple{Symbol, Symbol}}},
        weight_function = calculate_delta_and_distance,
        default_weight = 0
    )

    # Iterate over each node and extract its properties from the data
    for node_name in nodes
        node_symbol = Symbol(node_name)
        node_data = Dict{Symbol, Any}()  # Store all attributes in a dictionary

        # Dynamically extract all features defined in the metadata
        for feature in meta["feature_names"]
            feature_info = meta["features"][feature]
            feature_key = feature_info["key"]  # Key format e.g., "node[%d].mesh_pos"

            # Format key correctly for the current node by replacing %d with the node index
            node_feature_key = replace(feature_key, "%d" => match(r"\d+", node_name).match)

            # If this feature exists for this node, extract its value
            if haskey(data, node_feature_key)
                value = data[node_feature_key]

                # Handle different dimensionality of features (1D, 2D, 3D)
                if feature_info["dim"] == 1
                    if feature_info["type"] == "dynamic"
                        # For dynamic features, extract the value at the specified time step
                        node_data[Symbol(feature)] = value[time_step]
                    else
                        # For static features, use the value as is
                        node_data[Symbol(feature)] = value
                    end
                elseif feature_info["dim"] >= 2
                    if feature_info["type"] == "dynamic"
                        # For multi-dimensional dynamic features, convert to tuple for the specified time step
                        node_data[Symbol(feature)] = Tuple(value[:, time_step])
                    else
                        # For multi-dimensional static features, convert to tuple
                        node_data[Symbol(feature)] = Tuple(value)
                    end
                end
            else
                # If the feature doesn't exist for this node, mark it as missing
                node_data[Symbol(feature)] = missing
            end
        end

        # Store the node's index for reference
        node_data[:node_idx] = parse(Int, match(r"\d+", node_name).match)

        # Add the node with all its attributes to the graph
        graph[node_symbol] = node_data
    end

    println("Graph successfully created with ", length(nodes), " nodes.")

    # If "cells" data exists, use it to create edges between nodes
    # Cells typically define triangular elements in the mesh
    if haskey(data, "cells")
        # Convert triangular faces to edges
        senders, receivers = triangles_to_edges(data["cells"][:, :, 1])

        # Handle 0-indexed data by incrementing indices if needed
        if 0 in senders || 0 in receivers
            senders .+= 1
            receivers .+= 1
        end

        # Convert senders and receivers arrays into edge tuples
        edgs = collect(zip(senders, receivers))
    end

    # Add all edges to the graph
    for (sender, receiver) in edgs
        sender_symbol = Symbol("node[$sender]")
        receiver_symbol = Symbol("node[$receiver]")

        # Ensure both nodes exist in the graph before adding the edge
        if !haskey(graph, sender_symbol) || !haskey(graph, receiver_symbol)
            println("Warning: Node missing for edge ($sender_symbol, $receiver_symbol)")
            continue
        end

        # Add the edge to the graph with metadata needed for weight calculation
        graph[sender_symbol, receiver_symbol] = (
            graph = graph, vertexes = (sender_symbol, receiver_symbol))
    end

    # Close the data file as it's no longer needed
    close(datafile)

    # Create and display the interactive graph visualization
    plot_graph(graph; selected_keys = selected_keys, pos_key = pos_key)
end

# if !isdefined(Base, :interactive) || !Base.interactive
#     #########################
#     # Paths and filenames  #
#     #########################

#     # Path to the example directory
#     path = "examples/cylinder_flow"

#     # Name of the file to be loaded
#     file = "test_single_trajectory.jld2"

#     ###################
#     # Load data       #
#     ###################

#     # Open the JLD2 file and load metadata
#     # The metadata contains important information about the simulation
#     datafile = jldopen(joinpath(path, file), "r")
#     meta = JSON.parse(Base.read(joinpath(path, "meta.json"), String))

#     #############################
#     # Analyze trajectory       #
#     #############################

#     # Name of the trajectory to be analyzed
#     trajectory = "trajectory_1"

#     # Time step for analysis
#     # Specifies the time at which data points are evaluated
#     time_step = 1

#     # List of attributes to be used for visualization
#     selected_ordered_attributes = [:Node, :mesh_pos, :pressure, :velocity]

#     ########################
#     # Generate and plot graph #
#     ########################

#     # The function `create_and_plot_graph` generates a graph from the loaded data
#     # and visualizes it.
#     create_and_plot_graph(
#         datafile, trajectory, meta, time_step, selected_ordered_attributes)
# end
