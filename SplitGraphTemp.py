def split_graph_and_relabel_with_all_nodes(file_path):
    import numpy as np
    from collections import defaultdict

    # Read the file and collect the edges
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            source, target, timestamp = line.split()
            edges.append((int(source), int(target), int(timestamp)))

    # Sort edges by timestamp
    edges.sort(key=lambda x: x[2])

    # Extract timestamps and determine split points
    timestamps = [edge[2] for edge in edges]
    quartiles = np.percentile(timestamps, [25, 50, 75])

    # Initialize subgraphs with all nodes
    all_nodes = set()
    for edge in edges:
        all_nodes.update([edge[0], edge[1]])

    subgraphs = [{node: [] for node in all_nodes} for _ in range(4)]

    # Distribute edges into subgraphs based on timestamp intervals
    for edge in edges:
        if edge[2] <= quartiles[0]:
            subgraphs[0][edge[0]].append(edge)
            subgraphs[0][edge[1]].append(edge)
        elif edge[2] <= quartiles[1]:
            subgraphs[1][edge[0]].append(edge)
            subgraphs[1][edge[1]].append(edge)
        elif edge[2] <= quartiles[2]:
            subgraphs[2][edge[0]].append(edge)
            subgraphs[2][edge[1]].append(edge)
        else:
            subgraphs[3][edge[0]].append(edge)
            subgraphs[3][edge[1]].append(edge)

    # Remove isolated nodes and relabel node IDs in each subgraph
    for i, subgraph in enumerate(subgraphs):
        node_map = {}
        new_edges = []
        current_id = 0
        
        # Remove isolated nodes
        for node in list(subgraph.keys()):
            if not subgraph[node]:
                del subgraph[node]

        # Relabel nodes
        for node in subgraph.keys():
            if node not in node_map:
                node_map[node] = current_id
                current_id += 1

        for edges_list in subgraph.values():
            for edge in edges_list:
                source, target, timestamp = edge
                new_source = node_map[source]
                new_target = node_map[target]
                new_edges.append((new_source, new_target, timestamp))

        # Output the relabeled subgraph
        output_file = f'subgraph_{i + 1}.txt'
        with open(output_file, 'w') as out_file:
            for edge in new_edges:
                out_file.write(f"{edge[0]} {edge[1]} {edge[2]}\n")

    print("Subgraphs have been created, relabeled, and saved to files.")

# Example usage
if __name__ == "__main__":
    file_path = 'CollegeMsg.txt'
    split_graph_and_relabel_with_all_nodes(file_path)
