# NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt # needed to show the visualizations
```

## Creating Graphs

| Name  | Code |
|:-:|:-:|
| Simple Undirected Graph | `G = nx.Graph()` |
| Simple Directed Graph | `G = nx.DiGraph()` |
| Undirected Graph (multiple edges between nodes) | `G = nx.MultiGraph()` |
| Directed Graph (multiple edges between nodes) | `G = nx.MultiDiGraph()`|


##

| Name | Description | Code |
|:-:|:-:|:-:|
| Add Edge | if nodes doesn't exist, it's going to create them. | `G.add_edge(node_1, node_2)` |
| Add Edges (from list) | example: `edge_list=[(1,2), (1,3)]` | `G.add_edges_from(edge_list)` |