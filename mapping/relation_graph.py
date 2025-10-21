
from typing import Optional
import clip
import networkx as nx
import numpy as np
import torch
from matplotlib.patches import Rectangle

class RelationGraph:
    
    def __init__(self):
        self.relation_graph = nx.DiGraph()
        self.num_landmarks = 0

    def add_landmark(self, info: dict) -> tuple[int, bool]:
        """
            info -> dictionary with visual features, map location (x,y,z/Layernum), map extent/variance/size (infered from adjacent map location?)
            Returns 
                node id, 
                true if new node is added
        """

        ## to-do - how to update?
        ## update when the incoming visual features have high similarity with an existing node and they have same/adjacent map locations
        ### - can be done during creation or later in a "merge/collapse nodes" stage
        
        # check if a node exists for the same map location
        node = [n for n in self.relation_graph.nodes(data=True) if n[1]["info"]["map_location"] == info["map_location"]]
        if len(node) > 0:
            nx.set_node_attributes(self.relation_graph, {node[0][0]: {"info": info}})
            return node[0][0], False
        else:
            self.relation_graph.add_node(self.num_landmarks, info=info)
            self.num_landmarks += 1
            return (self.num_landmarks - 1), True
        
    def add_relation(self, from_node_id: int, to_node_id: int, relation_info: dict) -> None:
        if self.relation_graph.has_edge(from_node_id, to_node_id):
            # retrieve existing relations
            relation_data = self.relation_graph.get_edge_data(from_node_id, to_node_id)["info"]
            for k,v in relation_info.items():
                relation_data[k] = v
            # update
            nx.set_edge_attributes(self.relation_graph, {(from_node_id, to_node_id): {"info": relation_data}})
        else:
            # add
            self.relation_graph.add_edge(from_node_id, to_node_id, info=relation_info)

    def update_relations(self, from_node_id: int, to_map_locations: list[list[int]], relation_info: dict) -> None:
        for to_map_loc in to_map_locations:
            node = [n for n in self.relation_graph.nodes(data=True) if n[1]["info"]["map_location"] == tuple(to_map_loc)]
            to_node_info = {"map_location": tuple(to_map_loc)}
            if len(node) > 0:
                # nx.set_node_attributes(self.relation_graph, {node[0][0]: to_node_info}) # no need to update map location
                nx.set_edge_attributes(self.relation_graph, {(from_node_id, node[0][0]): {"info": relation_info}})
            else:
                self.relation_graph.add_node(self.num_landmarks, info=to_node_info)
                self.relation_graph.add_edge(from_node_id, self.num_landmarks, info=relation_info)
                self.num_landmarks += 1

    def has_nodes(self, exclude_node_id: Optional[int] = None) -> tuple[list[int], bool]:
        if exclude_node_id:
            nodes_in_graph = [n for n in self.relation_graph.nodes(data=True) if n[0] != exclude_node_id]
        else:
            nodes_in_graph = [n for n in self.relation_graph.nodes(data=True)]
        return nodes_in_graph, len(nodes_in_graph) > 0

    def update_landmark_extent(self, node_id: int, map_extent: list[int], vis_feats) -> None:
        # get existing info
        _data = self.relation_graph.nodes(data=True)[node_id]["info"]
        for k,v in _data.items():
            if k == "map_extent":
                v.append(map_extent)
                break
        
        _data[k] = v
        _data["vis_feats"] = vis_feats

        nx.set_node_attributes(self.relation_graph, {node_id: {"info": _data}})

    def get_landmark_extent(self, node_id: int) -> list[list[int]]:
        # get existing info
        _data = self.relation_graph.nodes(data=True)[node_id]["info"]
        return _data["map_extent"]

    def find_landmark(self, name: str):
        pass

    def find_landmark_relations(self, name: str):
        pass

    def remove_landmark(self, landmark_id: int):
        self.relation_graph.remove_node(landmark_id)

    def remove_relation(self, from_landmark_id: int, to_landmark_id: int):
        self.relation_graph.remove_edge(from_landmark_id, to_landmark_id)

    def reset_graph(self):
        self.relation_graph.clear()

    @property
    def graph(self):
        return self.relation_graph
    
    @property
    def landmark_nodes(self):
        pass

    @property
    def relations(self):
        pass

# grid_viz_nx.py
import os
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def _get_xyz(G: nx.DiGraph, n):
    """Return (x,y,z) from node attribute 'map_location'."""
    x, y, z = G.nodes[n].get("info").get("map_location", (None, None, None))
    if x is None or y is None or z is None:
        raise ValueError(f"Node {n!r} missing 'map_location'=(x,y,z)")
    return int(x), int(y), int(z)

def _extent_count_in_layer(node_attrs, z):
    """
    Count unique (x,y) cells covered by this node on layer z.
    node_attrs should contain 'map_extent' as a list of (x,y) or (x,y,z).
    """
    ext = node_attrs.get("info").get("map_extent", None)
    if not ext:
        return 0
    cells = []
    for t in ext:
        if len(t) == 2:
            # (x,y) only -> treat as present on all layers, or assume current z
            x, y = t
            cells.append((int(x), int(y)))
        elif len(t) == 3:
            x, y, zz = t
            if int(zz) == int(z):
                cells.append((int(x), int(y)))
    return len(set(cells))

def _size_from_count(count, base=200, k=60, sqrt=True, min_sz=120, max_sz=4000):
    """
    Convert cell count -> matplotlib marker size (points^2) for NetworkX.
    base: constant offset; k: scale; sqrt=True keeps growth tame.
    """
    if count <= 0:
        return min_sz
    val = base + (k * (count ** 0.5 if sqrt else count))
    return int(max(min_sz, min(max_sz, val)))

def visualize_grid_digraph(
    G: nx.DiGraph,
    out_dir: str = "graph_imgs",
    dpi: int = 200,
    cell_size: int = 20,          # pixels per grid cell (controls figure size)
    y_down: bool = True,          # True if your grid's y increases downward (image coords)
    draw_labels: bool = True,
    show_edge_labels: bool = True,
    draw_all_layers_overview: bool = True,
    max_labels_per_layer: int = 200,  # avoid clutter on huge graphs
    _pad: int = 100,
    size_from_extent: bool = True,
    draw_footprints: bool = False,          # draw per-cell coverage as faint squares
    footprint_alpha: float = 0.25
):
    """
    Saves one image per z layer:
      - nodes at integer (x,y) grid centers
      - edges drawn only when both endpoints share the layer
      - faint grid lines
    Also saves an 'all_layers.png' overview if requested.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect coords and z-slices
    nodes = list(G.nodes)
    xyz = {n: _get_xyz(G, n) for n in nodes}
    zs = sorted({z for (_, _, z) in xyz.values()})

    # Global bounds (pad by 0.5 so nodes sit inside their cells)
    xs = [x for (x,_,_) in xyz.values()]
    ys = [y for (_,y,_) in xyz.values()]
    xmin, xmax = min(xs), max(max(xs), 500)
    ymin, ymax = min(ys), max(max(ys), 500)

    width_cells  = (xmax - xmin + 1)
    height_cells = (ymax - ymin + 1)
    figsize = (width_cells * cell_size / dpi, height_cells * cell_size / dpi)

    # Color per layer (repeatable)
    rng = np.random.default_rng(0)
    layer_colors = {z: rng.random(3) for z in zs}

    # ----- per-layer renders -----
    for z in zs:
        layer_nodes = [n for n in nodes if xyz[n][2] == z]
        if not layer_nodes:
            continue
        pos = {n: (xyz[n][0], xyz[n][1]) for n in layer_nodes}  # (x,y) in grid coords
        edges_same_layer = [(u, v) for (u, v) in G.edges
                            if xyz.get(u, (0,0,None))[2] == z and xyz.get(v, (0,0,None))[2] == z]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Grid background
        ax.set_xlim(xmin - _pad, xmax + _pad)
        ax.set_ylim(ymin - _pad, ymax + _pad)
        if y_down:
            ax.invert_yaxis()
        ax.set_aspect("equal")
        # ax.set_xticks(np.arange(0, 1000, 1), minor=True)
        # ax.set_yticks(np.arange(0, 1000, 1), minor=True)
        ax.grid(which="minor", color="#dddddd", linewidth=1.5)
        ax.set_xticks([]); ax.set_yticks([])

        if draw_footprints:
            for n in layer_nodes:
                ext = G.nodes[n].get("info").get("map_extent", None)
                if not ext:
                    continue
                for t in ext:
                    if (len(t) == 2) or (len(t) == 3 and int(t[2]) == z):
                        x, y = (t[0], t[1])
                        # cell square centered on integer grid; our axes already use cell edges at +/-0.5
                        r = Rectangle((x-0.5, y-0.5), 1, 1,
                                      facecolor=np.r_[layer_colors[z], footprint_alpha],
                                      edgecolor="none")
                        ax.add_patch(r)

        if size_from_extent:
            sizes = []
            for n in layer_nodes:
                cnt = _extent_count_in_layer(G.nodes[n], z)
                sizes.append(_size_from_count(cnt, base=50))
        else:
            sizes = 150

        # Draw nodes (colored by layer)
        node_color = [layer_colors[z]] * len(layer_nodes)
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=layer_nodes,
            node_color=node_color,
            node_size=sizes,
            edgecolors="black",
            linewidths=0.6,
            ax=ax
        )

        # Labels (optional / capped)
        if draw_labels:
            if max_labels_per_layer is None or len(layer_nodes) <= max_labels_per_layer:
                nx.draw_networkx_labels(
                    G, pos,
                    labels={n: str(n) for n in layer_nodes},
                    font_size=8, font_color="black", ax=ax
                )

        # Directed edges within this layer
        if edges_same_layer:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges_same_layer,
                edge_color="black",
                arrows=True,
                arrowstyle="->",
                arrowsize=20,
                width=1.2,
                connectionstyle="arc3,rad=0.0",
                ax=ax
            )

        ax.set_title(f"Layer z={z}")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"layer_z{z}.png")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    # ----- optional overview (all layers collapsed) -----
    if draw_all_layers_overview:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_xlim(xmin - _pad, xmax + _pad)
        ax.set_ylim(ymin - _pad, ymax + _pad)
        if y_down:
            ax.invert_yaxis()
        ax.set_aspect("equal")
        # ax.set_xticks(np.arange(xmin - 0.5, xmax + 1.5, 1), minor=True)
        # ax.set_yticks(np.arange(ymin - 0.5, ymax + 1.5, 1), minor=True)
        # ax.grid(visible=True, which="both", color="#180505", linewidth=1.5)
        for x in range(xmin - _pad, xmax + _pad):
            ax.axvline(x, color="#ACA9A9", linewidth=0.5)
        for y in range(ymin - _pad, ymax + _pad):
            ax.axhline(y, color="#ACA9A9", linewidth=0.5)
        ax.set_xticks([]); ax.set_yticks([])

        # nodes by layer/color
        for z in zs:
            layer_nodes = [n for n in nodes if xyz[n][2] == z]
            pos = {n: (xyz[n][0], xyz[n][1]) for n in layer_nodes}
            if draw_footprints:
                for n in layer_nodes:
                    ext = G.nodes[n].get("info").get("map_extent", None)
                    if not ext:
                        continue
                    for t in ext:
                        if (len(t) == 2) or (len(t) == 3 and int(t[2]) == z):
                            x, y = (t[0], t[1])
                            # cell square centered on integer grid; our axes already use cell edges at +/-0.5
                            r = Rectangle((x-0.5, y-0.5), 1, 1,
                                        facecolor=np.r_[layer_colors[z], footprint_alpha],
                                        edgecolor="none")
                            ax.add_patch(r)

            if size_from_extent:
                sizes = []
                for n in layer_nodes:
                    cnt = _extent_count_in_layer(G.nodes[n], z)
                    sizes.append(_size_from_count(cnt, base=50))
            else:
                sizes = 150

            nx.draw_networkx_nodes(
                G, pos,
                nodelist=layer_nodes,
                node_color=[layer_colors[z]] * len(layer_nodes),
                label=f"z={z}",
                node_size=sizes,
                edgecolors="black",
                linewidths=0.5,
                ax=ax
            )

        # all edges (same-layer solid, cross-layer dashed + faint)
        pos_all = {n: (xyz[n][0], xyz[n][1]) for n in nodes}
        same = [(u, v) for (u, v) in G.edges if xyz[u][2] == xyz[v][2]]
        cross = [(u, v) for (u, v) in G.edges if xyz[u][2] != xyz[v][2]]

        if same:
            nx.draw_networkx_edges(
                G, pos_all, edgelist=same, edge_color="#333333",
                arrows=True, arrowstyle="->", arrowsize=20, width=1.0, ax=ax
            )
        if cross:
            nx.draw_networkx_edges(
                G, pos_all, edgelist=cross, edge_color="#888888",
                style="dashed", alpha=0.6,
                arrows=True, arrowstyle="->", arrowsize=20, width=0.8, ax=ax
            )

        if draw_labels and (max_labels_per_layer is None or len(nodes) <= max_labels_per_layer):
            nx.draw_networkx_labels(
                G, pos_all, labels={n: str(n) for n in nodes}, font_size=7, ax=ax
            )

        # --- edge labels
        if show_edge_labels:
            same.extend(cross)
            lbls = {
                (u, v): ",".join(list(G[u][v].get("info").keys()))
                for (u, v) in same
            }
            # drop empties
            lbls = {e: t for e, t in lbls.items() if t}
            if lbls:
                nx.draw_networkx_edge_labels(
                    G, pos_all, edge_labels=lbls, font_size=16, rotate=False,
                    label_pos=0.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                    ax=ax
                )

        ax.set_title("All layers (colors by z; dashed = cross-layer edges)")
        plt.tight_layout()
        out_path = os.path.join(out_dir, "all_layers.png")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    print(f"Saved images to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    G = RelationGraph()
    # add nodes with map_location=(x,y,z)
    node1, _ = G.add_landmark(info={"map_location":(200, 300, 0), "map_extent": [(200, 300, 0),(201, 300, 0),(202, 300, 0),(203, 300, 0),(201, 301, 0),(201, 302, 0),(201, 303, 0),(201, 304, 0)]})
    node2, _ = G.add_landmark(info={"map_location":(205, 312, 0), "map_extent": [(205, 312, 0)]})
    node3, _ = G.add_landmark(info={"map_location":(215, 332, 1), "map_extent": [(215, 332, 0)]})

    # directed edges
    G.add_relation(node1, node2, relation_info={"near": True})
    G.add_relation(node2, node3, relation_info={"below": True})

    visualize_grid_digraph(G.relation_graph, out_dir="graph_imgs", y_down=True)

