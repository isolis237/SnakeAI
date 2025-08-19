import numpy as np
import pygame as pg
from NN.layer import Dense, Relu
from dataclasses import dataclass

class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], relu_on_last: bool = False):
        """
        layer_sizes like [in, h1, h2, out]
        """
        from NN.layer import Dense, Relu  # keep view-free
        self.shape = list(layer_sizes)
        self.layers = []

        last_linear_idx = len(layer_sizes) - 2
        for idx, (inp, out) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            self.layers.append(Dense(input_size=inp, output_size=out))
            use_relu = (idx < last_linear_idx) or relu_on_last
            if use_relu:
                self.layers.append(Relu())

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def __str__(self):
        desc = ["Neural Network:"]
        for layer in self.layers:
            if isinstance(layer, Dense):
                desc.append(f"  Dense({layer.input_size} → {layer.output_size})")
            else:
                desc.append(f"  {layer.__class__.__name__}")
        return "\n".join(desc)

    def activations(self, x: np.ndarray) -> list[np.ndarray]:
        """
        Return per-layer activations matching self.shape:
        [input, layer1_act, layer2_act, ..., output_act]
        Uses post-ReLU values where ReLU exists (or linear if not).
        """
        a = np.asarray(x).reshape(-1)  # ensure 1D
        acts: list[np.ndarray] = [a.copy()]

        i = 0
        L = len(self.layers)
        while i < L:
            dense = self.layers[i]
            if not isinstance(dense, Dense):
                raise RuntimeError("Layer sequence expected to be Dense(+Relu?) blocks.")
            z = dense.forward(a)

            # If next layer is Relu, apply it; else leave as linear
            if i + 1 < L and isinstance(self.layers[i + 1], Relu):
                a = self.layers[i + 1].forward(z)
                i += 2
            else:
                a = z
                i += 1

            acts.append(np.asarray(a).reshape(-1))

        if len(acts) != len(self.shape):
            # Safety: keep shapes aligned even if model/view drift
            raise RuntimeError(f"activations list length {len(acts)} != network.shape length {len(self.shape)}")
        return acts


# -----------------------
# View config (single spot)
# -----------------------
@dataclass
class ViewStyle:
    canvas_bg_rgba: tuple[int, int, int, int] = (0, 0, 0, 0)

    # Node look
    node_radius: int = 32
    node_fill: tuple[int, int, int] = (220, 220, 220)
    node_border: tuple[int, int, int] = (100, 100, 100)
    node_border_width: int = 2

    # Layout
    margin_ratio_x: float = 0.12
    margin_ratio_y: float = 0.12

    # Edge look
    edge_neutral_gray: tuple[int, int, int] = (180, 180, 180)
    edge_color_neg: tuple[int, int, int] = (255, 0, 0)   # -1
    edge_color_pos: tuple[int, int, int] = (0, 200, 0)   # +1
    edge_min_width: int = 1
    edge_max_width: int = 4

    # Text (node values)
    font_name: str | None = None   # default font
    font_size: int = 24
    text_color: tuple[int, int, int] = (130, 130, 30)
    value_decimals: int = 3
    show_values: bool = True

class NeuralNetworkView(pg.sprite.Sprite):
    def __init__(self, network: NeuralNetwork, canvas_size: tuple[int, int], style: ViewStyle | None = None):
        """
        network: the core (pure) NN to visualize
        canvas_size: (width, height)
        style: optional ViewStyle configuration
        """
        super().__init__()
        self.network = network
        self.style = style or ViewStyle()

        self.image = pg.Surface(canvas_size, pg.SRCALPHA).convert_alpha()
        self.rect = self.image.get_rect(topleft=(0, 0))
        self.font = pg.font.SysFont(self.style.font_name, self.style.font_size)

        self.node_centers: list[list[tuple[int, int]]] = self._compute_node_centers(canvas_size, network.shape)
        # Optional runtime data to show inside nodes (list of arrays matching layer sizes)
        self.node_values: list[np.ndarray] | None = None

        self.redraw()  # initial render

    # ---------- Public API ----------
    def set_node_values(self, values_by_layer: list[np.ndarray] | None) -> None:
        """
        values_by_layer must have the same lengths as network.shape, and each array length must match the node count.
        Pass None to hide values.
        """
        if values_by_layer is not None:
            if len(values_by_layer) != len(self.network.shape):
                raise ValueError("node_values length must match number of layers")
            for arr, expected in zip(values_by_layer, self.network.shape):
                if len(arr) != expected:
                    raise ValueError("node_values per layer must match layer size")
        self.node_values = values_by_layer
        self.redraw()

    def redraw(self) -> None:
        """Re-render edges and nodes using current weights and node values."""
        self.image.fill(self.style.canvas_bg_rgba)
        self._draw_edges()
        self._draw_nodes(with_values=self.style.show_values)

    # ---------- Layout ----------
    def _compute_node_centers(self, canvas_size: tuple[int, int], layer_sizes: list[int]) -> list[list[tuple[int, int]]]:
        width, height = canvas_size
        margin_x = self.style.margin_ratio_x * width
        margin_y = self.style.margin_ratio_y * height

        num_layers = len(layer_sizes)
        if num_layers == 1:
            layer_x_positions = [width // 2]
        else:
            layer_x_positions = [
                int(margin_x + i * (width - 2 * margin_x) / (num_layers - 1))
                for i in range(num_layers)
            ]

        centers: list[list[tuple[int, int]]] = []
        for layer_index, node_count in enumerate(layer_sizes):
            if node_count == 1:
                y_positions = [height // 2]
            else:
                y_positions = [
                    int(margin_y + j * (height - 2 * margin_y) / (node_count - 1))
                    for j in range(node_count)
                ]
            centers.append([(layer_x_positions[layer_index], y) for y in y_positions])
        return centers

    # ---------- Edges ----------
    def _collect_dense_weight_matrices(self) -> list[np.ndarray]:
        """
        Returns weight matrices (shape: in_nodes x out_nodes) in order of Dense layers.
        """
        matrices: list[np.ndarray] = []
        for layer in getattr(self.network, "layers", []):
            weights = getattr(layer, "weights", None)
            if weights is not None:
                matrices.append(np.asarray(weights))
        return matrices

    def _weight_to_color(self, weight_value: float) -> tuple[int, int, int]:
        """
        Map weight in [-1,1] to red (neg) -> gray (0) -> green (pos).
        """
        w = float(np.clip(weight_value, -1.0, 1.0))
        if w < 0:
            t = w + 1.0  # [-1,0] -> [0,1]
            r = int((1 - t) * self.style.edge_color_neg[0] + t * self.style.edge_neutral_gray[0])
            g = int((1 - t) * self.style.edge_color_neg[1] + t * self.style.edge_neutral_gray[1])
            b = int((1 - t) * self.style.edge_color_neg[2] + t * self.style.edge_neutral_gray[2])
        else:
            t = w  # [0,1] -> [0,1]
            r = int((1 - t) * self.style.edge_neutral_gray[0] + t * self.style.edge_color_pos[0])
            g = int((1 - t) * self.style.edge_neutral_gray[1] + t * self.style.edge_color_pos[1])
            b = int((1 - t) * self.style.edge_neutral_gray[2] + t * self.style.edge_color_pos[2])
        return (r, g, b)

    def _edge_width_from_weight(self, abs_weight: float, layer_max_abs: float) -> int:
        if layer_max_abs <= 0:
            return self.style.edge_min_width
        t = np.clip(abs_weight / layer_max_abs, 0.0, 1.0)
        return int(self.style.edge_min_width + t * (self.style.edge_max_width - self.style.edge_min_width))

    def _draw_edges(self) -> None:
        dense_weight_matrices = self._collect_dense_weight_matrices()
        num_connections = len(self.node_centers) - 1

        for layer_index in range(num_connections):
            source_centers = self.node_centers[layer_index]
            target_centers = self.node_centers[layer_index + 1]

            weight_matrix = dense_weight_matrices[layer_index] if layer_index < len(dense_weight_matrices) else None
            layer_max_abs = float(np.abs(weight_matrix).max()) if (weight_matrix is not None and weight_matrix.size) else 0.0

            for src_idx, (src_x, src_y) in enumerate(source_centers):
                for tgt_idx, (tgt_x, tgt_y) in enumerate(target_centers):
                    if weight_matrix is None:
                        color = self.style.edge_neutral_gray
                        width = self.style.edge_min_width
                    else:
                        w = float(weight_matrix[src_idx, tgt_idx])
                        color = self._weight_to_color(w)
                        width = self._edge_width_from_weight(abs(w), layer_max_abs)

                    pg.draw.line(self.image, color, (src_x, src_y), (tgt_x, tgt_y), width)

    # ---------- Nodes ----------
    def _draw_nodes(self, with_values: bool = True) -> None:
        for layer_index, centers in enumerate(self.node_centers):
            for node_index, (cx, cy) in enumerate(centers):
                pg.draw.circle(self.image, self.style.node_fill, (cx, cy), self.style.node_radius)
                pg.draw.circle(
                    self.image,
                    self.style.node_border,
                    (cx, cy),
                    self.style.node_radius,
                    width=self.style.node_border_width,
                )

                if with_values and self.node_values is not None:
                    value_array = self.node_values[layer_index]
                    # Safe guard: if value array is shorter, skip
                    if node_index < len(value_array):
                        val = float(value_array[node_index])
                        text = f"{val:.{self.style.value_decimals}f}"
                        text_surface = self.font.render(text, True, self.style.text_color)
                        text_rect = text_surface.get_rect(center=(cx, cy))
                        self.image.blit(text_surface, text_rect)
