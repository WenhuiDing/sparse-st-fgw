"""The validated full-atlas configuration.

Algorithmic fields are intentionally not exposed as CLI switches. Changing one
creates a new experiment rather than reproducing the reported best run.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BestConfig:
    seed: int = 0
    pca_components: int = 30
    pca_batch_size: int = 4096
    spatial_neighbors: int = 12
    gnn_norm: bool = True
    gnn_activation: bool = True
    gnn_dropout: float = 0.1
    gnn_hidden: int = 64
    gnn_epochs: int = 5
    gnn_batch_size: int = 8192
    gnn_lr: float = 0.002
    gnn_mask_probability: float = 0.15
    gnn_consistency_weight: float = 0.1
    temporal_knn: int = 32
    temporal_cap: int = 256
    temporal_topk: int = 3
    temporal_epsilon: float = 0.002
    rho_a: float = 0.005
    rho_b: float = 0.30
    temporal_iterations: int = 1000
    temporal_solver: str = "masked_fgw"
    temporal_fgw_weight: float = 0.6
    temporal_fgw_outer_iterations: int = 5
    temporal_fgw_inner_iterations: int = 300
    temporal_fgw_step: float = 0.25
    final_knn: int = 64
    final_cap: int = 256
    final_epsilon: float = 0.002
    warmstart_iterations: int = 1000
    fgw_weight: float = 0.6
    fgw_outer_iterations: int = 5
    fgw_inner_iterations: int = 300
    fgw_step: float = 0.25
    # Logical JAX device after CUDA_VISIBLE_DEVICES has been applied.
    ot_device: str = "cuda:0"

    def to_dict(self) -> dict:
        return asdict(self)
