import torch
import torch.nn as nn
import torch.optim as optim
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set, find_ab_params
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Union, Tuple, List
from sklearn.base import BaseEstimator, TransformerMixin

def get_umap_graph(
    T_features: Optional[np.ndarray] = None,
    D_matrix: Optional[np.ndarray] = None,
    n_neighbors: int = 15,
    metric: str = "euclidean",
) -> tuple[torch.Tensor, float, float]:
    """
    Constructs the high-dimensional UMAP connectivity graph from either
    features or a precomputed distance matrix.
    """
    if T_features is None and D_matrix is None:
        raise ValueError("Must provide either T_features or D_matrix.")
    if T_features is not None and D_matrix is not None:
        raise ValueError("Cannot provide both T_features and D_matrix.")

    random_state = np.random.RandomState(42)

    if D_matrix is not None:
        knn_indices, knn_dists, _ = nearest_neighbors(
            D_matrix,
            n_neighbors=n_neighbors,
            metric="precomputed",
            metric_kwds={},
            angular=False,
            random_state=random_state,
        )
        v_ij_sparse, _, _ = fuzzy_simplicial_set(
            X=D_matrix,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric="precomputed",
            metric_kwds={},
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            angular=False,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
        )
    else:
        knn_indices, knn_dists, _ = nearest_neighbors(
            T_features,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds={},
            angular=False,
            random_state=random_state,
        )
        v_ij_sparse, _, _ = fuzzy_simplicial_set(
            X=T_features,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            metric_kwds={},
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            angular=False,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
        )

    v_ij = torch.tensor(v_ij_sparse.toarray(), dtype=torch.float32)
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    return v_ij, float(a), float(b)


def _prepare_scale_init(
    scale_init: Optional[Union[np.ndarray, torch.Tensor]],
    K_restarts: int,
    N_dim: int,
) -> torch.Tensor:
    """
    Returns log-scales of shape (K_restarts, N_dim).

    scale_init is specified in the ordinary positive scale domain.
    Unspecified restarts are initialized with scale = 1.
    """
    log_scale = torch.zeros(K_restarts, N_dim, dtype=torch.float32)

    if scale_init is None:
        return log_scale

    scale_init = torch.as_tensor(scale_init, dtype=torch.float32).detach().cpu()

    if scale_init.ndim != 2 or scale_init.shape[1] != N_dim:
        raise ValueError(
            f"scale_init must have shape (K_init, {N_dim}), "
            f"got {tuple(scale_init.shape)}."
        )

    K_init = scale_init.shape[0]
    if K_init > K_restarts:
        raise ValueError("scale_init has more restarts than K_restarts.")
    if torch.any(scale_init <= 0):
        raise ValueError("All scale_init values must be strictly positive.")

    log_scale[:K_init] = torch.log(scale_init)
    return log_scale


class TopologicalFilterBatch(nn.Module):
    """
    Batched multi-dimensional topological spatial filter.

    Each embedding coordinate is

        y[i, d] = scale[d] * log(w[d]^T C[i] w[d]),

    where scale[d] > 0 is learned independently for every restart and dimension.
    """

    def __init__(
        self,
        M: int,
        N_dim: int,
        K_restarts: int,
        w_init: Optional[torch.Tensor] = None,
        scale_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__()
        self.M = M
        self.N_dim = N_dim
        self.K = K_restarts

        w_tensor = torch.randn(K_restarts, N_dim, M, dtype=torch.float32)

        if w_init is not None:
            w_init = torch.as_tensor(
                w_init, dtype=torch.float32
            ).detach().cpu()

            if w_init.ndim != 3 or w_init.shape[1:] != (N_dim, M):
                raise ValueError(
                    f"w_init must have shape (K_init, {N_dim}, {M}), "
                    f"got {tuple(w_init.shape)}."
                )

            K_init = w_init.shape[0]
            if K_init > K_restarts:
                raise ValueError("w_init has more restarts than K_restarts.")

            w_tensor[:K_init] = w_init

        self.w = nn.Parameter(w_tensor)
        self.log_scale = nn.Parameter(
            _prepare_scale_init(scale_init, K_restarts, N_dim)
        )

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        power = torch.einsum(
            "kdm,nml,kdl->knd",
            self.w,
            C,
            self.w,
        )
        y_raw = torch.log(power.clamp_min(1e-8))

        # (K, D) -> (K, 1, D), broadcasting over epochs
        return y_raw * self.scale.unsqueeze(1)


def umap_cross_entropy_loss(
    y: torch.Tensor,
    v_ij: torch.Tensor,
    a: float,
    b: float,
) -> torch.Tensor:
    """
    Returns one UMAP cross-entropy loss per restart.

    y shape:
        (K_restarts, N_epochs, N_dim)
    """
    _, N_epochs, _ = y.shape

    diff = y.unsqueeze(2) - y.unsqueeze(1)
    dist_sq = torch.sum(diff.square(), dim=-1)

    w_kij = 1.0 / (
        1.0 + a * torch.pow(dist_sq + 1e-12, b)
    )

    w_kij_clamped = torch.clamp(
        w_kij,
        min=1e-7,
        max=1.0 - 1e-7,
    )
    v_ij_clamped = torch.clamp(
        v_ij,
        min=1e-7,
        max=1.0 - 1e-7,
    )

    v_ij_bc = v_ij_clamped.unsqueeze(0)

    term1 = v_ij_bc * torch.log(
        v_ij_bc / w_kij_clamped
    )
    term2 = (1.0 - v_ij_bc) * torch.log(
        (1.0 - v_ij_bc)
        / (1.0 - w_kij_clamped)
    )
    
    loss_matrix = term1 + term2
    
    mask = ~torch.eye(
        N_epochs,
        dtype=torch.bool,
        device=y.device,
    )
    masked_loss = loss_matrix * mask.unsqueeze(0)
    
    valid_pairs = N_epochs * (N_epochs - 1)
    return masked_loss.sum(dim=(1, 2)) / valid_pairs


def _scale_regularization_per_restart(
    log_scale: torch.Tensor,
    scale_reg: float,
) -> torch.Tensor:
    """
    A weak penalty around scale = 1, i.e. log_scale = 0.

    Returns shape (K_restarts,).
    """
    if scale_reg <= 0:
        return torch.zeros(
            log_scale.shape[0],
            device=log_scale.device,
            dtype=log_scale.dtype,
        )
    
    return scale_reg * log_scale.square().mean(dim=1)


class TopologicalSpatialFilter(BaseEstimator, TransformerMixin):
    """
    Scikit-learn совместимый класс для топологической пространственной фильтрации.
    Логика разделена: 
      - fit() формирует целевой топологический граф на основе D_matrices и меток.
      - transform() ищет оптимальные пространственные фильтры для переданных ковариационных матриц C.
    """
    def __init__(
        self,
        N_dim: int = 2,
        K_restarts: int = 5,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        target_metric: str = "categorical",
        target_weight: float = 0.5,
        unknown_label: int = -1,
        epochs: int = 500,
        lr: float = 0.01,
        scale_reg: float = 1e-4,
        log_scale_min: float = -5.0,
        log_scale_max: float = 5.0,
        w_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
        scale_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.N_dim = N_dim
        self.K_restarts = K_restarts
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.target_metric = target_metric
        self.target_weight = target_weight
        self.unknown_label = unknown_label
        self.epochs = epochs
        self.lr = lr
        self.scale_reg = scale_reg
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.w_init = w_init
        self.scale_init = scale_init
        self.verbose = verbose
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Атрибуты целевой топологии (заполняются в fit)
        self.v_ij_ = None
        self.umap_a_ = None
        self.umap_b_ = None
        
        # Обученные параметры фильтров (заполняются в transform)
        self.w_opt_ = None
        self.scales_opt_ = None
        self.loss_history_ = None
        self.individual_losses_ = None

    def _combine_distance_matrices(self, D_matrices: List[np.ndarray]) -> torch.Tensor:
        """
        Строит базовые графы по матрицам расстояний и объединяет их.
        """
        v_ij_combined = None
        
        for D in D_matrices:
            v_ij, a, b = get_umap_graph(
                D_matrix=D, 
                n_neighbors=self.n_neighbors,
                metric=self.metric
            )
            self.umap_a_, self.umap_b_ = a, b 
            v_ij = v_ij.to(self.device)
            
            if v_ij_combined is None:
                v_ij_combined = v_ij
            else:
                v_ij_combined = v_ij_combined + v_ij - (v_ij_combined * v_ij)
                
        return v_ij_combined

    def fit(self, D_matrices: List[np.ndarray], y: Optional[Union[np.ndarray, list]] = None):
        """
        Формирует топологическую структуру данных.
        
        :param D_matrices: Список матриц попарных расстояний (N_epochs, N_epochs).
        :param y: Внешние метки (категориальные или непрерывные) формы (N_epochs,).
        """
        if not D_matrices:
            raise ValueError("D_matrices must be provided to build the topology.")

        if self.verbose:
            print(f"Building combined UMAP graph from {len(D_matrices)} distance matrices...")
            
        v_ij = self._combine_distance_matrices(D_matrices)

        # UMAP TARGET SIMPLICIAL SET INTERSECTION
        if y is not None:
            if self.verbose:
                print(f"Applying supervised topological intersection ({self.target_metric})...")
            
            labels = np.asarray(y, dtype=np.float32)
            
            if labels.ndim == 1:
                if self.target_metric == "categorical":
                    if self.target_weight < 1.0:
                        far_dist = 2.5 * (1.0 / (1.0 - self.target_weight))
                    else:
                        far_dist = 1.0e12
        
                    labels_t = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
                    is_unknown = (labels_t == self.unknown_label)
                    same_class = (labels_t.unsqueeze(1) == labels_t.unsqueeze(0))
        
                    penalty = torch.ones_like(v_ij)
                    unknown_mask = is_unknown.unsqueeze(1) | is_unknown.unsqueeze(0)
                    penalty[unknown_mask] = np.exp(-1.0)
                    diff_class_mask = (~unknown_mask) & (~same_class)
                    penalty[diff_class_mask] = float(np.exp(-far_dist))
        
                    v_ij = v_ij * penalty
                else:
                    labels_2d = labels.reshape(-1, 1)
                    target_v_ij, _, _ = get_umap_graph(
                        T_features=labels_2d,
                        n_neighbors=self.n_neighbors,
                        metric=self.target_metric
                    )
                    target_v_ij = target_v_ij.to(self.device)
        
                    eps = 1e-8
                    if self.target_weight < 0.5:
                        power = self.target_weight / (1.0 - self.target_weight)
                        v_ij = v_ij * torch.pow(target_v_ij.clamp_min(eps), power)
                    else:
                        power = 0.0 if self.target_weight == 1.0 else (1.0 - self.target_weight) / self.target_weight
                        v_ij = torch.pow(v_ij.clamp_min(eps), power) * target_v_ij
                        
            elif labels.ndim == 2:
                target_v_ij, _, _ = get_umap_graph(
                    T_features=labels,
                    n_neighbors=self.n_neighbors,
                    metric=self.target_metric
                )
                target_v_ij = target_v_ij.to(self.device)
        
                eps = 1e-8
                if self.target_weight < 0.5:
                    power = self.target_weight / (1.0 - self.target_weight)
                    v_ij = v_ij * torch.pow(target_v_ij.clamp_min(eps), power)
                else:
                    power = 0.0 if self.target_weight == 1.0 else (1.0 - self.target_weight) / self.target_weight
                    v_ij = torch.pow(v_ij.clamp_min(eps), power) * target_v_ij
            else:
                raise ValueError("Labels must be 1D or 2D array.")
        
            # Восстановление локальной связности
            row_max = v_ij.max(dim=1, keepdim=True).values.clamp_min(1e-8)
            v_ij = v_ij / row_max
            v_ij_t = v_ij.t()
            v_ij = v_ij + v_ij_t - (v_ij * v_ij_t)

        self.v_ij_ = v_ij
        return self

    def transform(self, C: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Ищет пространственные фильтры для матриц ковариации на основе графа из fit().
        
        :param C: Нефильтрованные матрицы ковариации, форма (N_epochs, M_channels, M_channels).
        :return: Оптимальная проекция (N_epochs, N_dim)
        """
        if self.v_ij_ is None:
            raise RuntimeError("Topology not built. Call .fit() with D_matrices first.")

        C_tensor = torch.as_tensor(C, dtype=torch.float32).to(self.device)
        N_epochs, M_channels, M_channels_2 = C_tensor.shape
        
        if M_channels != M_channels_2:
            raise ValueError("C must have shape (N_epochs, M, M).")

        model = TopologicalFilterBatch(
            M=M_channels,
            N_dim=self.N_dim,
            K_restarts=self.K_restarts,
            w_init=self.w_init,
            scale_init=self.scale_init,
        ).to(self.device)

        optimizer = optim.AdamW(
            [
                {"params": [model.w], "weight_decay": 1e-2},
                {"params": [model.log_scale], "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=10,
            factor=0.5,
        )

        loss_history = torch.zeros(self.epochs, self.K_restarts, dtype=torch.float32)

        pbar = tqdm(range(self.epochs), disable=not self.verbose, desc="Finding Spatial Filters")

        for epoch in pbar:
            optimizer.zero_grad()

            y_pred = model(C_tensor)
            losses = umap_cross_entropy_loss(
                y_pred,
                self.v_ij_,
                self.umap_a_,
                self.umap_b_,
            )

            scale_penalty = _scale_regularization_per_restart(
                model.log_scale,
                self.scale_reg,
            )
            objective = losses + scale_penalty
            total_objective = objective.mean()

            total_objective.backward()
            optimizer.step()

            with torch.no_grad():
                w_norms = torch.linalg.vector_norm(model.w, dim=2, keepdim=True)
                model.w.div_(w_norms.clamp_min(1e-8))
                model.log_scale.clamp_(min=self.log_scale_min, max=self.log_scale_max)

            loss_history[epoch] = losses.detach().cpu()
            current_loss = losses.mean().item()
            scheduler.step(total_objective.item())

            pbar.set_postfix(
                mean_umap_loss=f"{current_loss:.4f}",
                mean_scale=f"{model.scale.mean().item():.3f}",
            )

        with torch.no_grad():
            final_y = model(C_tensor)
            final_losses = umap_cross_entropy_loss(
                final_y,
                self.v_ij_,
                self.umap_a_,
                self.umap_b_,
            ).detach().cpu()

            w_final = model.w.detach().cpu().clone()
            scales_final = model.scale.detach().cpu().clone()
            individual_losses = torch.zeros(self.K_restarts, self.N_dim, dtype=torch.float32)

            for k in range(self.K_restarts):
                for d in range(self.N_dim):
                    y_1d = final_y[k:k + 1, :, d:d + 1]
                    individual_losses[k, d] = umap_cross_entropy_loss(
                        y_1d, self.v_ij_, self.umap_a_, self.umap_b_
                    ).item()

                sorted_dims = torch.argsort(individual_losses[k])
                individual_losses[k] = individual_losses[k, sorted_dims]
                w_final[k] = w_final[k, sorted_dims]
                scales_final[k] = scales_final[k, sorted_dims]

            sorted_restarts = torch.argsort(final_losses)

            # Сохраняем все варианты, отсортированные по качеству
            self.w_opt_ = w_final[sorted_restarts].numpy()
            self.scales_opt_ = scales_final[sorted_restarts].numpy()
            self.loss_history_ = loss_history[:, sorted_restarts].numpy()
            self.individual_losses_ = individual_losses[sorted_restarts].numpy()
            
            # Извлекаем признаки для самого успешного рестарта (k=0 после сортировки)
            best_y = final_y[sorted_restarts[0]].cpu().numpy()

        # Возвращаем спроецированные данные, фильтры доступны через self.w_opt_
        return best_y