import torch
import torch.nn as nn
import torch.optim as optim
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set, find_ab_params
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Union, Tuple


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


class NormalizedPatternFilterBatch(nn.Module):
    """
    Batched multi-dimensional normalized pattern spatial filter.

    For each epoch i and component d:

        num = a[d]^T B_i^{-1} C_i B_i^{-1} a[d]
        den = a[d]^T B_i^{-1} a[d]

        y[i, d] = scale[d] * (log(num) - 2 log(den))

    The exponent 2 corresponds to the output power of the unit-gain
    adaptive filter

        w_i = B_i^{-1} a / (a^T B_i^{-1} a).
    """

    def __init__(
        self,
        M: int,
        N_dim: int,
        K_restarts: int,
        a_init: Optional[torch.Tensor] = None,
        scale_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__()
        self.M = M
        self.N_dim = N_dim
        self.K = K_restarts

        a_tensor = (
            torch.randn(K_restarts, N_dim, M, dtype=torch.float32) * 0.1
        )

        if a_init is not None:
            a_init = torch.as_tensor(
                a_init, dtype=torch.float32
            ).detach().cpu()

            if a_init.ndim != 3 or a_init.shape[1:] != (N_dim, M):
                raise ValueError(
                    f"a_init must have shape (K_init, {N_dim}, {M}), "
                    f"got {tuple(a_init.shape)}."
                )

            K_init = a_init.shape[0]
            if K_init > K_restarts:
                raise ValueError("a_init has more restarts than K_restarts.")

            a_tensor[:K_init] = a_init

        self.a = nn.Parameter(a_tensor)
        self.log_scale = nn.Parameter(
            _prepare_scale_init(scale_init, K_restarts, N_dim)
        )

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    def forward(
        self,
        C_num: torch.Tensor,
        C_den: torch.Tensor,
    ) -> torch.Tensor:
        num = torch.einsum(
            "kdm,nml,kdl->knd",
            self.a,
            C_num,
            self.a,
        )
        den = torch.einsum(
            "kdm,nml,kdl->knd",
            self.a,
            C_den,
            self.a,
        )

        eps = 1e-8
        y_raw = (
            torch.log(num.clamp_min(eps))
            - 2.0 * torch.log(den.clamp_min(eps))
        )

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


def fit_filters(
    C: Union[np.ndarray, torch.Tensor],
    N_dim: int,
    K_restarts: int,
    T_features: Optional[np.ndarray] = None,
    D_matrix: Optional[np.ndarray] = None,
    labels: Optional[Union[np.ndarray, list]] = None,  
    unknown_label: int = -1,                           
    w_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
    scale_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    epochs: int = 500,
    lr: float = 0.01,
    scale_reg: float = 1e-4,
    log_scale_min: float = -5.0,
    log_scale_max: float = 5.0,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Trains K restarts of an N_dim-dimensional topological spatial filter.

    Returns
    -------
    w_opt:
        (K_restarts, N_dim, M_channels)
    scales_opt:
        Learned positive coordinate scales,
        shape (K_restarts, N_dim)
    final_losses_sorted:
        Pure final UMAP losses, shape (K_restarts,)
    loss_history_sorted:
        Pure UMAP-loss history, shape (epochs, K_restarts)
    individual_losses_sorted:
        One-dimensional scaled losses,
        shape (K_restarts, N_dim)
    """
    if T_features is None and D_matrix is None:
        raise ValueError(
            "Must provide either T_features or D_matrix to fit_filters."
        )

    C = torch.as_tensor(C, dtype=torch.float32)

    N_epochs, M_channels, M_channels_2 = C.shape
    if M_channels != M_channels_2:
        raise ValueError("C must have shape (N_epochs, M, M).")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    C = C.to(device)

    if verbose:
        print(f"Building UMAP graph and moving data to {device}...")

    # Строим базовый топологический граф по данным
    v_ij, umap_a, umap_b = get_umap_graph(
        T_features=T_features,
        D_matrix=D_matrix,
        n_neighbors=n_neighbors,
        metric=metric,
    )
    v_ij = v_ij.to(device)

    # =====================================================================
    # UMAP CATEGORICAL SIMPLICIAL SET INTERSECTION
    # =====================================================================
    if labels is not None:
        if verbose:
            print("Applying supervised topological intersection based on labels...")
        
        labels_t = torch.as_tensor(labels, dtype=torch.float32, device=device)
        
        # Матрица совпадения классов: 1.0 если классы одинаковые, 0.0 если разные
        same_class = (labels_t.unsqueeze(1) == labels_t.unsqueeze(0)).float()
        
        # Обработка неизвестных меток (Semi-supervised подход UMAP)
        is_unknown = (labels_t == unknown_label).float()
        
        # Если хотя бы одна из эпох в паре не размечена, мы сохраняем исходную связь
        keep_original = (is_unknown.unsqueeze(1) + is_unknown.unsqueeze(0) > 0).float()
        
        # Итоговая маска: оставляем связь, если классы совпали ИЛИ класс неизвестен
        intersection_mask = torch.max(same_class, keep_original)
        
        # Поэлементное умножение (пересечение графов)
        v_ij = v_ij * intersection_mask
    # =====================================================================

    model = TopologicalFilterBatch(
        M=M_channels,
        N_dim=N_dim,
        K_restarts=K_restarts,
        w_init=w_init,
        scale_init=scale_init,
    ).to(device)

    # Explicitly exclude log_scale from AdamW weight decay.
    # Its regularization is controlled by scale_reg.
    optimizer = optim.AdamW(
        [
            {"params": [model.w], "weight_decay": 1e-2},
            {"params": [model.log_scale], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=10,
        factor=0.5,
    )

    loss_history = torch.zeros(
        epochs,
        K_restarts,
        dtype=torch.float32,
    )

    pbar = tqdm(
        range(epochs),
        disable=not verbose,
        desc="Optimizing Filters",
    )

    for epoch in pbar:
        optimizer.zero_grad()

        y = model(C)
        losses = umap_cross_entropy_loss(
            y,
            v_ij,
            umap_a,
            umap_b,
        )

        scale_penalty = _scale_regularization_per_restart(
            model.log_scale,
            scale_reg,
        )
        objective = losses + scale_penalty
        total_objective = objective.mean()

        total_objective.backward()
        optimizer.step()

        with torch.no_grad():
            # This fixes an otherwise unidentifiable spatial-vector norm.
            # It does not set the embedding-coordinate scale.
            w_norms = torch.linalg.vector_norm(
                model.w,
                dim=2,
                keepdim=True,
            )
            model.w.div_(w_norms.clamp_min(1e-8))

            model.log_scale.clamp_(
                min=log_scale_min,
                max=log_scale_max,
            )

        loss_history[epoch] = losses.detach().cpu()

        current_loss = losses.mean().item()
        scheduler.step(total_objective.item())

        pbar.set_postfix(
            mean_umap_loss=f"{current_loss:.4f}",
            mean_scale=f"{model.scale.mean().item():.3f}",
        )

    with torch.no_grad():
        final_y = model(C)
        final_losses = umap_cross_entropy_loss(
            final_y,
            v_ij,
            umap_a,
            umap_b,
        ).detach().cpu()

        w_final = model.w.detach().cpu().clone()
        scales_final = model.scale.detach().cpu().clone()

        individual_losses = torch.zeros(
            K_restarts,
            N_dim,
            dtype=torch.float32,
        )

        for k in range(K_restarts):
            for d in range(N_dim):
                y_1d = final_y[k:k + 1, :, d:d + 1]
                individual_losses[k, d] = (
                    umap_cross_entropy_loss(
                        y_1d,
                        v_ij,
                        umap_a,
                        umap_b,
                    ).item()
                )

            sorted_dims = torch.argsort(
                individual_losses[k]
            )

            individual_losses[k] = (
                individual_losses[k, sorted_dims]
            )
            w_final[k] = w_final[k, sorted_dims]
            scales_final[k] = scales_final[k, sorted_dims]

        sorted_restarts = torch.argsort(final_losses)

        w_opt = w_final[sorted_restarts].numpy()
        scales_opt = scales_final[sorted_restarts].numpy()
        final_losses_sorted = (
            final_losses[sorted_restarts].numpy()
        )
        loss_history_sorted = (
            loss_history[:, sorted_restarts].numpy()
        )
        individual_losses_sorted = (
            individual_losses[sorted_restarts].numpy()
        )

    return (
        w_opt,
        scales_opt,
        final_losses_sorted,
        loss_history_sorted,
        individual_losses_sorted,
    )