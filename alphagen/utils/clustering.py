import torch
import random
from typing import List, Tuple
from alphagen_qlib.stock_data import StockData, FeatureType

def _extract_dtw_features(data: StockData, lookback: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts rolling windows of 'lookback' days, filters NaNs, and normalizes.
    """
    device = data.data.device
    close_idx = int(FeatureType.CLOSE)
    
    start_idx = data.max_backtrack_days - lookback + 1
    end_idx = data.max_backtrack_days + data.n_days
    
    raw_close = data.data[start_idx:end_idx, close_idx, :]
    windows = raw_close.unfold(0, lookback, 1)
    
    n_days, n_stocks, _ = windows.shape
    days_grid = torch.arange(n_days, device=device).unsqueeze(1).expand(n_days, n_stocks)
    stocks_grid = torch.arange(n_stocks, device=device).unsqueeze(0).expand(n_days, n_stocks)
    
    windows_flat = windows.reshape(-1, lookback)
    days_flat = days_grid.reshape(-1)
    stocks_flat = stocks_grid.reshape(-1)
    
    valid_mask = ~windows_flat.isnan().any(dim=1)
    windows_valid = windows_flat[valid_mask]
    days_valid = days_flat[valid_mask]
    stocks_valid = stocks_flat[valid_mask]
    
    # Z-score normalization
    means = windows_valid.mean(dim=1, keepdim=True)
    stds = windows_valid.std(dim=1, keepdim=True) + 1e-8
    windows_norm = (windows_valid - means) / stds
    
    return windows_norm, days_valid, stocks_valid


def batched_dtw_distances(X: torch.Tensor, barycenters: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch GPU wavefront implementation of DTW.
    X: [N, L]
    barycenters: [K, L]
    Returns: [N, K] distance matrix
    """
    N, L = X.shape
    K, _ = barycenters.shape
    device = X.device

    # Expand to [N, K, L, L] cost matrix (squared Euclidean distance)
    cost = (X.view(N, 1, L, 1) - barycenters.view(1, K, 1, L)) ** 2

    # Initialize DP matrix [N, K, L+1, L+1] with infinity
    dp = torch.full((N, K, L + 1, L + 1), float('inf'), device=device, dtype=X.dtype)
    dp[:, :, 0, 0] = 0.0

    # Wavefront parallelization over anti-diagonals
    for d in range(1, 2 * L):
        start_i = max(1, d - L + 1)
        end_i = min(d, L)
        
        i = torch.arange(start_i, end_i + 1, device=device)
        j = d + 1 - i
        
        cost_diag = cost[:, :, i - 1, j - 1]
        
        dp_left = dp[:, :, i, j - 1]
        dp_up = dp[:, :, i - 1, j]
        dp_diag = dp[:, :, i - 1, j - 1]

        min_prev = torch.min(torch.min(dp_left, dp_up), dp_diag)
        dp[:, :, i, j] = cost_diag + min_prev

    return dp[:, :, L, L]


def kmeans(
    dataset: StockData, 
    n_clusters: int = 7, 
    lookback: int = 20, 
    max_iters: int = 20,
    batch_size: int = 50000 
) -> Tuple[torch.Tensor, List[Tuple[List[int], List[int]]]]:
    """
    GPU-Accelerated DTW K-Means with Sampled K-Medoids update.
    Returns:
        barycenters: [K, lookback] tensor to be passed to calc_clusters.
        clusters: A list of (days_list, stocks_list) tuples.
    """
    windows_norm, days_valid, stocks_valid = _extract_dtw_features(dataset, lookback)
    N = windows_norm.shape[0]
    device = dataset.data.device

    print(f"[Clustering] Initializing GPU DTW K-Means on {N} valid pairs...")

    init_indices = random.sample(range(N), n_clusters)
    barycenters = windows_norm[init_indices].clone()

    labels = torch.zeros(N, dtype=torch.long, device=device)

    for iteration in range(max_iters):
        all_distances = []
        for i in range(0, N, batch_size):
            batch_X = windows_norm[i:i + batch_size]
            dists = batched_dtw_distances(batch_X, barycenters)
            all_distances.append(dists)
            
        distances_tensor = torch.cat(all_distances, dim=0)
        new_labels = torch.argmin(distances_tensor, dim=1)
        
        changes = (new_labels != labels).sum().item()
        labels = new_labels
        
        print(f"  Iter {iteration+1}/{max_iters} | Label shifts: {changes}")
        if changes == 0:
            break

        # Update barycenters using Sampled K-Medoids to prevent wave flattening
        sample_size = 500  
        
        for k in range(n_clusters):
            mask = (labels == k)
            cluster_points = windows_norm[mask]
            num_points = cluster_points.shape[0]
            
            if num_points > 0:
                n_samples = min(num_points, sample_size)
                indices = torch.randperm(num_points, device=device)[:n_samples]
                sampled_points = cluster_points[indices]
                
                # Pairwise distance within the sample: shape [n_samples, n_samples]
                dists = batched_dtw_distances(sampled_points, sampled_points)
                
                sum_dists = dists.sum(dim=1)
                medoid_idx = sum_dists.argmin()
                
                barycenters[k] = sampled_points[medoid_idx]

    clusters = []
    for i in range(n_clusters):
        mask = (labels == i)
        c_days = days_valid[mask].tolist()
        c_stocks = stocks_valid[mask].tolist()
        clusters.append((c_days, c_stocks))
        print(f"  Cluster {i}: {len(c_days)} pairs")

    return barycenters, clusters


def calc_clusters(
    barycenters: torch.Tensor, 
    dataset: StockData, 
    lookback: int = 20,
    batch_size: int = 50000
) -> List[Tuple[List[int], List[int]]]:
    """
    Assigns test data sequences to the pre-computed training barycenters using GPU DTW.
    """
    windows_norm, days_valid, stocks_valid = _extract_dtw_features(dataset, lookback)
    N = windows_norm.shape[0]
    device = dataset.data.device
    
    print(f"[Clustering] Assigning {N} test pairs to existing barycenters...")

    all_distances = []
    for i in range(0, N, batch_size):
        batch_X = windows_norm[i:i + batch_size]
        dists = batched_dtw_distances(batch_X, barycenters)
        all_distances.append(dists)
        
    distances_tensor = torch.cat(all_distances, dim=0)
    labels = torch.argmin(distances_tensor, dim=1)
    
    clusters = []
    for i in range(barycenters.shape[0]):
        mask = (labels == i)
        c_days = days_valid[mask].tolist()
        c_stocks = stocks_valid[mask].tolist()
        clusters.append((c_days, c_stocks))

    return clusters