
import torch
import random
from typing import List, Tuple
from alphagen_qlib.stock_data import StockData, FeatureType
import matplotlib.pyplot as plt

def plot_cluster_waves(
    barycenters: torch.Tensor, 
    clusters: List[Tuple[List[int], List[int]]], 
    dataset: StockData, 
    lookback: int = 20, 
    num_samples: int = 50
):
    """
    Plots the medoid over a random sample of constituent waves for each cluster.
    """
    device = dataset.data.device
    close_idx = int(FeatureType.CLOSE)
    K = barycenters.shape[0]

    fig, axes = plt.subplots(K, 1, figsize=(8, 2.5 * K))
    if K == 1: 
        axes = [axes]

    for k in range(K):
        ax = axes[k]
        c_days, c_stocks = clusters[k]
        n_items = len(c_days)
        
        if n_items == 0:
            ax.set_title(f"Cluster {k} (Empty)")
            continue
            
        # Sample randomly to avoid plotting 100,000 lines
        sample_idx = random.sample(range(n_items), min(n_items, num_samples))
        sampled_days = torch.tensor([c_days[i] for i in sample_idx], dtype=torch.long, device=device)
        sampled_stocks = torch.tensor([c_stocks[i] for i in sample_idx], dtype=torch.long, device=device)

        # Advanced indexing to fetch the exact 20-day windows for the sample
        shifted_days = sampled_days + dataset.max_backtrack_days
        offsets = torch.arange(-lookback + 1, 1, device=device)
        
        window_days = shifted_days.unsqueeze(-1) + offsets
        window_stocks = sampled_stocks.unsqueeze(-1).expand_as(window_days)
        
        waves = dataset.data[window_days, close_idx, window_stocks]
        
        # Z-score normalize the samples so they match the medoid scale
        means = waves.mean(dim=1, keepdim=True)
        stds = waves.std(dim=1, keepdim=True) + 1e-8
        waves_norm = (waves - means) / stds

        # Plot faint background waves (samples)
        ax.plot(waves_norm.cpu().numpy().T, color='blue', alpha=0.1)
        
        # Plot the thick red Medoid (barycenter)
        ax.plot(barycenters[k].cpu().numpy(), color='red', linewidth=3, label='Medoid')
        
        ax.set_title(f"Cluster {k}: {n_items} total pairs ({len(sample_idx)} sampled)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/kaggle/working/cluster_waves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)



def _extract_dtw_features(               ## Check it is AI generated!!!
    data: StockData, 
    lookback: int, 
    future_window: int = 20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts rolling windows of 'lookback' days, safely truncated to ensure 
    targets can be calculated, filters NaNs, and normalizes.
    """
    device = data.data.device
    close_idx = int(FeatureType.CLOSE)
    
    # NEW FIX: Calculate exact boundaries to trim both ends
    # 1. Trim the end: Ensure we don't peek past max_future_days for the target
    cut_days = max(0, future_window - data.max_future_days)
    
    # 2. Trim the beginning: Ensure we don't slice negatively if history is too short
    skip_days = max(0, lookback - 1 - data.max_backtrack_days)
    
    effective_n_days = data.n_days - cut_days - skip_days
    
    if effective_n_days <= 0:
        raise ValueError(f"Dataset too short! n_days={data.n_days}, cut={cut_days}, skip={skip_days}")
    
    # start_idx will now safely bottom out at exactly 0
    start_idx = data.max_backtrack_days + skip_days - lookback + 1
    end_idx = data.max_backtrack_days + data.n_days - cut_days
    
    # We now only slice the guaranteed safe zone
    raw_close = data.data[start_idx:end_idx, close_idx, :]
    windows = raw_close.unfold(0, lookback, 1)
    
    # The grid matches the safe zone dimensions, but days must start at skip_days, not 0
    days_1d = torch.arange(skip_days, skip_days + effective_n_days, device=device)
    days_grid = days_1d.unsqueeze(1).expand(effective_n_days, data.n_stocks)
    stocks_grid = torch.arange(data.n_stocks, device=device).unsqueeze(0).expand(effective_n_days, data.n_stocks)

    
    windows_flat = windows.reshape(-1, lookback)
    days_flat = days_grid.reshape(-1)
    stocks_flat = stocks_grid.reshape(-1)
    
    # Filter out NaNs
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
