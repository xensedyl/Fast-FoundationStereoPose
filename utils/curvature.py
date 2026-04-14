import numpy as np
import open3d as o3d


def compute_point_curvatures(
    pcd: o3d.geometry.PointCloud,
    knn: int,
) -> np.ndarray:
    points = np.asarray(pcd.points)
    n = points.shape[0]
    if n < 3:
        raise ValueError("Not enough points for curvature computation.")

    k = max(3, min(int(knn), n))
    tree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = np.zeros(n, dtype=np.float64)

    for i in range(n):
        nn_count, idx, _ = tree.search_knn_vector_3d(points[i], k)
        if nn_count < 3:
            curvatures[i] = 0.0
            continue

        neighborhood = points[np.asarray(idx, dtype=np.int64)]
        centered = neighborhood - neighborhood.mean(axis=0)
        cov = (centered.T @ centered) / max(nn_count - 1, 1)
        evals = np.linalg.eigvalsh(cov)
        evals = np.clip(evals, 0.0, None)
        denom = float(evals.sum())
        curvatures[i] = float(evals[0] / denom) if denom > 1e-12 else 0.0

    return curvatures


def print_curvature_distribution(curvatures: np.ndarray, hist_bins: int) -> None:
    pcts = np.percentile(curvatures, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print("Curvature stats       :")
    print(
        "  min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
            float(curvatures.min()),
            float(curvatures.max()),
            float(curvatures.mean()),
            float(curvatures.std()),
        )
    )
    print(
        "  p01={:.6f}, p05={:.6f}, p10={:.6f}, p25={:.6f}, p50={:.6f}, p75={:.6f}, p90={:.6f}, p95={:.6f}, p99={:.6f}".format(
            *[float(v) for v in pcts]
        )
    )

    bins = max(3, int(hist_bins))
    hist, edges = np.histogram(curvatures, bins=bins)
    print(f"Curvature histogram   : {bins} bins")
    for i in range(bins):
        left = float(edges[i])
        right = float(edges[i + 1])
        count = int(hist[i])
        ratio = count / max(len(curvatures), 1) * 100.0
        print(f"  [{left:.6f}, {right:.6f}) -> {count:5d} ({ratio:6.2f}%)")


def segment_by_curvature(
    pcd: o3d.geometry.PointCloud,
    curvatures: np.ndarray,
    threshold: float,
    select_mode: str,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    mode = str(select_mode).strip().lower()
    if mode not in {"high", "low"}:
        raise ValueError("curvature_select must be 'high' or 'low'.")

    if mode == "high":
        mask = curvatures >= float(threshold)
    else:
        mask = curvatures <= float(threshold)

    indices = np.where(mask)[0]
    segmented = pcd.select_by_index(indices)
    return segmented, mask

