import numpy as np
import open3d as o3d


def extract_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    eps: float,
    min_points: int,
) -> tuple[o3d.geometry.PointCloud, int, int]:
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=float(eps),
            min_points=int(min_points),
            print_progress=False,
        )
    )
    valid = labels >= 0
    if not np.any(valid):
        raise ValueError(
            "DBSCAN found no valid cluster. "
            "Try increasing cluster_eps_multiplier or decreasing cluster_min_points."
        )

    cluster_labels, cluster_counts = np.unique(labels[valid], return_counts=True)
    largest_label = int(cluster_labels[np.argmax(cluster_counts)])
    largest_count = int(np.max(cluster_counts))
    indices = np.where(labels == largest_label)[0]
    largest_cluster = pcd.select_by_index(indices)
    return largest_cluster, largest_label, largest_count

