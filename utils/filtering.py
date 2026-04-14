import numpy as np
import open3d as o3d


def denoise_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_divisor: float,
    nb_neighbors: int,
    std_ratio: float,
    radius_multiplier: float,
    min_radius_neighbors: int,
) -> tuple[o3d.geometry.PointCloud, float, dict[str, int]]:
    points = np.asarray(pcd.points)
    bbox_size = points.max(axis=0) - points.min(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_size))
    voxel_size = max(bbox_diag / float(voxel_divisor), 1e-5)

    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    stat_filtered, _ = downsampled.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio),
    )
    radius = voxel_size * float(radius_multiplier)
    denoised, _ = stat_filtered.remove_radius_outlier(
        nb_points=int(min_radius_neighbors),
        radius=radius,
    )

    if len(denoised.points) < 3:
        raise ValueError("Too few points after denoising. Please relax filter parameters.")

    stage_counts = {
        "original": int(len(pcd.points)),
        "downsampled": int(len(downsampled.points)),
        "stat_filtered": int(len(stat_filtered.points)),
        "denoised": int(len(denoised.points)),
    }
    return denoised, voxel_size, stage_counts

