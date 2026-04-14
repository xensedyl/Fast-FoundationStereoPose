from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import (  # noqa: E402
    compute_obb_from_pca,
    compute_pca_pose,
    compute_point_curvatures,
    denoise_point_cloud,
    extract_largest_cluster,
    load_point_cloud,
    print_curvature_distribution,
    rotation_matrix_to_quaternion,
    save_point_cloud,
    segment_by_curvature,
    visualize,
)


CONFIG = {
    # Input / output
    "ply_path": str(Path(__file__).resolve().parent / "aaaa.ply"),
    "save_denoised": True,
    "out_path": str(Path(__file__).resolve().parent / "aaaa_denoised_largest_cluster.ply"),
    # Denoise
    "voxel_divisor": 50.0,
    "nb_neighbors": 10,
    "std_ratio": 1.0,
    "radius_multiplier": 1.2,  # radius = voxel_size * radius_multiplier
    "min_radius_neighbors": 4,
    # DBSCAN clustering (after denoise)
    "cluster_eps_multiplier": 2.2,  # eps = voxel_size * cluster_eps_multiplier
    "cluster_min_points": 10,
    # Curvature (on final processed cloud = largest cluster)
    "curvature_knn": 30,
    "curvature_hist_bins": 10,
    "curvature_threshold": 0.02,
    "curvature_select": "low",  # "high" or "low"
    "save_curvature_segment": True,
    "curvature_segment_out_path": str(Path(__file__).resolve().parent / "aaaa_curvature_segment.ply"),
    # Visualization
    "visualize": True,
}


def main() -> None:
    pcd = load_point_cloud(CONFIG["ply_path"])

    denoised, voxel_size, stage_counts = denoise_point_cloud(
        pcd=pcd,
        voxel_divisor=float(CONFIG["voxel_divisor"]),
        nb_neighbors=int(CONFIG["nb_neighbors"]),
        std_ratio=float(CONFIG["std_ratio"]),
        radius_multiplier=float(CONFIG["radius_multiplier"]),
        min_radius_neighbors=int(CONFIG["min_radius_neighbors"]),
    )

    cluster_eps = voxel_size * float(CONFIG["cluster_eps_multiplier"])
    largest_cluster, largest_label, largest_count = extract_largest_cluster(
        pcd=denoised,
        eps=cluster_eps,
        min_points=int(CONFIG["cluster_min_points"]),
    )

    pca_center, pca_axes, eigenvalues = compute_pca_pose(largest_cluster)
    main_vector = pca_axes[:, 0]
    obb = compute_obb_from_pca(
        pcd=largest_cluster,
        pca_center=pca_center,
        pca_axes=pca_axes,
    )
    quat_wxyz = rotation_matrix_to_quaternion(obb.R)

    print(f"Input file            : {Path(CONFIG['ply_path']).expanduser().resolve()}")
    print(f"Original points       : {stage_counts['original']}")
    print(f"After voxel           : {stage_counts['downsampled']}")
    print(f"After stat filter     : {stage_counts['stat_filtered']}")
    print(f"After radius filter   : {stage_counts['denoised']}")
    print(f"DBSCAN eps            : {cluster_eps:.6f}")
    print(f"Largest cluster label : {largest_label}")
    print(f"Largest cluster points: {largest_count}")
    reduction = (1.0 - largest_count / max(stage_counts["original"], 1)) * 100.0
    print(f"Total reduction ratio : {reduction:.2f}%")
    print("PCA eigenvalues       : " + np.array2string(eigenvalues, precision=6, suppress_small=True))
    print("PCA main vector       : " + np.array2string(main_vector, precision=6, suppress_small=True))
    print("OBB center (xyz)      : " + np.array2string(obb.center, precision=6, suppress_small=True))
    print("OBB extent (xyz)      : " + np.array2string(obb.extent, precision=6, suppress_small=True))
    print("OBB rotation R        : " + np.array2string(obb.R, precision=6, suppress_small=True))
    print("OBB quat (wxyz)       : " + np.array2string(quat_wxyz, precision=6, suppress_small=True))

    curvatures = compute_point_curvatures(
        largest_cluster,
        knn=int(CONFIG["curvature_knn"]),
    )
    print_curvature_distribution(
        curvatures,
        hist_bins=int(CONFIG["curvature_hist_bins"]),
    )

    curvature_threshold = float(CONFIG["curvature_threshold"])
    curvature_select = str(CONFIG["curvature_select"])
    curvature_segment, curvature_mask = segment_by_curvature(
        pcd=largest_cluster,
        curvatures=curvatures,
        threshold=curvature_threshold,
        select_mode=curvature_select,
    )
    seg_count = len(curvature_segment.points)
    seg_ratio = seg_count / max(len(largest_cluster.points), 1) * 100.0
    print(f"Curvature threshold   : {curvature_threshold:.6f}")
    print(f"Curvature select mode : {curvature_select}")
    print(f"Curvature seg points  : {seg_count} ({seg_ratio:.2f}%)")




    if bool(CONFIG["save_denoised"]):
        out_path = save_point_cloud(
            largest_cluster,
            CONFIG["out_path"],
            label="largest cluster point cloud",
        )
        print(f"Saved largest cluster : {out_path}")

    if bool(CONFIG["save_curvature_segment"]):
        seg_path = save_point_cloud(
            curvature_segment,
            CONFIG["curvature_segment_out_path"],
            label="curvature segment point cloud",
        )
        print(f"Saved curvature seg   : {seg_path}")

    if bool(CONFIG["visualize"]):
        visualize(
            pcd=largest_cluster,
            obb=obb,
            main_vector=main_vector,
            curvature_mask=curvature_mask,
            title="Largest Cluster + Curvature Segmentation + 6D OBB",
        )


if __name__ == "__main__":
    main()
