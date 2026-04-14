from pathlib import Path

import numpy as np
import open3d as o3d


CONFIG = {
    # Input / output
    "ply_path": str(Path(__file__).resolve().parent / "aaaa.ply"),
    "save_point_cloud": False,
    "out_path": str(Path(__file__).resolve().parent / "aaaa_original_for_6d.ply"),
    # Visualization
    "visualize": True,
}


def load_point_cloud(ply_path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Failed to load point cloud or file is empty: {ply_path}")

    points = np.asarray(pcd.points)
    finite_mask = np.isfinite(points).all(axis=1)
    finite_indices = np.where(finite_mask)[0]
    pcd = pcd.select_by_index(finite_indices)
    if len(pcd.points) < 3:
        raise ValueError("Not enough valid points after finite filtering.")
    return pcd


def compute_pca_pose(
    pcd: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1.0
    return center, eigenvectors, eigenvalues


def compute_obb_from_pca(
    pcd: o3d.geometry.PointCloud,
    pca_center: np.ndarray,
    pca_axes: np.ndarray,
) -> o3d.geometry.OrientedBoundingBox:
    points = np.asarray(pcd.points)
    local_points = (points - pca_center) @ pca_axes

    local_min = local_points.min(axis=0)
    local_max = local_points.max(axis=0)
    extent = np.maximum(local_max - local_min, 1e-6)
    center_local = (local_min + local_max) * 0.5
    center_world = pca_center + pca_axes @ center_local

    obb = o3d.geometry.OrientedBoundingBox(center=center_world, R=pca_axes, extent=extent)
    obb.color = (1.0, 0.2, 0.2)
    return obb


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def build_main_axis_lineset(
    center: np.ndarray,
    main_vector: np.ndarray,
    axis_length: float,
) -> o3d.geometry.LineSet:
    axis_length = max(float(axis_length), 1e-3)
    p0 = center - main_vector * (axis_length * 0.5)
    p1 = center + main_vector * (axis_length * 0.5)

    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.vstack([p0, p1]))
    axis.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    axis.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.1, 0.1]], dtype=np.float64))
    return axis


def visualize(
    pcd: o3d.geometry.PointCloud,
    obb: o3d.geometry.OrientedBoundingBox,
    main_vector: np.ndarray,
) -> None:
    pcd_vis = o3d.geometry.PointCloud(pcd)
    pcd_vis.paint_uniform_color([0.75, 0.75, 0.75])

    obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    obb_lines.paint_uniform_color([1.0, 0.2, 0.2])

    axis = build_main_axis_lineset(obb.center, main_vector, axis_length=obb.extent[0])

    diag = max(float(np.linalg.norm(obb.extent)), 1e-3)
    center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=diag * 0.02)
    center_marker.paint_uniform_color([0.1, 0.9, 0.1])
    center_marker.translate(obb.center)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=diag * 0.18, origin=[0.0, 0.0, 0.0])
    coord.rotate(obb.R, center=np.zeros(3))
    coord.translate(obb.center)

    o3d.visualization.draw_geometries(
        [pcd_vis, obb_lines, axis, center_marker, coord],
        window_name="Original Point Cloud + 6D OBB",
    )


def main() -> None:
    ply_path = Path(CONFIG["ply_path"]).expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    pcd = load_point_cloud(ply_path)
    pca_center, pca_axes, eigenvalues = compute_pca_pose(pcd)
    main_vector = pca_axes[:, 0]
    obb = compute_obb_from_pca(pcd, pca_center, pca_axes)
    quat_wxyz = rotation_matrix_to_quaternion(obb.R)

    print(f"Input file            : {ply_path}")
    print(f"Point count           : {len(pcd.points)}")
    print("PCA eigenvalues       : " + np.array2string(eigenvalues, precision=6, suppress_small=True))
    print("PCA main vector       : " + np.array2string(main_vector, precision=6, suppress_small=True))
    print("OBB center (xyz)      : " + np.array2string(obb.center, precision=6, suppress_small=True))
    print("OBB extent (xyz)      : " + np.array2string(obb.extent, precision=6, suppress_small=True))
    print("OBB rotation R        : " + np.array2string(obb.R, precision=6, suppress_small=True))
    print("OBB quat (wxyz)       : " + np.array2string(quat_wxyz, precision=6, suppress_small=True))

    if bool(CONFIG["save_point_cloud"]):
        out_path = Path(CONFIG["out_path"]).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = o3d.io.write_point_cloud(str(out_path), pcd)
        if not ok:
            raise RuntimeError(f"Failed to write point cloud: {out_path}")
        print(f"Saved point cloud     : {out_path}")

    if bool(CONFIG["visualize"]):
        visualize(pcd, obb, main_vector)


if __name__ == "__main__":
    main()
