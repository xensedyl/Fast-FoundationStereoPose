import numpy as np
import open3d as o3d


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
    curvature_mask: np.ndarray | None = None,
    title: str = "Largest Cluster + Curvature Segmentation + 6D OBB",
) -> None:
    pcd_vis = o3d.geometry.PointCloud(pcd)

    if curvature_mask is None:
        pcd_vis.paint_uniform_color([0.75, 0.75, 0.75])
    else:
        n = len(pcd.points)
        if curvature_mask.shape[0] != n:
            raise ValueError("Curvature mask size mismatch.")
        colors = np.tile(np.array([0.70, 0.70, 0.70], dtype=np.float64), (n, 1))
        colors[curvature_mask] = np.array([1.0, 0.2, 0.2], dtype=np.float64)
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)

    axis = build_main_axis_lineset(obb.center, main_vector, axis_length=obb.extent[0])
    obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    obb_lines.paint_uniform_color([1.0, 0.2, 0.2])

    diag = max(float(np.linalg.norm(obb.extent)), 1e-3)
    center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=diag * 0.02)
    center_marker.paint_uniform_color([0.1, 0.9, 0.1])
    center_marker.translate(obb.center)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=diag * 0.18, origin=[0.0, 0.0, 0.0])
    coord.rotate(obb.R, center=np.zeros(3))
    coord.translate(obb.center)

    o3d.visualization.draw_geometries([pcd_vis, obb_lines, axis, center_marker, coord], window_name=title)

