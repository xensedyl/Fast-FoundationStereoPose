import numpy as np
import open3d as o3d


def compute_pca_pose(
    pcd: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(pcd.points)
    if points.shape[0] < 3:
        raise ValueError("Not enough points for PCA.")

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

