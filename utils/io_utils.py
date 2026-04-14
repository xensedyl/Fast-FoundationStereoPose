from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(ply_path: Path | str) -> o3d.geometry.PointCloud:
    path = Path(ply_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Failed to load point cloud or file is empty: {path}")

    points = np.asarray(pcd.points)
    finite_mask = np.isfinite(points).all(axis=1)
    finite_indices = np.where(finite_mask)[0]
    pcd = pcd.select_by_index(finite_indices)
    if len(pcd.points) < 3:
        raise ValueError("Not enough valid points after finite filtering.")
    return pcd


def save_point_cloud(pcd: o3d.geometry.PointCloud, out_path: Path | str, label: str = "point cloud") -> Path:
    path = Path(out_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd)
    if not ok:
        raise RuntimeError(f"Failed to write {label}: {path}")
    return path

