"""
RealSense D415 + SAM2 Tracking + Fast-FoundationStereo Point Cloud Combined Demo

Left window: RGB image + SAM2 mask overlay + interaction (OpenCV)
Right window: Interactive point cloud, tracked object highlighted in red (Open3D)

Usage:
  conda activate ffs
  python d415_sam2_stereo.py

Controls (focus on OpenCV window):
  - Left-click drag: Draw bounding box → initialize tracking
  - Left-click: Select foreground point → initialize tracking
  - r: Reset selection
  - p: Toggle IR projector
  - q: Quit
"""

import os, sys, time, logging
import numpy as np
import torch
import yaml
import cv2
import pyrealsense2 as rs
import open3d as o3d

# SAM2 path
SAM2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

# FFS path
FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== GPU config (SAM2 requires bfloat16) =====
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ===== Parameters =====
FFS_MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"


VALID_ITERS = 6
MAX_DISP = 192
ZFAR = 5.0
ZNEAR = 0.16          # D415 min depth ~0.16m
IMG_WIDTH = 640
IMG_HEIGHT = 480
PCD_STRIDE = 2
MASK_ALPHA = 0.5
MASK_COLOR_BGR = [75, 70, 203]            # 2D red highlight (BGR)
MASK_COLOR_RGB = np.array([203, 70, 75], dtype=np.float64) / 255.0  # Point cloud red highlight (RGB normalized)
IR_PROJECTOR_ON = True  # Enable IR projector (recommended for textureless surfaces)

# ===== 1. Load FFS model =====
logging.info("Loading FFS model...")
torch.autograd.set_grad_enabled(False)

with open(os.path.join(os.path.dirname(FFS_MODEL_DIR), "cfg.yaml"), 'r') as f:
    cfg = yaml.safe_load(f)
cfg['valid_iters'] = VALID_ITERS
cfg['max_disp'] = MAX_DISP

ffs_model = torch.load(FFS_MODEL_DIR, map_location='cpu', weights_only=False)
ffs_model.args.valid_iters = VALID_ITERS
ffs_model.args.max_disp = MAX_DISP
ffs_model.cuda().eval()
logging.info("FFS model loaded")

# ===== 2. Load SAM2 model =====
logging.info("Loading SAM2 model...")
sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
sam2_predictor.fill_hole_area = 0
logging.info("SAM2 model loaded")

# ===== 3. Initialize RealSense D415 =====
logging.info("Initializing RealSense D415...")
pipeline = rs.pipeline()
config = rs.config()

# Enable IR left/right streams + RGB stream
config.enable_stream(rs.stream.infrared, 1, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   # IR left
config.enable_stream(rs.stream.infrared, 2, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   # IR right
config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)       # RGB

profile = pipeline.start(config)

# IR projector control
device = profile.get_device()
depth_sensor = device.first_depth_sensor()
ir_projector_enabled = IR_PROJECTOR_ON
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if ir_projector_enabled else 0)
    logging.info(f"IR projector: {'ON' if ir_projector_enabled else 'OFF'}")

# ===== 4. Get camera intrinsics and extrinsics =====
frames = pipeline.wait_for_frames()
ir_left_frame = frames.get_infrared_frame(1)
color_frame = frames.get_color_frame()

# IR left camera intrinsics
ir_left_profile = ir_left_frame.get_profile().as_video_stream_profile()
ir_intrinsics = ir_left_profile.get_intrinsics()
K_ir = np.array([
    [ir_intrinsics.fx, 0, ir_intrinsics.ppx],
    [0, ir_intrinsics.fy, ir_intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

# RGB camera intrinsics
color_profile = color_frame.get_profile().as_video_stream_profile()
color_intrinsics = color_profile.get_intrinsics()
K_color = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

# IR left → RGB extrinsics
ir_to_color_extrinsics = ir_left_profile.get_extrinsics_to(color_profile)
R_ir_to_color = np.array(ir_to_color_extrinsics.rotation).reshape(3, 3).astype(np.float32)
T_ir_to_color = np.array(ir_to_color_extrinsics.translation).astype(np.float32)

# Baseline: IR left → IR right
ir_right_frame = frames.get_infrared_frame(2)
ir_right_profile = ir_right_frame.get_profile().as_video_stream_profile()
ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
baseline = abs(ir_left_to_right.translation[0])

fx_ir, fy_ir = K_ir[0, 0], K_ir[1, 1]
cx_ir, cy_ir = K_ir[0, 2], K_ir[1, 2]

logging.info(f"Baseline: {baseline*1000:.1f}mm, fx={fx_ir:.1f}")

# Pre-compute pixel grid
u_grid, v_grid = np.meshgrid(np.arange(0, IMG_WIDTH, PCD_STRIDE), np.arange(0, IMG_HEIGHT, PCD_STRIDE))
u_flat = u_grid.reshape(-1).astype(np.float32)
v_flat = v_grid.reshape(-1).astype(np.float32)

# Full resolution grid for RGB projection
u_full, v_full = np.meshgrid(np.arange(IMG_WIDTH), np.arange(IMG_HEIGHT))
u_full_flat = u_full.reshape(-1).astype(np.float32)
v_full_flat = v_full.reshape(-1).astype(np.float32)

# ===== 5. Warm up FFS =====
logging.info("Warming up FFS model...")
dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
padder = InputPadder(dummy.shape, divis_by=32, force_square=False)
d0, d1 = padder.pad(dummy, dummy)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = ffs_model.forward(d0, d1, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
del dummy, d0, d1
torch.cuda.empty_cache()
logging.info("Warm-up complete")

# ===== 6. Open3D visualizer =====
vis = o3d.visualization.Visualizer()
vis.create_window("Point Cloud (interactive)", width=720, height=540, left=700, top=50)
vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
obb_lineset = o3d.geometry.LineSet()
vis.add_geometry(obb_lineset)
vis.get_render_option().line_width = 5.0

# --- OBB smoothing state ---
prev_axes = None
obb_smooth_center = None
obb_smooth_extent = None
obb_smooth_R = None
OBB_SMOOTH = 0.75

# --- Rigid body extent stabilization ---
from collections import deque
EXTENT_WINDOW = 20
EXTENT_ALPHA_INIT = 0.4
EXTENT_ALPHA_MIN = 0.02
EXTENT_ALPHA_DECAY = 0.92
EXTENT_MAX_CHANGE_RATE = 0.05
extent_history = deque(maxlen=EXTENT_WINDOW)
extent_frame_count = 0

def create_camera_frustum(fx_, fy_, cx_, cy_, w, h, scale=0.15):
    corners_2d = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    pts = []
    for u, v in corners_2d:
        x = (u - cx_) / fx_ * scale
        y = -(v - cy_) / fy_ * scale
        pts.append([x, y, scale])
    origin = [0, 0, 0]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    points_ls = [origin] + pts
    colors_ls = [[0, 1, 0]] * len(lines)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points_ls)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors_ls)
    return ls

cam_frustum = create_camera_frustum(fx_ir, fy_ir, cx_ir, cy_ir, IMG_WIDTH, IMG_HEIGHT)
vis.add_geometry(cam_frustum)
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
vis.add_geometry(coord_frame)

# ===== 7. OpenCV window + mouse interaction =====
cv2.namedWindow("RGB + SAM2", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("RGB + SAM2", 30, 50)

drawing = False
ix, iy, fx_mouse, fy_mouse = -1, -1, -1, -1
pending_bbox = None
pending_point = None
sam2_initialized = False
need_reset = False
current_mask = None


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx_mouse, fy_mouse, pending_bbox, pending_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx_mouse, fy_mouse = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx_mouse, fy_mouse = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx_mouse, fy_mouse = x, y
        dx = abs(fx_mouse - ix)
        dy = abs(fy_mouse - iy)
        if dx > 8 and dy > 8:
            x1, y1 = min(ix, fx_mouse), min(iy, fy_mouse)
            x2, y2 = max(ix, fx_mouse), max(iy, fy_mouse)
            pending_bbox = (x1, y1, x2, y2)
        else:
            pending_point = (x, y)


cv2.setMouseCallback("RGB + SAM2", mouse_callback)

first_frame = True
frame_count = 0

logging.info("Drag/click to select target, r=reset, p=toggle IR projector, q=quit")

# ===== 8. Main loop =====
try:
    while True:
        t0 = time.time()

        # Capture frames from RealSense
        frames = pipeline.wait_for_frames()
        ir_left = np.asanyarray(frames.get_infrared_frame(1).get_data())   # (H, W) uint8
        ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())  # (H, W) uint8
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())     # (H, W, 3) uint8 BGR

        H, W = ir_left.shape[:2]

        # --- SAM2: Reset ---
        if need_reset:
            sam2_predictor.reset_state()
            sam2_initialized = False
            need_reset = False
            pending_bbox = None
            pending_point = None
            current_mask = None
            obb_lineset.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            obb_lineset.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            prev_axes = None
            obb_smooth_center = None
            obb_smooth_extent = None
            obb_smooth_R = None
            extent_history.clear()
            extent_frame_count = 0
            logging.info("Reset, select new target")

        # --- SAM2: Initialize (bbox) ---
        if pending_bbox is not None and not sam2_initialized:
            sam2_predictor.load_first_frame(color_bgr)
            x1, y1, x2, y2 = pending_bbox
            bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox_arr)
            sam2_initialized = True
            pending_bbox = None
            logging.info(f"Tracking initialized (bbox: {x1},{y1},{x2},{y2})")

        # --- SAM2: Initialize (point) ---
        elif pending_point is not None and not sam2_initialized:
            sam2_predictor.load_first_frame(color_bgr)
            px, py = pending_point
            points = np.array([[px, py]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, points=points, labels=labels)
            sam2_initialized = True
            pending_point = None
            logging.info(f"Tracking initialized (point: {px},{py})")

        # --- SAM2: Track ---
        if sam2_initialized:
            out_obj_ids, out_mask_logits = sam2_predictor.track(color_bgr)
            if len(out_obj_ids) > 0:
                current_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).byte().cpu().numpy().squeeze()
            else:
                current_mask = None

        # --- 2D display ---
        display = color_bgr.copy()
        if current_mask is not None and np.any(current_mask):
            overlay = display.copy()
            overlay[current_mask > 0] = MASK_COLOR_BGR
            display = cv2.addWeighted(display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        if drawing and ix >= 0:
            cv2.rectangle(display, (ix, iy), (fx_mouse, fy_mouse), (255, 200, 0), 2)

        # --- 3D branch: FFS with IR stereo ---
        left_rgb = np.stack([ir_left] * 3, axis=-1)
        right_rgb = np.stack([ir_right] * 3, axis=-1)

        img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_p, img1_p = padder.pad(img0, img1)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = ffs_model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

        xx = np.arange(W)[None, :].repeat(H, axis=0)
        invalid = (xx - disp) < 0
        disp[invalid] = np.inf

        depth = fx_ir * baseline / disp
        depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

        # Edge filtering
        grad_x = np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3))
        depth[(grad_x > 0.5) | (grad_y > 0.5)] = 0

        depth_ds = depth[::PCD_STRIDE, ::PCD_STRIDE]
        z_flat = depth_ds.reshape(-1)
        valid_mask = z_flat > 0

        z = z_flat[valid_mask]
        u = u_flat[valid_mask]
        v = v_flat[valid_mask]

        # IR pixel → 3D point (IR left coordinate system)
        x3d = (u - cx_ir) * z / fx_ir
        y3d = -(v - cy_ir) * z / fy_ir  # Flip Y for visualization
        print(f"x3d:{x3d}, y3d:{y3d}, z:{z}")
        points_3d = np.stack([x3d, y3d, z], axis=-1)
        
        # # Create an Open3D PointCloud object
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points_3d)
        # pcd.colors = o3d.utility.Vector3dVector(np.ones((len(z), 3), dtype=np.float64))
        # o3d.visualization.draw_geometries([pcd])


        # RGB coloring: IR → RGB projection
        pts_ir = np.stack([(u - cx_ir) * z / fx_ir, (v - cy_ir) * z / fy_ir, z], axis=-1)
        pts_color = (R_ir_to_color @ pts_ir.T).T + T_ir_to_color
        u_rgb = (K_color[0, 0] * pts_color[:, 0] / pts_color[:, 2] + K_color[0, 2]).astype(np.int32)
        v_rgb = (K_color[1, 1] * pts_color[:, 1] / pts_color[:, 2] + K_color[1, 2]).astype(np.int32)
        in_bounds = (u_rgb >= 0) & (u_rgb < W) & (v_rgb >= 0) & (v_rgb < H)

        colors = np.zeros((len(z), 3), dtype=np.float64)
        colors[in_bounds] = color_bgr[v_rgb[in_bounds], u_rgb[in_bounds], ::-1].astype(np.float64) / 255.0

        # --- Point cloud highlight: Map SAM2 mask (RGB space) to IR space ---
        if current_mask is not None and np.any(current_mask):
            highlight = np.zeros(len(z), dtype=bool)
            highlight[in_bounds] = current_mask[v_rgb[in_bounds], u_rgb[in_bounds]] > 0

            if np.any(highlight):
                colors[highlight] = colors[highlight] * 0.2 + MASK_COLOR_RGB * 0.8

                # --- 6D BBox: PCA + axis consistency + extent stabilization ---
                obj_pts = points_3d[highlight]
                if len(obj_pts) >= 10:
                    centroid = obj_pts.mean(axis=0)
                    dists = np.linalg.norm(obj_pts - centroid, axis=1)
                    thresh = np.percentile(dists, 90)
                    filtered = obj_pts[dists <= thresh]

                    if len(filtered) >= 10:
                        center = filtered.mean(axis=0)
                        cov = np.cov((filtered - center).T)
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        idx = np.argsort(eigenvalues)[::-1]
                        eigenvalues = eigenvalues[idx]
                        axes = eigenvectors[:, idx]

                        if np.linalg.det(axes) < 0:
                            axes[:, 2] = -axes[:, 2]

                        if prev_axes is not None:
                            for i in range(3):
                                if np.dot(axes[:, i], prev_axes[:, i]) < 0:
                                    axes[:, i] = -axes[:, i]
                        prev_axes = axes.copy()

                        local = (filtered - center) @ axes
                        raw_extent = local.max(axis=0) - local.min(axis=0)
                        local_center_offset = (local.max(axis=0) + local.min(axis=0)) / 2
                        center = center + axes @ local_center_offset

                        extent_frame_count += 1

                        if obb_smooth_center is not None:
                            obb_smooth_center = OBB_SMOOTH * center + (1 - OBB_SMOOTH) * obb_smooth_center
                            obb_smooth_R = OBB_SMOOTH * axes + (1 - OBB_SMOOTH) * obb_smooth_R
                            u0 = obb_smooth_R[:, 0]
                            u0 = u0 / np.linalg.norm(u0)
                            u1 = obb_smooth_R[:, 1] - np.dot(obb_smooth_R[:, 1], u0) * u0
                            u1 = u1 / np.linalg.norm(u1)
                            u2 = np.cross(u0, u1)
                            obb_smooth_R = np.column_stack([u0, u1, u2])

                            extent_history.append(raw_extent.copy())
                            ext_alpha = max(EXTENT_ALPHA_MIN,
                                            EXTENT_ALPHA_INIT * (EXTENT_ALPHA_DECAY ** extent_frame_count))
                            if len(extent_history) >= 3:
                                median_ext = np.median(np.array(extent_history), axis=0)
                                candidate_extent = 0.5 * raw_extent + 0.5 * median_ext
                            else:
                                candidate_extent = raw_extent
                            max_delta = obb_smooth_extent * EXTENT_MAX_CHANGE_RATE
                            delta = candidate_extent - obb_smooth_extent
                            clamped_extent = obb_smooth_extent + np.clip(delta, -max_delta, max_delta)
                            obb_smooth_extent = ext_alpha * clamped_extent + (1 - ext_alpha) * obb_smooth_extent
                        else:
                            obb_smooth_center = center.copy()
                            obb_smooth_extent = raw_extent.copy()
                            obb_smooth_R = axes.copy()
                            extent_history.append(raw_extent.copy())

                        half = obb_smooth_extent / 2
                        corners_local = np.array([
                            [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                            [-1,-1, 1], [1,-1, 1], [1,1, 1], [-1,1, 1]
                        ], dtype=np.float64) * half
                        corners_world = corners_local @ obb_smooth_R.T + obb_smooth_center

                        obb_edges = [[0,1],[1,2],[2,3],[3,0],
                                     [4,5],[5,6],[6,7],[7,4],
                                     [0,4],[1,5],[2,6],[3,7]]
                        obb_lineset.points = o3d.utility.Vector3dVector(corners_world)
                        obb_lineset.lines = o3d.utility.Vector2iVector(obb_edges)
                        obb_lineset.colors = o3d.utility.Vector3dVector(
                            [[0, 1, 0]] * len(obb_edges))
                    else:
                        obb_lineset.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                        obb_lineset.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
                else:
                    obb_lineset.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                    obb_lineset.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        else:
            obb_lineset.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            obb_lineset.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))

        t1 = time.time()
        fps = 1.0 / (t1 - t0)

        # FPS + status bar
        cv2.putText(display, f"FPS: {fps:.1f}", (IMG_WIDTH - 130, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        ir_status = "IR:ON" if ir_projector_enabled else "IR:OFF"
        if sam2_initialized:
            status = f"TRACKING | {ir_status} | r=reset p=IR q=quit"
            if obb_smooth_extent is not None:
                ext = obb_smooth_extent
                ext_alpha = max(EXTENT_ALPHA_MIN,
                                EXTENT_ALPHA_INIT * (EXTENT_ALPHA_DECAY ** extent_frame_count))
                ext_str = f"BBox: {ext[0]*100:.1f}x{ext[1]*100:.1f}x{ext[2]*100:.1f}cm a={ext_alpha:.3f}"
                cv2.putText(display, ext_str, (10, IMG_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            status = f"Draw bbox / Click to select | {ir_status} | q=quit"
        cv2.putText(display, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("RGB + SAM2", display)

        # Update Open3D
        pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if first_frame:
            vis.reset_view_point(True)
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            first_frame = False

        vis.update_geometry(pcd)
        vis.update_geometry(obb_lineset)
        vis.poll_events()
        vis.update_renderer()

        frame_count += 1
        if frame_count % 30 == 0:
            logging.info(f"Frame {frame_count}, FPS: {fps:.1f}, points: {len(points_3d)}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            need_reset = True
        elif key == ord('p'):
            # Toggle IR projector
            ir_projector_enabled = not ir_projector_enabled
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1 if ir_projector_enabled else 0)
            logging.info(f"IR projector: {'ON' if ir_projector_enabled else 'OFF'}")

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    vis.destroy_window()
    cv2.destroyAllWindows()
    logging.info("Exited")
