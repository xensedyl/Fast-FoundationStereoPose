"""
RealSense D415 + SAM2 Tracking + Fast-FoundationStereo Point Cloud Combined Demo
(ROS 2 Publisher Integrated)+ publish(/target_pose_cam)
"""

import os, sys, time, logging
import numpy as np
import torch
import yaml
import cv2
import pyrealsense2 as rs
import open3d as o3d

#  新增：ROS 2 和 矩阵转换的依赖
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as SciPyRot

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

# ===== GPU config =====
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
ZNEAR = 0.16          
IMG_WIDTH = 640
IMG_HEIGHT = 480
PCD_STRIDE = 2
MASK_ALPHA = 0.5
MASK_COLOR_BGR = [75, 70, 203]            
MASK_COLOR_RGB = np.array([203, 70, 75], dtype=np.float64) / 255.0  
IR_PROJECTOR_ON = True  

#  新增：初始化 ROS 2 节点
rclpy.init()
ros_node = rclpy.create_node('vision_pose_publisher')
pose_pub = ros_node.create_publisher(PoseStamped, '/target_pose_cam', 10)
logging.info("ROS 2 Publisher Initialized: /target_pose_cam")

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

# ===== 2.5 Load YOLO-OBB model =====
from ultralytics import YOLO
logging.info("Loading YOLO-OBB model...")
yolo_model = YOLO("/home/zdh/ultralytics/runs/obb/train5/weights/best.pt")
logging.info("YOLO model loaded")

logging.info("Loading SAM2 model...")
sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
sam2_predictor.fill_hole_area = 0

# ===== 3. Initialize RealSense D415 =====
logging.info("Initializing RealSense D415...")
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.infrared, 1, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   
config.enable_stream(rs.stream.infrared, 2, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   
config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)       

profile = pipeline.start(config)

device = profile.get_device()
depth_sensor = device.first_depth_sensor()
ir_projector_enabled = IR_PROJECTOR_ON
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if ir_projector_enabled else 0)

# ===== 4. Get camera intrinsics and extrinsics =====
frames = pipeline.wait_for_frames()
ir_left_frame = frames.get_infrared_frame(1)
color_frame = frames.get_color_frame()

ir_left_profile = ir_left_frame.get_profile().as_video_stream_profile()
ir_intrinsics = ir_left_profile.get_intrinsics()
K_ir = np.array([
    [ir_intrinsics.fx, 0, ir_intrinsics.ppx],
    [0, ir_intrinsics.fy, ir_intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

color_profile = color_frame.get_profile().as_video_stream_profile()
color_intrinsics = color_profile.get_intrinsics()
K_color = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

ir_to_color_extrinsics = ir_left_profile.get_extrinsics_to(color_profile)
R_ir_to_color = np.array(ir_to_color_extrinsics.rotation).reshape(3, 3).astype(np.float32)
T_ir_to_color = np.array(ir_to_color_extrinsics.translation).astype(np.float32)

ir_right_frame = frames.get_infrared_frame(2)
ir_right_profile = ir_right_frame.get_profile().as_video_stream_profile()
ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
baseline = abs(ir_left_to_right.translation[0])

fx_ir, fy_ir = K_ir[0, 0], K_ir[1, 1]
cx_ir, cy_ir = K_ir[0, 2], K_ir[1, 2]

u_grid, v_grid = np.meshgrid(np.arange(0, IMG_WIDTH, PCD_STRIDE), np.arange(0, IMG_HEIGHT, PCD_STRIDE))
u_flat = u_grid.reshape(-1).astype(np.float32)
v_flat = v_grid.reshape(-1).astype(np.float32)

u_full, v_full = np.meshgrid(np.arange(IMG_WIDTH), np.arange(IMG_HEIGHT))
u_full_flat = u_full.reshape(-1).astype(np.float32)
v_full_flat = v_full.reshape(-1).astype(np.float32)

# ===== 5. Warm up FFS =====
dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
padder = InputPadder(dummy.shape, divis_by=32, force_square=False)
d0, d1 = padder.pad(dummy, dummy)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = ffs_model.forward(d0, d1, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
del dummy, d0, d1
torch.cuda.empty_cache()

# ===== 6. Open3D visualizer & Helpers =====
vis = o3d.visualization.Visualizer()
vis.create_window("Point Cloud (interactive)", width=720, height=540, left=700, top=50)
vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
obb_lineset = o3d.geometry.LineSet()
vis.add_geometry(obb_lineset)
vis.get_render_option().line_width = 5.0

obb_smooth_center = None
obb_smooth_extent = None
obb_smooth_R = None
OBB_SMOOTH = 0.65  # 降低平滑系数，增加响应速度

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

pca_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
vis.add_geometry(pca_frame)

# 新增核心函数：基于物理高低点的坐标轴绝对定向校验 ###################
# def validate_and_correct_axes(obj_pts, center, pca_axes):
#     axes = pca_axes.copy()
    
#     # 1. 校验 Z 轴：
#     # 在 RealSense 光学坐标系中，Z轴正方向指向物体内部。
#     # 所以"朝上"(指向相机) 意味着 Z 分量应该为负数。
#     # if axes[2, 2] > 0: #z轴朝上
#     if axes[2, 2] < 0: #如果z轴朝上，那就翻转朝下
#         axes[:, 2] = -axes[:, 2]

#     # 2. 校验 X 轴：正方向必须朝向低点 (深度值 Z 越大的地方为低点)
#     x_axis = axes[:, 0]
#     # 将过滤后的点云投影到当前的 X 轴上
#     projections = np.dot(obj_pts - center, x_axis)
    
#     # 排序找到 X 轴正反两端的点
#     sort_idx = np.argsort(projections)
#     # 取两端各 10% 的点求平均，过滤掉个别飞点导致的误差
#     margin = max(1, int(len(sort_idx) * 0.1)) 
    
#     neg_end_pts = obj_pts[sort_idx[:margin]]  # X轴负向最远端的点群
#     pos_end_pts = obj_pts[sort_idx[-margin:]] # X轴正向最远端的点群
    
#     neg_end_z = np.mean(neg_end_pts[:, 2]) # 负端点的平均深度 (Z值)
#     pos_end_z = np.mean(pos_end_pts[:, 2]) # 正端点的平均深度 (Z值)
    
#     # 分析判断：如果正端点的深度 < 负端点的深度，说明正方向指向上坡(高点)了
#     if pos_end_z < neg_end_z:  #x轴往点云低处走
#         # 强制翻转 X 轴，使其朝向低点
#     # if pos_end_z > neg_end_z:  #x往点云高点走  
#         axes[:, 0] = -axes[:, 0]
        
#     # 3. 强制重构 Y 轴以保证标准的右手直角坐标系
#     axes[:, 1] = np.cross(axes[:, 2], axes[:, 0])
    
#     # 正交归一化，消除浮点数计算带来的微小误差
#     axes[:, 0] /= np.linalg.norm(axes[:, 0])
#     axes[:, 1] /= np.linalg.norm(axes[:, 1])
#     axes[:, 2] /= np.linalg.norm(axes[:, 2])
    
#     return axes
#################################################################################################
#  新增核心函数：基于【高低点+宽窄双重融合机制】的坐标轴绝对定向校验
# def validate_and_correct_axes(obj_pts, center, pca_axes):
#     axes = pca_axes.copy()
    
#     # 1. 校验 Z 轴：永远朝下 (指向桌面 / 远离相机)
#     # 在 RealSense 光学坐标系中，Z分量大于0是远离相机(朝下)
#     if axes[2, 2] < 0:
#         axes[:, 2] = -axes[:, 2]

#     # 2. 校验 X 轴：双重融合决策 (高点优先，宽处辅助)
#     x_axis = axes[:, 0]
#     y_axis = axes[:, 1]
    
#     # 将所有点云投影到 X 轴上
#     projections = np.dot(obj_pts - center, x_axis)
    
#     # === 提取数据 ===
#     # 用于计算高低的极端点 (两端各取 10%)
#     sort_idx = np.argsort(projections)
#     margin = max(1, int(len(sort_idx) * 0.1)) 
#     neg_end_pts = obj_pts[sort_idx[:margin]]  # X轴负向最远端
#     pos_end_pts = obj_pts[sort_idx[-margin:]] # X轴正向最远端
    
#     # 用于计算宽窄的半区 (正半轴和负半轴所有点)
#     pos_half = obj_pts[projections > 0]
#     neg_half = obj_pts[projections < 0]

#     # === 机制 A：高低点得分 ===
#     # 我们期望 X 轴指向高点 (即深度值 Z 较小的一端)
#     neg_end_z = np.mean(neg_end_pts[:, 2])
#     pos_end_z = np.mean(pos_end_pts[:, 2])
    
#     # 计算深度差 (正向Z - 负向Z)。如果 > 0，说明当前正方向更深(也就是低点)，倾向于翻转
#     diff_z = pos_end_z - neg_end_z
#     # 将深度差归一化到约 [-1, 1] 的分数。+0.002 代表 2mm 的软性噪声容忍度
#     score_height = diff_z / (abs(diff_z) + 0.002)

#     # === 机制 B：宽窄得分 ===
#     # 我们期望 X 轴指向宽处 (即 Y 轴方差较大的一端)
#     score_width = 0.0
#     if len(pos_half) > 10 and len(neg_half) > 10:
#         pos_y_var = np.var(np.dot(pos_half - center, y_axis))
#         neg_y_var = np.var(np.dot(neg_half - center, y_axis))
        
#         # 计算宽度差 (负向宽 - 正向宽)。如果 > 0，说明负向更宽，倾向于翻转
#         diff_w = neg_y_var - pos_y_var
#         max_var = max(pos_y_var, neg_y_var)
#         if max_var > 1e-6: # 防止除零
#             # 将宽度差归一化到 [-1, 1] 的分数
#             score_width = diff_w / max_var

#     # === 综合打分决策 ===
#     # 根据您的需求：高低判断的权重稍微大于宽度判断
#     WEIGHT_HEIGHT = 0.8
#     WEIGHT_WIDTH = 0.2
    
#     # 计算综合“翻转得分”
#     total_flip_score = (WEIGHT_HEIGHT * score_height) + (WEIGHT_WIDTH * score_width)
    
#     # 如果综合得分 > 0，说明整体评估结果指向“应该翻转”
#     if total_flip_score > 0:
#         axes[:, 0] = -axes[:, 0]
        
#     # 3. 强制重构 Y 轴以保证标准的右手直角坐标系 (Y = Z cross X)
#     axes[:, 1] = np.cross(axes[:, 2], axes[:, 0])
    
#     # 正交归一化，消除浮点计算微小误差
#     axes[:, 0] /= np.linalg.norm(axes[:, 0])
#     axes[:, 1] /= np.linalg.norm(axes[:, 1])
#     axes[:, 2] /= np.linalg.norm(axes[:, 2])
    
#     return axes
# 
# ========== 新增：曲率计算 ==========
def compute_curvature(points, radius=0.02, min_neighbors=6):
    """
    基于局部邻域 PCA 计算每个点的曲率。
    曲率 = λ_min / (λ_0 + λ_1 + λ_2)，值越大表示越"尖锐"/"边缘"。
    """
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_tmp)
    
    N = len(points)
    curvatures = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        [k, idx, _] = kdtree.search_radius_vector_3d(points[i], radius)
        if k < min_neighbors:
            continue
        neighbors = points[idx, :]
        cov = np.cov((neighbors - neighbors.mean(axis=0)).T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 0, None)
        total = eigvals.sum()
        if total > 1e-12:
            curvatures[i] = eigvals[0] / total
    
    return curvatures


# ========== 坐标轴校验：高低点 + 曲率 双重融合 ==========
def validate_and_correct_axes(obj_pts, center, pca_axes):
    axes = pca_axes.copy()
    
    # 1. 校验 Z 轴：朝下（远离相机，Z 分量为正）
    if axes[2, 2] < 0:
        axes[:, 2] = -axes[:, 2]

    # 2. 校验 X 轴：双重融合机制
    x_axis = axes[:, 0]
    projections = np.dot(obj_pts - center, x_axis)
    
    # ---------- 机制 A：高低点 ----------
    sort_idx = np.argsort(projections)
    margin = max(1, int(len(sort_idx) * 0.1))
    neg_end_pts = obj_pts[sort_idx[:margin]]
    pos_end_pts = obj_pts[sort_idx[-margin:]]
    neg_end_z = np.mean(neg_end_pts[:, 2])
    pos_end_z = np.mean(pos_end_pts[:, 2])
    diff_z = neg_end_z - pos_end_z  # > 0 表示当前正方向指向高点 → 需翻转
    score_height = diff_z / (abs(diff_z) + 0.002)

    # ---------- 机制 B：曲率 ----------
    score_curvature = 0.0
    try:
        curvatures = compute_curvature(obj_pts, radius=0.02, min_neighbors=6)
        valid_curv = curvatures[curvatures > 0]
        if len(valid_curv) >= 10:
            threshold = np.percentile(valid_curv, 75)
            high_curv_mask = curvatures >= threshold
            high_curv_pts = obj_pts[high_curv_mask]
            
            if len(high_curv_pts) >= 5:
                high_curv_centroid = high_curv_pts.mean(axis=0)
                proj_hc = np.dot(high_curv_centroid - center, x_axis)
                half_span = max(abs(projections.max()), abs(projections.min()), 1e-6)
                score_curvature = np.clip(proj_hc / half_span, -1.0, 1.0)
    except Exception:
        score_curvature = 0.0

    # ---------- 融合决策 ----------
    WEIGHT_HEIGHT = 0.5
    WEIGHT_CURVATURE = 0.5
    total_flip_score = WEIGHT_HEIGHT * score_height + WEIGHT_CURVATURE * score_curvature
    
    if total_flip_score > 0:
        axes[:, 0] = -axes[:, 0]
        
    # 3. 重构 Y 轴
    axes[:, 1] = np.cross(axes[:, 2], axes[:, 0])
    axes[:, 0] /= np.linalg.norm(axes[:, 0])
    axes[:, 1] /= np.linalg.norm(axes[:, 1])
    axes[:, 2] /= np.linalg.norm(axes[:, 2])
    
    return axes
# #############################################################################################

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
        if drawing: fx_mouse, fy_mouse = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx_mouse, fy_mouse = x, y
        if abs(fx_mouse - ix) > 8 and abs(fy_mouse - iy) > 8:
            pending_bbox = (min(ix, fx_mouse), min(iy, fy_mouse), max(ix, fx_mouse), max(iy, fy_mouse))
        else:
            pending_point = (x, y)

cv2.setMouseCallback("RGB + SAM2", mouse_callback)
first_frame = True
frame_count = 0

try:
    while True:
        t0 = time.time()
        
        #  新增：处理 ROS 2 回调和后台任务
        rclpy.spin_once(ros_node, timeout_sec=0)

        frames = pipeline.wait_for_frames()
        ir_left = np.asanyarray(frames.get_infrared_frame(1).get_data())   
        ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())  
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())     

        H, W = ir_left.shape[:2]

        if need_reset:
            sam2_predictor.reset_state()
            sam2_initialized = False
            need_reset = False
            pending_bbox = pending_point = current_mask = None
            obb_lineset.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            obb_lineset.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            obb_smooth_center = obb_smooth_extent = obb_smooth_R = None
            extent_history.clear()
            extent_frame_count = 0

        if not sam2_initialized and pending_bbox is None and pending_point is None:
            results = yolo_model(color_bgr, conf=0.1, verbose=False)
            if len(results) > 0 and results[0].obb is not None:
                obbs = results[0].obb
                if len(obbs) > 0:
                    best_idx = np.argmax(obbs.conf.cpu().numpy())
                    corners = obbs.xyxyxyxy[best_idx].cpu().numpy()
                    x1, y1 = np.min(corners, axis=0)
                    x2, y2 = np.max(corners, axis=0)
                    pending_bbox = (int(x1), int(y1), int(x2), int(y2))

        if not sam2_initialized and 'results' in locals() and len(results) > 0: 
            display = results[0].plot() 

        if pending_bbox is not None and not sam2_initialized:
            sam2_predictor.load_first_frame(color_bgr)
            bbox_arr = np.array([[pending_bbox[0], pending_bbox[1]], [pending_bbox[2], pending_bbox[3]]], dtype=np.float32)
            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox_arr)
            sam2_initialized = True
            pending_bbox = None

        elif pending_point is not None and not sam2_initialized:
            sam2_predictor.load_first_frame(color_bgr)
            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, points=np.array([[pending_point[0], pending_point[1]]], dtype=np.float32), labels=np.array([1], dtype=np.int32))
            sam2_initialized = True
            pending_point = None

        if sam2_initialized:
            out_obj_ids, out_mask_logits = sam2_predictor.track(color_bgr)
            if len(out_obj_ids) > 0:
                current_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).byte().cpu().numpy().squeeze()
            else:
                current_mask = None

        display = color_bgr.copy()
        if current_mask is not None and np.any(current_mask):
            overlay = display.copy()
            overlay[current_mask > 0] = MASK_COLOR_BGR
            display = cv2.addWeighted(display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        if drawing and ix >= 0:
            cv2.rectangle(display, (ix, iy), (fx_mouse, fy_mouse), (255, 200, 0), 2)

        left_rgb = np.stack([ir_left] * 3, axis=-1)
        right_rgb = np.stack([ir_right] * 3, axis=-1)

        img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_p, img1_p = padder.pad(img0, img1)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = ffs_model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W).clip(0, None)

        xx = np.arange(W)[None, :].repeat(H, axis=0)
        disp[(xx - disp) < 0] = np.inf

        depth = fx_ir * baseline / disp
        depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

        grad_x = np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3))
        depth[(grad_x > 0.5) | (grad_y > 0.5)] = 0

        depth_ds = depth[::PCD_STRIDE, ::PCD_STRIDE]
        z_flat = depth_ds.reshape(-1)
        valid_mask = z_flat > 0

        z = z_flat[valid_mask]
        u = u_flat[valid_mask]
        v = v_flat[valid_mask]

        x3d = (u - cx_ir) * z / fx_ir
        y3d = (v - cy_ir) * z / fy_ir 
        points_3d = np.stack([x3d, y3d, z], axis=-1)

        pts_ir = np.stack([(u - cx_ir) * z / fx_ir, (v - cy_ir) * z / fy_ir, z], axis=-1)
        pts_color = (R_ir_to_color @ pts_ir.T).T + T_ir_to_color
        u_rgb = (K_color[0, 0] * pts_color[:, 0] / pts_color[:, 2] + K_color[0, 2]).astype(np.int32)
        v_rgb = (K_color[1, 1] * pts_color[:, 1] / pts_color[:, 2] + K_color[1, 2]).astype(np.int32)
        in_bounds = (u_rgb >= 0) & (u_rgb < W) & (v_rgb >= 0) & (v_rgb < H)

        colors = np.zeros((len(z), 3), dtype=np.float64)
        colors[in_bounds] = color_bgr[v_rgb[in_bounds], u_rgb[in_bounds], ::-1].astype(np.float64) / 255.0

        if current_mask is not None and np.any(current_mask):
            highlight = np.zeros(len(z), dtype=bool)
            highlight[in_bounds] = current_mask[v_rgb[in_bounds], u_rgb[in_bounds]] > 0

            if np.any(highlight):
                colors[highlight] = colors[highlight] * 0.2 + MASK_COLOR_RGB * 0.8
                obj_pts = points_3d[highlight]
                if len(obj_pts) >= 10:
                    centroid = obj_pts.mean(axis=0)
                    dists = np.linalg.norm(obj_pts - centroid, axis=1)
                    filtered = obj_pts[dists <= np.percentile(dists, 90)]

                    if len(filtered) >= 10:
                        center = filtered.mean(axis=0)
                        
                        # ======== PCA 核心运算区 ========
                        cov = np.cov((filtered - center).T)
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        idx = np.argsort(eigenvalues)[::-1]
                        axes = eigenvectors[:, idx]
                        
                        # 确保基础右手定则
                        if np.linalg.det(axes) < 0: axes[:, 2] = -axes[:, 2]
                        
                        #  新增：物理法则校验拦截器 (完全抛弃 prev_axes) 
                        axes = validate_and_correct_axes(filtered, center, axes)
                        # ==============================================================

                        local = (filtered - center) @ axes
                        raw_extent = local.max(axis=0) - local.min(axis=0)
                        center = center + axes @ ((local.max(axis=0) + local.min(axis=0)) / 2)

                        extent_frame_count += 1
                        if obb_smooth_center is not None:
                            obb_smooth_center = OBB_SMOOTH * center + (1 - OBB_SMOOTH) * obb_smooth_center
                            obb_smooth_R = OBB_SMOOTH * axes + (1 - OBB_SMOOTH) * obb_smooth_R
                            u0 = obb_smooth_R[:, 0] / np.linalg.norm(obb_smooth_R[:, 0])
                            u1 = obb_smooth_R[:, 1] - np.dot(obb_smooth_R[:, 1], u0) * u0
                            u1 = u1 / np.linalg.norm(u1)
                            obb_smooth_R = np.column_stack([u0, u1, np.cross(u0, u1)])

                            extent_history.append(raw_extent.copy())
                            ext_alpha = max(EXTENT_ALPHA_MIN, EXTENT_ALPHA_INIT * (EXTENT_ALPHA_DECAY ** extent_frame_count))
                            candidate_extent = 0.5 * raw_extent + 0.5 * np.median(np.array(extent_history), axis=0) if len(extent_history) >= 3 else raw_extent
                            max_delta = obb_smooth_extent * EXTENT_MAX_CHANGE_RATE
                            obb_smooth_extent = ext_alpha * (obb_smooth_extent + np.clip(candidate_extent - obb_smooth_extent, -max_delta, max_delta)) + (1 - ext_alpha) * obb_smooth_extent
                        else:
                            obb_smooth_center = center.copy()
                            obb_smooth_extent = raw_extent.copy()
                            obb_smooth_R = axes.copy()
                            extent_history.append(raw_extent.copy())

                        corners_local = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1], [-1,-1, 1], [1,-1, 1], [1,1, 1], [-1,1, 1]], dtype=np.float64) * (obb_smooth_extent / 2)
                        corners_world = corners_local @ obb_smooth_R.T + obb_smooth_center
                        obb_edges = [[0,1],[1,2],[2,3],[3,0], [4,5],[5,6],[6,7],[7,4], [0,4],[1,5],[2,6],[3,7]]
                        obb_lineset.points = o3d.utility.Vector3dVector(corners_world)
                        obb_lineset.lines = o3d.utility.Vector2iVector(obb_edges)
                        obb_lineset.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(obb_edges))
                        
                        #  新增：实时更新并渲染 PCA 坐标系
                        pca_frame_temp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
                        T_pca = np.eye(4)
                        T_pca[:3, :3] = obb_smooth_R
                        T_pca[:3, 3] = obb_smooth_center
                        pca_frame_temp.transform(T_pca)
                        pca_frame.vertices = pca_frame_temp.vertices
                        pca_frame.vertex_colors = pca_frame_temp.vertex_colors
                        vis.update_geometry(pca_frame)
                        
                        #  新增：在这里将实时位姿发布到 ROS 2
                        if sam2_initialized:
                            pose_msg = PoseStamped()
                            pose_msg.header.stamp = ros_node.get_clock().now().to_msg()
                            pose_msg.header.frame_id = "camera_color_optical_frame" 
                            
                            # 填入位置 (X, Y, Z)
                            pose_msg.pose.position.x = float(obb_smooth_center[0])
                            pose_msg.pose.position.y = float(obb_smooth_center[1])
                            pose_msg.pose.position.z = float(obb_smooth_center[2])
                            
                            # 将旋转矩阵转化为四元数并填入
                            quat = SciPyRot.from_matrix(obb_smooth_R).as_quat()
                            pose_msg.pose.orientation.x = 0.00
                            pose_msg.pose.orientation.y = 0.00 
                            pose_msg.pose.orientation.z = float(quat[2])
                            pose_msg.pose.orientation.w = float(quat[3])

                            print('='*60)
                            print(f'发布相机坐标系下的XYZ：[{pose_msg.pose.position.x}, {pose_msg.pose.position.y}, {pose_msg.pose.position.z}]')
                            print(f'发布相机坐标系下的旋转四元数：[{pose_msg.pose.orientation.x}, {pose_msg.pose.orientation.y}, {pose_msg.pose.orientation.z}, {pose_msg.pose.orientation.w}]')
                            print('='*60)
                            pose_pub.publish(pose_msg)
                            
                    else:
                        obb_lineset.points, obb_lineset.lines = o3d.utility.Vector3dVector(np.zeros((0, 3))), o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
                else:
                    obb_lineset.points, obb_lineset.lines = o3d.utility.Vector3dVector(np.zeros((0, 3))), o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        else:
            obb_lineset.points, obb_lineset.lines = o3d.utility.Vector3dVector(np.zeros((0, 3))), o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))

        t1 = time.time()
        fps = 1.0 / (t1 - t0)

        cv2.putText(display, f"FPS: {fps:.1f}", (IMG_WIDTH - 130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        ir_status = "IR:ON" if ir_projector_enabled else "IR:OFF"
        if sam2_initialized:
            if obb_smooth_extent is not None:
                cv2.putText(display, f"BBox: {obb_smooth_extent[0]*100:.1f}x{obb_smooth_extent[1]*100:.1f}x{obb_smooth_extent[2]*100:.1f}cm", (10, IMG_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(display, f"TRACKING | {ir_status} | r=reset p=IR q=quit" if sam2_initialized else f"Draw bbox / Click to select | {ir_status} | q=quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("RGB + SAM2", display)

        pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if first_frame:
            vis.reset_view_point(True)
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1]); ctr.set_up([0, -1, 0])
            first_frame = False

        vis.update_geometry(pcd)
        vis.update_geometry(obb_lineset)
        vis.poll_events()
        vis.update_renderer()

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            print('用户按下R，重新检测') 
            need_reset = True
        elif key == ord('p'):
            ir_projector_enabled = not ir_projector_enabled
            print(f"IR 投影仪: {'ON' if ir_projector_enabled else 'OFF'}")
            if depth_sensor.supports(rs.option.emitter_enabled): depth_sensor.set_option(rs.option.emitter_enabled, 1 if ir_projector_enabled else 0)

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    vis.destroy_window()
    cv2.destroyAllWindows()
    # 新增：销毁 ROS 2 节点
    ros_node.destroy_node()
    rclpy.shutdown()
    logging.info("Exited")