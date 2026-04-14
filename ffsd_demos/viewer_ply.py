import open3d as o3d

# 加载 PLY 格式的点云文件
point_cloud = o3d.io.read_point_cloud("/home/xense/Fast-FoundationStereoPose/ffsd_demos/aaaa_denoised.ply")

# 显示点云
o3d.visualization.draw_geometries([point_cloud], window_name="PLY Point Cloud")
