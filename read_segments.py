import open3d as o3d

combined_scene = o3d.io.read_point_cloud("saved_scenes/clustered_interpolated_scene.ply")

o3d.visualization.draw_geometries([combined_scene])
