import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def density_equalize_point_cloud_with_intensity(cloud, intensities, voxel_size=0.1, radius_multiplier=2.0):
    """
    Equalize the density of the point cloud by interpolating sparse regions based on intensity.

    Parameters:
        cloud: open3d.geometry.PointCloud
            The input point cloud.
        intensities: ndarray
            The intensity values for each point in the point cloud.
        voxel_size: float
            The size of the voxel used for density equalization.
        radius_multiplier: float
            Multiplier to define the radius for density estimation.
    
    Returns:
        equalized_cloud: open3d.geometry.PointCloud
            The density-equalized point cloud.
        equalized_intensities: ndarray
            The corresponding intensity values for the equalized point cloud.
    """
    print("Equalizing density with intensity...")

    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)

    # Use a KDTree to estimate densities
    kdtree = cKDTree(points)
    radius = voxel_size * radius_multiplier

    # Estimate density by counting neighbors within a radius
    densities = np.array([len(kdtree.query_ball_point(p, radius)) for p in points])
    target_density = np.mean(densities)  # Target density

    # Add points in low-density regions
    new_points = []
    new_colors = []
    new_intensities = []

    for i, point in enumerate(points):
        density = densities[i]
        if density < target_density:
            # Number of points to add
            points_to_add = int(target_density - density)
            
            # Random offsets for new points
            random_offsets = np.random.uniform(-voxel_size, voxel_size, size=(points_to_add, 3))
            interpolated_points = point + random_offsets

            # Assign the same color and intensity to the new points
            interpolated_colors = np.tile(colors[i], (points_to_add, 1))
            interpolated_intensities = np.full(points_to_add, intensities[i])

            new_points.append(interpolated_points)
            new_colors.append(interpolated_colors)
            new_intensities.append(interpolated_intensities)

    if new_points:
        new_points = np.vstack(new_points)
        new_colors = np.vstack(new_colors)
        new_intensities = np.concatenate(new_intensities)

        all_points = np.vstack((points, new_points))
        all_colors = np.vstack((colors, new_colors))
        all_intensities = np.concatenate((intensities, new_intensities))
    else:
        all_points = points
        all_colors = colors
        all_intensities = intensities

    equalized_cloud = o3d.geometry.PointCloud()
    equalized_cloud.points = o3d.utility.Vector3dVector(all_points)
    equalized_cloud.colors = o3d.utility.Vector3dVector(all_colors)

    print(f"Density equalization complete. Final point count: {len(equalized_cloud.points)}")
    return equalized_cloud, all_intensities

def filter_cloud_by_xyz(cloud, intensities, x_max=35, y_max=35, z_min=-0.5, z_max=7):
    """
    Filter an Open3D point cloud and its intensities to include only points within:
    - |x| <= x_max
    - |y| <= y_max
    - z_min < z < z_max
    """

    points = np.asarray(cloud.points)
    
    indices = np.where(
        (np.abs(points[:, 0]) <= x_max) &  # |x| <= x_max
        (np.abs(points[:, 1]) <= y_max) &  # |y| <= y_max
        (points[:, 2] > z_min) &           # z > z_min
        (points[:, 2] < z_max) &           # z < z_max
        ~(np.all(points == 0, axis=1))     # Remove all-zero points
    )[0]
    # Filter the cloud and intensities
    filtered_cloud = cloud.select_by_index(indices)
    filtered_intensities = intensities[indices]
    filtered_points = points[indices]

    print(f"Filtered points count: {len(filtered_points)}")
    print(f"Filtered cloud: {len(filtered_cloud.points)} points remaining (|x| <= {x_max}, |y| <= {y_max}, {z_min} < z < {z_max}).")
    return filtered_cloud, filtered_intensities

def preprocess_point_cloud_with_intensity(cloud, intensities, voxel_size=0.1, radius_multiplier=2.0):
    
    print("Starting preprocessing with intensity...")

    # 1. Voxel down-sample for uniformity
    cloud = cloud.voxel_down_sample(voxel_size=voxel_size)
    print(f"Down-sampled to {len(cloud.points)} points.")
    cloud, intensities = density_equalize_point_cloud_with_intensity(cloud, intensities, voxel_size, radius_multiplier)
    print("Preprocessing complete.")
    return cloud, intensities

def load_bin_to_o3d(file_path):
    """
    Load a .bin point cloud into an Open3D point cloud object and preprocess it for clustering.
    """
    # Load the .bin file as a NumPy array
    # Each point is stored as x, y, z, intensity, r, g, b (6 floats per point)
    point_cloud_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 6)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])

    # Normalize intensity/rgb values
    rgb_values = point_cloud_data[:, 3:6]  
    rgb_values_normalized = rgb_values / 255.0  
    
    cloud.colors = o3d.utility.Vector3dVector(rgb_values_normalized)

    intensities = point_cloud_data[:, 3] 
    intensities_normalized = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # Normalize to [0, 1]

    print(f"Loaded point cloud with {len(cloud.points)} points, including intensity.")
    cloud, intensities_normalized = preprocess_point_cloud_with_intensity(cloud, intensities_normalized, 0.1)
    return cloud, intensities_normalized

def improved_euclidean_region_extraction(cloud, eps=0.3, min_points=10):
    
    # Perform DBSCAN clustering
    labels = np.array(cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    segments = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
        
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) > 0:
            segment = cloud.select_by_index(cluster_indices)
            segments.append(segment)
    
    print(f"Found {len(segments)} clusters using DBSCAN with eps={eps}, min_points={min_points}")
    return segments

if __name__ == "__main__":

    file_path = "lidar_points/1713937149753726976.bin"

    cloud, intensities = load_bin_to_o3d(file_path)
    filtered_cloud, filtered_intensities = filter_cloud_by_xyz(cloud, intensities, 
                                                            x_max=35, y_max=35, 
                                                            z_min=.1, z_max=10)

    segments = improved_euclidean_region_extraction(filtered_cloud, 
                                                    eps=0.7,  
                                                    min_points=5)  
    
    for i, segment in enumerate(segments):
        color = np.random.rand(3)    
        segment.paint_uniform_color(color)

    combined_cloud = o3d.geometry.PointCloud()

    for segment in segments:
        combined_cloud += segment

    o3d.visualization.draw_geometries(segments)
    o3d.io.write_point_cloud("saved_scenes/clustered_scene.ply", combined_cloud)