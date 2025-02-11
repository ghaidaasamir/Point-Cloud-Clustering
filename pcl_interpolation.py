import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata

from sklearn.neighbors import NearestNeighbors

def interpolate_within_clusters(cloud, eps=0.5, min_points=10, num_interpolated_points_per_cluster=1000):
    """
    Interpolate within dense regions of a point cloud using DBSCAN clustering and KNN interpolation.
    
    Args:
        cloud (o3d.geometry.PointCloud): The input point cloud.
        eps (float): DBSCAN parameter for maximum distance between points in a cluster.
        min_points (int): DBSCAN parameter for minimum points to form a cluster.
        num_interpolated_points_per_cluster (int): Number of points to interpolate per cluster.
    
    Returns:
        interpolated_cloud (o3d.geometry.PointCloud): The point cloud with interpolated points added.
    """
    # Convert the point cloud to a NumPy array
    points = np.asarray(cloud.points)
    
    # Perform DBSCAN clustering to identify dense regions
    labels = np.array(cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    unique_labels = np.unique(labels)
    
    # Initialize list to store interpolated points
    all_interpolated_points = []
    
    # Iterate over each cluster (ignore noise, label = -1)
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points
        
        # Extract points in the current cluster
        cluster_indices = np.where(labels == label)[0]
        cluster_points = points[cluster_indices]
        
        # Skip if the cluster is too small
        if len(cluster_points) < min_points:
            continue
        
        # Interpolate within the cluster using KNN
        knn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(cluster_points)
        
        # Generate random points within the bounds of the cluster
        min_bounds = np.min(cluster_points, axis=0)
        max_bounds = np.max(cluster_points, axis=0)
        random_points = np.random.uniform(min_bounds, max_bounds, size=(num_interpolated_points_per_cluster, 3))
        
        # Find nearest neighbors for the random points
        distances, indices = knn.kneighbors(random_points)
        
        # Interpolate using inverse distance weighting (IDW)
        weights = 1.0 / (distances + 1e-6)  # Avoid division by zero
        weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights
        interpolated_points = np.sum(weights[:, :, np.newaxis] * cluster_points[indices], axis=1)
        
        # Add interpolated points to the list
        all_interpolated_points.append(interpolated_points)
    
    # Combine all interpolated points into a single array
    if len(all_interpolated_points) > 0:
        all_interpolated_points = np.vstack(all_interpolated_points)
    else:
        all_interpolated_points = np.empty((0, 3))  # No points to interpolate
    
    # Combine original points and interpolated points
    combined_points = np.vstack([points, all_interpolated_points])
    
    # Create a new Open3D point cloud
    interpolated_cloud = o3d.geometry.PointCloud()
    interpolated_cloud.points = o3d.utility.Vector3dVector(combined_points)
    
    return interpolated_cloud

def interpolate_points_knn(points, intensities, num_interpolated_points=1000):
    """
    Interpolate between points using K-Nearest Neighbors (KNN).
    
    Args:
        points (np.ndarray): The original point cloud (N x 3).
        intensities (np.ndarray): The corresponding intensities (N x 1).
        num_interpolated_points (int): Number of points to interpolate.
    
    Returns:
        interpolated_points (np.ndarray): Interpolated points (M x 3).
        interpolated_intensities (np.ndarray): Interpolated intensities (M x 1).
    """
    # Fit KNN model to the original points
    knn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(points)
    
    # Generate random points within the bounds of the original point cloud
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    random_points = np.random.uniform(min_bounds, max_bounds, size=(num_interpolated_points, 3))
    
    # Find nearest neighbors for the random points
    distances, indices = knn.kneighbors(random_points)
    
    # Interpolate intensities using inverse distance weighting
    weights = 1.0 / (distances + 1e-6)  # Avoid division by zero
    weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights
    interpolated_intensities = np.sum(weights * intensities[indices], axis=1)
    
    # Return interpolated points and intensities
    return random_points, interpolated_intensities

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

    o3d.io.write_point_cloud("saved_scenes/original_scene.ply", cloud)

    filtered_cloud, filtered_intensities = filter_cloud_by_xyz(cloud, intensities, 
                                                             x_max=35, y_max=35, 
                                                             z_min=0.1, z_max=10)
    

    o3d.io.write_point_cloud("saved_scenes/filtered_scene.ply", filtered_cloud)

    interpolated_cloud = interpolate_within_clusters(filtered_cloud, eps=0.5, min_points=10, num_interpolated_points_per_cluster=1000)
    
    o3d.io.write_point_cloud("saved_scenes/interpolated_scene.ply", interpolated_cloud)

    o3d.visualization.draw_geometries([filtered_cloud, interpolated_cloud])

