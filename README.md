# Point Cloud Clustering and Processing 

## Overview

This project focuses on the analysis and processing of 3D point cloud data obtained from LiDAR sensors. Using the Open3D library, the project implements techniques for clustering and interpolating within dense regions of the point cloud for further analysis.

## Features

- **DBSCAN Clustering**: Identifies densely populated regions in the point cloud, facilitating noise reduction and data segmentation.
- **K-Nearest Neighbors (KNN) Interpolation**: Enhances the resolution of identified clusters by interpolating additional points within these dense regions.
- **Intensity-based Filtering**: Processes point clouds to include only those points that meet specified intensity and spatial criteria, ensuring that only relevant data is considered.

## Prerequisites

Install:
- Python 3.x
- Open3D
- NumPy
- scikit-learn

## Description

### Clustering with DBSCAN

**Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** is utilized to identify clusters within the point cloud. This algorithm groups points that are closely packed together while labeling points in low-density regions as noise. It operates based on two parameters:

- `eps`: The maximum distance between two points for them to be considered as part of the same cluster.
- `min_points`: The minimum number of points required to form a cluster.

### KNN Interpolation

After clustering, **K-Nearest Neighbors (KNN)** interpolation is applied within each cluster to interpolate additional points. This process involves:

1. Identifying the nearest neighbors for randomly generated points within the cluster's bounding box.
2. Applying inverse distance weighting (IDW) to estimate new point positions based on the distances and positions of these neighbors.

### Intensity-based Filtering

The point cloud is filtered based on the `x`, `y`, and `z` coordinates, as well as intensity values to ensure that only points within certain spatial and intensity thresholds are processed. 

## Data Flow

1. **Load Data**: Load point cloud data from `.bin` files.
2. **Preprocess Data**: Voxel down-sampling.
3. **Filter Data**: Spatial and intensity filtering.
4. **Cluster Data**: Apply DBSCAN clustering.
5. **Interpolate Data**: Enhance clusters using KNN interpolation.
6. **Visualize Data**: Display the processed data using Open3D visualization tools.

## Output

Processed point clouds are saved in the `saved_scenes/` directory as `.ply` files, which can be viewed using open 3D visualization.
