
import open3d as o3d
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def filter_mesh_by_cylinder(mesh, cylinder_radius, cylinder_height, cylinder_center=None):
    """
    Filters the mesh to keep only the vertices inside a bounding cylinder.
    
    Parameters:
    - mesh (o3d.geometry.TriangleMesh): The mesh to filter.
    - cylinder_radius (float): The radius of the bounding cylinder.
    - cylinder_height (float): The height of the bounding cylinder.
    - cylinder_center (tuple): The center of the cylinder (x, y, z). If None, defaults to (0, 0, 0).
    
    Returns:
    - o3d.geometry.TriangleMesh: The filtered mesh.
    """
    if cylinder_center is None:
        cylinder_center = np.array([0, 0, 0])

    # Clean up the mesh (remove degenerate triangles and duplicated triangles)
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    # Get vertices of the mesh
    vertices = np.asarray(mesh.vertices)
    
    # Define the cylinder height along the Z-axis and the radius in the XY plane
    filtered_vertices = []
    for vertex in vertices:
        # Translate vertex by the cylinder center
        translated_vertex = vertex - cylinder_center
        x, y, z = translated_vertex
        
        # Check if the vertex is within the cylindrical bounds
        if np.sqrt(x**2 + y**2) <= cylinder_radius and abs(z) <= cylinder_height / 2:
            filtered_vertices.append(vertex)
    
    # Create a new mesh with the filtered vertices
    indices_to_keep = np.where(np.isin(vertices, filtered_vertices))[0]
    filtered_mesh = mesh.select_by_index(indices_to_keep)
    
    return filtered_mesh


def visualize_filtered_mesh(filtered_mesh):
    """Visualizes the filtered mesh."""
    o3d.visualization.draw_geometries([filtered_mesh], window_name="Filtered Mesh by Cylinder")


def remove_degenerate_triangles(mesh):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    # Check if the area of each triangle is zero
    degenerate_indices = []
    for i, tri in enumerate(triangles):
        v0, v1, v2 = vertices[tri]
        # Compute the area of the triangle using the cross product
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0
        if area < 1e-6:  # Threshold for degenerate triangles
            degenerate_indices.append(i)
    
    # Remove degenerate triangles
    triangles = np.delete(triangles, degenerate_indices, axis=0)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh

def check_point_inside_cylinder(point, cylinder_center, radius, height):
    # Check if the point is inside the cylindrical region
    distance_to_axis = np.linalg.norm(point[:2] - cylinder_center[:2])  # Calculate distance in the x-y plane
    within_radius = distance_to_axis <= radius
    within_height = cylinder_center[2] - height / 2 <= point[2] <= cylinder_center[2] + height / 2
    return within_radius and within_height

def filter_and_color_points_by_cylinder(mesh, cylinder_center, radius, height):
    # Get the vertices of the mesh
    vertices = np.asarray(mesh.vertices)
    
    # Create a list of colors where points inside the cylinder are green, and points outside are red
    colors = []
    for vertex in vertices:
        if check_point_inside_cylinder(vertex, cylinder_center, radius, height):
            colors.append([0, 1, 0])  # Green for inside
        else:
            colors.append([1, 0, 0])  # Red for outside
    
    # Assign colors to the mesh vertices
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh

def filter_points_in_cylinder_min_max(pcd, cylinder_axis, cylinder_center, cylinder_radius, cylinder_height_min, cylinder_height_max, output_file=None):
    """
    Filters the points in the point cloud that lie within a specified cylinder.
    
    Parameters:
    - pcd: The input point cloud (Open3D PointCloud object).
    - cylinder_axis: The axis of the cylinder (e.g., np.array([0, 0, 1])).
    - cylinder_center: The center of the cylinder (e.g., np.array([0, 0, 0])).
    - cylinder_radius: The radius of the cylinder.
    - cylinder_height_min: The minimum height of the cylinder.
    - cylinder_height_max: The maximum height of the cylinder.
    - output_file: The path to save the filtered point cloud (optional).
    
    Returns:
    - filtered_pcd: The filtered point cloud (Open3D PointCloud object).
    """
    
    # Convert to numpy array for easier manipulation
    points = np.asarray(pcd.points)

    # Check if color information is available
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(points)  # Set to white if no color info is available

    # Calculate distance of each point from the axis (ignoring z-component)
    distances = np.linalg.norm(points[:, :2], axis=1)  # Only x and y for the distance from axis

    # Check if points are within the radius and height
    within_radius = distances <= cylinder_radius
    within_height = (points[:, 2] >= cylinder_height_min) & (points[:, 2] <= cylinder_height_max)

    # Combine both conditions
    valid_points = points[within_radius & within_height]
    valid_colors = colors[within_radius & within_height]  # Keep the colors of the valid points

    # Create a new point cloud with the valid points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(valid_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(valid_colors)  # Assign the original colors

    # Visualize the filtered point cloud
    o3d.visualization.draw_geometries([filtered_pcd])

    # Optionally, save the filtered point cloud to a new file
    if output_file:
        o3d.io.write_point_cloud(output_file, filtered_pcd)
        print(f"Filtered point cloud saved to {output_file}")

    return filtered_pcd

def find_cylinder_axis_and_center(mesh):
    """
    Given a mesh, this function returns the axis direction and center of a cylinder-like structure.
    It uses PCA (Principal Component Analysis) to find the axis and calculates the center as the mean of the points.
    
    Parameters:
    - mesh (open3d.geometry.TriangleMesh): The input mesh to process.
    
    Returns:
    - tuple: (cylinder_axis, cylinder_center)
      - cylinder_axis (numpy.ndarray): The direction of the cylinder's axis.
      - cylinder_center (numpy.ndarray): The center of the cylinder.
    """
    # Extract the points of the mesh
    vertices = np.asarray(mesh.vertices)

    # Run PCA to find the central axis
    pca = PCA(n_components=2)
    pca.fit(vertices)

    # Get the direction of the principal component (the cylinder's axis)
    cylinder_axis = pca.components_[0]  # The first component is the direction of the axis

    # The center of the cylinder can be estimated as the mean of the points
    cylinder_center = np.mean(vertices, axis=0)

    # Optionally, visualize the cylinder axis along with the mesh
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([cylinder_center, cylinder_center + 10 * cylinder_axis])  # Example line length
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])

    # Visualize the mesh with the cylinder axis
    o3d.visualization.draw_geometries([mesh, line_set])

    return cylinder_axis, cylinder_center



def plot_cylinder_axis(cylinder_center, cylinder_axis):
    """
    Plots the cylinder axis in 3D based on the provided center and axis vector.

    Parameters:
    - cylinder_center: numpy array or list with the coordinates of the cylinder center [x, y, z].
    - cylinder_axis: numpy array or list with the direction of the cylinder axis [dx, dy, dz].
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the cylinder axis vector
    ax.quiver(cylinder_center[0], cylinder_center[1], cylinder_center[2],
              cylinder_axis[0], cylinder_axis[1], cylinder_axis[2], 
              length=1, normalize=True, color='r', label="Cylinder Axis")

    # Set plot limits for better visualization
    ax.set_xlim([cylinder_center[0] - 1, cylinder_center[0] + 1])
    ax.set_ylim([cylinder_center[1] - 1, cylinder_center[1] + 1])
    ax.set_zlim([cylinder_center[2] - 1, cylinder_center[2] + 1])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.legend()
    plt.show()

def filter_mesh_by_bounding_box(mesh, min_bound, max_bound):
    """
    Filters the mesh to include only points within the given bounding box.
    
    :param mesh: The input Open3D mesh.
    :param min_bound: The minimum x, y, z coordinates for the bounding box.
    :param max_bound: The maximum x, y, z coordinates for the bounding box.
    :return: The filtered mesh.
    """
    # Convert mesh vertices to numpy array
    vertices = np.asarray(mesh.vertices)

    # Filter points based on the bounding box
    filtered_points = vertices[
        (vertices[:, 0] >= min_bound[0]) & (vertices[:, 0] <= max_bound[0]) &
        (vertices[:, 1] >= min_bound[1]) & (vertices[:, 1] <= max_bound[1]) &
        (vertices[:, 2] >= min_bound[2]) & (vertices[:, 2] <= max_bound[2])
    ]
    
    # Create a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # Visualize the filtered mesh
    o3d.visualization.draw_geometries([filtered_pcd])
    
    return filtered_pcd

def find_plane_normal(mesh, z_tolerance=0.1, distance_threshold=1.0):
    # Convert the mesh vertices to numpy array
    vertices = np.asarray(mesh.vertices)
    
    # Filter points that are near the x-y plane (z-values close to 0)
    # You can adjust z_tolerance to decide how strict the filter should be
    plane_points = vertices[np.abs(vertices[:, 2]) < z_tolerance]
    
    # If there are not enough points to perform PCA, return an error message
    if len(plane_points) < 3:
        raise ValueError("Not enough points in the x-y plane to determine a plane normal.")

    # Apply PCA to find the principal components
    pca = PCA(n_components=3)
    pca.fit(plane_points)

    # The normal vector of the plane corresponds to the smallest principal component
    normal_vector = pca.components_[-1]  # Smallest component corresponds to the normal

    # Return the normal vector and the center of the plane (mean of the points)
    plane_center = np.mean(plane_points, axis=0)

    return normal_vector, plane_center

def filter_mesh_by_box(mesh, min_bound, max_bound):
    """
    Filters the mesh to keep only the vertices inside a bounding box.
    
    Parameters:
    - mesh (o3d.geometry.TriangleMesh): The mesh to filter.
    - min_bound (array-like): The minimum (x, y, z) coordinates of the bounding box.
    - max_bound (array-like): The maximum (x, y, z) coordinates of the bounding box.
    
    Returns:
    - o3d.geometry.TriangleMesh: The filtered mesh.
    """
    # Ensure bounds are numpy arrays
    min_bound = np.array(min_bound)
    max_bound = np.array(max_bound)

    # Clean up the mesh (remove degenerate and duplicated triangles)
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    # Get mesh vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Find vertices inside the bounding box
    mask = np.all((vertices >= min_bound) & (vertices <= max_bound), axis=1)

    # Get indices of vertices to keep
    indices_to_keep = np.where(mask)[0]

    # Filter mesh based on indices
    filtered_mesh = mesh.select_by_index(indices_to_keep)

    return filtered_mesh


def find_plane_normal(mesh):
    """
    Computes the normal vector of a planar mesh using PCA.
    
    Parameters:
    - mesh (o3d.geometry.TriangleMesh): The input mesh containing a plane.
    
    Returns:
    - normal (numpy array): The normal vector to the plane.
    """
    # Extract vertices as a NumPy array
    vertices = np.asarray(mesh.vertices)
    
    # Run PCA on the vertex positions
    pca = PCA(n_components=3)
    pca.fit(vertices)
    
    # The normal to the plane is the third principal component (least variance direction)
    normal = pca.components_[-1]  # Last component corresponds to the smallest variance
    
    return normal


def plot_plane_normal(mesh, normal, scale=1.0):
    """
    Plots the normal vector of a plane in 3D space.
    
    Parameters:
    - mesh (o3d.geometry.TriangleMesh): The input mesh (assumed to be a plane).
    - normal (numpy array): The normal vector to the plane.
    - scale (float): Scaling factor for the normal vector length.
    """
    # Compute the centroid of the plane (to position the normal vector)
    centroid = np.mean(np.asarray(mesh.vertices), axis=0)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the plane points for reference
    vertices = np.asarray(mesh.vertices)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, alpha=0.5, label="Plane Points")
    
    # Plot the normal vector
    ax.quiver(
        centroid[0], centroid[1], centroid[2],  # Start point (centroid)
        normal[0], normal[1], normal[2],  # Direction
        length=scale, normalize=True, color='r', label="Normal Vector"
    )
    
    # Set plot limits
    ax.set_xlim([centroid[0] - 1, centroid[0] + 1])
    ax.set_ylim([centroid[1] - 1, centroid[1] + 1])
    ax.set_zlim([centroid[2] - 1, centroid[2] + 1])
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Show plot
    plt.show()


def plot_normal_vector(origin, normal, scale=1.0):
    """
    Plots the normal vector in 3D space.
    
    Parameters:
    - origin (numpy array): The starting point of the normal vector.
    - normal (numpy array): The normal vector direction.
    - scale (float): Scaling factor for the normal vector length.
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the normal vector
    ax.quiver(
        origin[0], origin[1], origin[2],  # Start point
        normal[0], normal[1], normal[2],  # Direction
        length=scale, normalize=True, color='r', label="Normal Vector"
    )
    
    # Set plot limits around the origin
    ax.set_xlim([origin[0] - scale, origin[0] + scale])
    ax.set_ylim([origin[1] - scale, origin[1] + scale])
    ax.set_zlim([origin[2] - scale, origin[2] + scale])
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Show plot
    plt.show()


def angle_between_vectors(v1, v2):
    """
    Computes the angle (in degrees) between two vectors.
    
    Parameters:
    - v1 (numpy array): First vector.
    - v2 (numpy array): Second vector.
    
    Returns:
    - float: Angle between the vectors in degrees.
    """
    # Normalize the vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Compute the dot product
    dot_product = np.dot(v1, v2)
    
    # Clip to avoid numerical errors outside valid range of arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Compute the angle in radians and convert to degrees
    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees



def plot_meshes_with_vectors(mesh_plate, normal_vector, plate_center, 
                             mesh_cylinder, cylinder_axis, cylinder_center):
    """
    Plots the mesh plate, its normal vector, the mesh cylinder, and its axis in 3D with square axes.
    
    Parameters:
    - mesh_plate (o3d.geometry.TriangleMesh): The plane mesh.
    - normal_vector (np.ndarray): Normal vector of the plane.
    - plate_center (np.ndarray): Center of the plane.
    - mesh_cylinder (o3d.geometry.TriangleMesh): The cylindrical mesh.
    - cylinder_axis (np.ndarray): Cylinder axis vector.
    - cylinder_center (np.ndarray): Center of the cylinder.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert Open3D meshes to numpy arrays for plotting
    plate_vertices = np.asarray(mesh_plate.vertices)
    cylinder_vertices = np.asarray(mesh_cylinder.vertices)
    angle = angle_between_vectors(normal_vector, cylinder_axis)
    
    # Plot the plate mesh vertices
    ax.scatter(plate_vertices[:, 0], plate_vertices[:, 1], plate_vertices[:, 2], 
               c='blue', s=1, alpha=0.2, #label="Plate Mesh"
              )
    
    # Plot the cylinder mesh vertices
    ax.scatter(cylinder_vertices[:, 0], cylinder_vertices[:, 1], cylinder_vertices[:, 2], 
               c='green', s=1, alpha=0.2, #label="Cylinder Mesh"
              )
    
    # Plot the normal vector of the plate
    ax.quiver(plate_center[0], plate_center[1], plate_center[2], 
              normal_vector[0], normal_vector[1], normal_vector[2], 
              length=3, normalize=True, color='red', alpha=0.5,
              label=f"Plate Normal: ({normal_vector[0]:.1e}, {normal_vector[1]:.1e}, {normal_vector[2]:.1e})", 
              arrow_length_ratio=0.8)
    
    # Plot the cylinder axis
    ax.quiver(cylinder_center[0], cylinder_center[1], cylinder_center[2], 
              cylinder_axis[0], cylinder_axis[1], cylinder_axis[2], 
              length=3, normalize=True, color='black', alpha=0.5,
              label=f"Cylinder Axis: ({cylinder_axis[0]:.1e}, {cylinder_axis[1]:.1e}, {cylinder_axis[2]:.1e})",
              arrow_length_ratio=0.8)
    ax.scatter([],[], label=f'Angle: {angle:.2f} deg', color='white')
    ax.scatter([],[], label="Plate mesh", color='blue', s=5)
    ax.scatter([],[], label="Plumb bob mesh", color='green', s=5)
    
    # Set axis limits based on data range to make the plot square
    all_points = np.vstack((plate_vertices, cylinder_vertices, 
                            plate_center, cylinder_center))  # Combine all relevant points
    x_limits = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
    y_limits = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
    z_limits = [np.min(all_points[:, 2]), np.max(all_points[:, 2])]
    
    # Find the max range
    max_range = max(x_limits[1] - x_limits[0], 
                    y_limits[1] - y_limits[0], 
                    z_limits[1] - z_limits[0]) / 2.0

    # Find midpoints
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    # Set limits symmetrically around the midpoint
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def filter_points_in_cylinder(pcd, cylinder_axis, cylinder_center, cylinder_radius, cylinder_height, output_file=None, visualization=True):
    """
    Filters the points in the point cloud that lie within a specified cylinder.
    
    Parameters:
    - pcd: The input point cloud (Open3D PointCloud object).
    - cylinder_axis: The axis of the cylinder (e.g., np.array([0, 0, 1])).
    - cylinder_center: The center of the cylinder (e.g., np.array([0, 0, 0])).
    - cylinder_radius: The radius of the cylinder.
    - cylinder_height: The height of the cylinder (the total height from the center, extending ± height/2 along the axis).
    - output_file: The path to save the filtered point cloud (optional).
    
    Returns:
    - filtered_pcd: The filtered point cloud (Open3D PointCloud object).
    """
    
    # Convert to numpy array for easier manipulation
    points = np.asarray(pcd.points)

    # Check if color information is available
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(points)  # Set to white if no color info is available

    # Compute vector from cylinder center to each point
    vectors = points - cylinder_center

    # Project each vector onto the cylinder axis
    projected_lengths = np.dot(vectors, cylinder_axis)  # Scalar projection onto the axis
    projected_points = np.outer(projected_lengths, cylinder_axis)  # Convert scalars to vectors

    # Compute the radial distance from the cylinder axis (perpendicular distance)
    radial_vectors = vectors - projected_points
    radial_distances = np.linalg.norm(radial_vectors, axis=1)

    # Compute height limits based on the center
    half_height = cylinder_height / 2
    within_height = np.abs(projected_lengths) <= half_height

    # Check if points are within the radius and height limits
    within_radius = radial_distances <= cylinder_radius

    # Combine both conditions
    valid_mask = within_radius & within_height
    valid_points = points[valid_mask]
    valid_colors = colors[valid_mask]  # Keep the colors of the valid points

    # Create a new point cloud with the valid points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(valid_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(valid_colors)  # Assign original colors

    # Visualize the filtered point cloud
    if visualization:
        o3d.visualization.draw_geometries([filtered_pcd])

    # Optionally, save the filtered point cloud to a new file
    if output_file:
        o3d.io.write_point_cloud(output_file, filtered_pcd)
        print(f"Filtered point cloud saved to {output_file}")

    return filtered_pcd


def find_cylinder_axis_center_radius(geometry):
    """
    Given a point cloud or mesh, this function returns the axis direction, center, and radius of a cylinder-like structure.
    It uses PCA (Principal Component Analysis) to find the axis and calculates the center as the mean of the points.
    The radius is estimated as the mean radial distance from the axis.

    Parameters:
    - geometry (open3d.geometry.TriangleMesh or open3d.geometry.PointCloud): The input mesh or point cloud to process.

    Returns:
    - tuple: (cylinder_axis, cylinder_center, cylinder_radius)
      - cylinder_axis (numpy.ndarray): The direction of the cylinder's axis.
      - cylinder_center (numpy.ndarray): The center of the cylinder.
      - cylinder_radius (float): The estimated radius of the cylinder.
    """
    # Extract points from the input geometry
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        points = np.asarray(geometry.vertices)
    elif isinstance(geometry, o3d.geometry.PointCloud):
        points = np.asarray(geometry.points)
    else:
        raise TypeError("Input must be an Open3D TriangleMesh or PointCloud.")

    # Run PCA to find the central axis
    pca = PCA(n_components=2)
    pca.fit(points)

    # Get the direction of the principal component (the cylinder's axis)
    cylinder_axis = pca.components_[0]  # The first component is the direction of the axis
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)  # Normalize to unit vector

    # The center of the cylinder can be estimated as the mean of the points
    cylinder_center = np.mean(points, axis=0)

    # Compute radial distances to estimate the radius
    vectors_to_points = points - cylinder_center

    # Project onto the cylinder axis to get the axial component
    axial_components = np.dot(vectors_to_points, cylinder_axis)[:, np.newaxis] * cylinder_axis

    # Compute the radial component (perpendicular to the axis)
    radial_components = vectors_to_points - axial_components

    # Compute the distances from the cylinder axis
    radial_distances = np.linalg.norm(radial_components, axis=1)

    # Estimate the cylinder radius as the mean radial distance
    cylinder_radius = np.mean(radial_distances)

    # Optionally, visualize the cylinder axis along with the geometry
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([cylinder_center, cylinder_center + 10 * cylinder_axis])  # Example line length
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])

    # Visualize the geometry with the cylinder axis
    o3d.visualization.draw_geometries([geometry, line_set])

    return cylinder_axis, cylinder_center, cylinder_radius


def find_cylinder_axis_and_center_pcd(point_cloud, visualization=True):
    """
    Given a dense point cloud, this function returns the axis direction and center of a cylinder-like structure.
    It uses PCA (Principal Component Analysis) to find the axis and calculates the center as the mean of the points.
    
    Parameters:
    - point_cloud (open3d.geometry.PointCloud): The input point cloud to process.
    
    Returns:
    - tuple: (cylinder_axis, cylinder_center)
      - cylinder_axis (numpy.ndarray): The direction of the cylinder's axis.
      - cylinder_center (numpy.ndarray): The center of the cylinder.
    """
    # Extract the points from the point cloud
    points = np.asarray(point_cloud.points)

    # Run PCA to find the central axis
    pca = PCA(n_components=2)
    pca.fit(points)

    # Get the direction of the principal component (the cylinder's axis)
    cylinder_axis = pca.components_[0]  # The first component is the direction of the axis

    # The center of the cylinder can be estimated as the mean of the points
    cylinder_center = np.mean(points, axis=0)

    # Optionally, visualize the cylinder axis along with the point cloud
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([cylinder_center, cylinder_center + 10 * cylinder_axis])  # Example line length
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    if visualization:
        o3d.visualization.draw_geometries([point_cloud, line_set])

        
    return cylinder_axis, cylinder_center
def compute_cylinder_variance(geometry, cylinder_axis, cylinder_center, cylinder_radius):
    """
    Computes the variance of the distances of points from the ideal cylinder surface.

    Parameters:
    - geometry: The input geometry (Open3D TriangleMesh or PointCloud object).
    - cylinder_axis: The unit vector along the cylinder's axis (numpy array).
    - cylinder_center: A point on the cylinder axis, defining its center (numpy array).
    - cylinder_radius: The expected radius of the cylinder.

    Returns:
    - variance: The variance of the distance errors.
    """
    # Extract points from the input geometry
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        points = np.asarray(geometry.vertices)
    elif isinstance(geometry, o3d.geometry.PointCloud):
        points = np.asarray(geometry.points)
    else:
        raise TypeError("Input must be an Open3D TriangleMesh or PointCloud.")

    # Normalize the cylinder axis to ensure it's a unit vector
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)

    # Compute the vector from the cylinder center to each point
    vectors_to_points = points - cylinder_center

    # Project these vectors onto the cylinder axis to get the axial component
    axial_components = np.dot(vectors_to_points, cylinder_axis)[:, np.newaxis] * cylinder_axis

    # Compute the radial component (perpendicular to the cylinder axis)
    radial_components = vectors_to_points - axial_components

    # Compute the distances from the cylinder axis (should ideally be equal to cylinder_radius)
    radial_distances = np.linalg.norm(radial_components, axis=1)

    # Compute the squared error from the expected radius
    errors = (radial_distances - cylinder_radius) ** 2

    # Compute variance
    variance = np.var(errors)

    return variance


def compute_binned_cylinder_variance(geometry, cylinder_axis, cylinder_center, cylinder_radius, num_bins=20):
    """
    Computes the variance of the distances of points from the ideal cylinder surface, 
    binned by their z-coordinates.

    Parameters:
    - geometry: The input geometry (Open3D TriangleMesh or PointCloud object).
    - cylinder_axis: The unit vector along the cylinder's axis (numpy array).
    - cylinder_center: A point on the cylinder axis, defining its center (numpy array).
    - cylinder_radius: The expected radius of the cylinder.
    - num_bins: The number of bins for z-coordinates.

    Returns:
    - bin_centers: The center z-values of the bins.
    - binned_variances: The variance of radial distances in each z-bin.
    """
    # Extract points from the input geometry
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        points = np.asarray(geometry.vertices)
    elif isinstance(geometry, o3d.geometry.PointCloud):
        points = np.asarray(geometry.points)
    else:
        raise TypeError("Input must be an Open3D TriangleMesh or PointCloud.")

    # Normalize the cylinder axis to ensure it's a unit vector
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)

    # Compute the vector from the cylinder center to each point
    vectors_to_points = points - cylinder_center

    # Project these vectors onto the cylinder axis to get the axial (z) component
    axial_components = np.dot(vectors_to_points, cylinder_axis)[:, np.newaxis] * cylinder_axis

    # Compute the radial component (perpendicular to the cylinder axis)
    radial_components = vectors_to_points - axial_components

    # Compute the distances from the cylinder axis (should ideally be equal to cylinder_radius)
    radial_distances = np.linalg.norm(radial_components, axis=1)

    # Get z-coordinates along the cylinder axis
    z_coords = np.dot(vectors_to_points, cylinder_axis)

    # Define bins for z-coordinates
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    bins = np.linspace(z_min, z_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers

    # Compute variance in each bin
    binned_variances = []
    for i in range(num_bins):
        in_bin = (z_coords >= bins[i]) & (z_coords < bins[i + 1])
        if np.sum(in_bin) > 1:  # Ensure we have at least two points to compute variance
            variance = np.var((radial_distances[in_bin] - cylinder_radius) ** 2)
        else:
            variance = np.nan  # Not enough points to compute variance
        binned_variances.append(variance)

    return bin_centers, np.array(binned_variances)

def compute_and_plot_binned_cylinder_variance(geometry, cylinder_axis, cylinder_center, cylinder_radius, num_bins=20):
    """
    Computes and plots the variance of the distances of points from the ideal cylinder surface, 
    binned by their z-coordinates.

    Parameters:
    - geometry: The input geometry (Open3D TriangleMesh or PointCloud object).
    - cylinder_axis: The unit vector along the cylinder's axis (numpy array).
    - cylinder_center: A point on the cylinder axis, defining its center (numpy array).
    - cylinder_radius: The expected radius of the cylinder.
    - num_bins: The number of bins for z-coordinates.

    Returns:
    - bin_centers: The center z-values of the bins.
    - binned_variances: The variance of radial distances in each z-bin.
    """
    # Extract points from the input geometry
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        points = np.asarray(geometry.vertices)
    elif isinstance(geometry, o3d.geometry.PointCloud):
        points = np.asarray(geometry.points)
    else:
        raise TypeError("Input must be an Open3D TriangleMesh or PointCloud.")

    # Normalize the cylinder axis to ensure it's a unit vector
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)

    # Compute the vector from the cylinder center to each point
    vectors_to_points = points - cylinder_center

    # Project these vectors onto the cylinder axis to get the axial (z) component
    axial_components = np.dot(vectors_to_points, cylinder_axis)[:, np.newaxis] * cylinder_axis

    # Compute the radial component (perpendicular to the cylinder axis)
    radial_components = vectors_to_points - axial_components

    # Compute the distances from the cylinder axis (should ideally be equal to cylinder_radius)
    radial_distances = np.linalg.norm(radial_components, axis=1)

    # Get z-coordinates along the cylinder axis
    z_coords = np.dot(vectors_to_points, cylinder_axis)

    # Define bins for z-coordinates
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    bins = np.linspace(z_min, z_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers

    # Compute variance in each bin
    binned_variances = []
    for i in range(num_bins):
        in_bin = (z_coords >= bins[i]) & (z_coords < bins[i + 1])
        if np.sum(in_bin) > 1:  # Ensure we have at least two points to compute variance
            variance = np.var((radial_distances[in_bin] - cylinder_radius) ** 2)
        else:
            variance = np.nan  # Not enough points to compute variance
        binned_variances.append(variance)

    # Convert to numpy arrays for easier handling
    bin_centers = np.array(bin_centers)
    binned_variances = np.array(binned_variances)

    # Plot the variance as a function of z
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, binned_variances, marker='o', linestyle='-', color='b', label="Radial Variance")
    plt.xlabel("Z Coordinate Along Cylinder Axis")
    plt.ylabel("Radial Variance")
    plt.title("Radial Variance as a Function of Z")
    plt.legend()
    plt.grid(True)
    plt.show()

    return bin_centers, binned_variances

def compute_and_plot_binned_cylinder_std(geometry, cylinder_axis, cylinder_center, cylinder_radius, num_bins=20):
    """
    Computes and plots the variance of the distances of points from the ideal cylinder surface, 
    binned by their z-coordinates.

    Parameters:
    - geometry: The input geometry (Open3D TriangleMesh or PointCloud object).
    - cylinder_axis: The unit vector along the cylinder's axis (numpy array).
    - cylinder_center: A point on the cylinder axis, defining its center (numpy array).
    - cylinder_radius: The expected radius of the cylinder.
    - num_bins: The number of bins for z-coordinates.

    Returns:
    - bin_centers: The center z-values of the bins.
    - binned_variances: The variance of radial distances in each z-bin.
    """
    # Extract points from the input geometry
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        points = np.asarray(geometry.vertices)
    elif isinstance(geometry, o3d.geometry.PointCloud):
        points = np.asarray(geometry.points)
    else:
        raise TypeError("Input must be an Open3D TriangleMesh or PointCloud.")

    # Normalize the cylinder axis to ensure it's a unit vector
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)

    # Compute the vector from the cylinder center to each point
    vectors_to_points = points - cylinder_center

    # Project these vectors onto the cylinder axis to get the axial (z) component
    axial_components = np.dot(vectors_to_points, cylinder_axis)[:, np.newaxis] * cylinder_axis

    # Compute the radial component (perpendicular to the cylinder axis)
    radial_components = vectors_to_points - axial_components

    # Compute the distances from the cylinder axis (should ideally be equal to cylinder_radius)
    radial_distances = np.linalg.norm(radial_components, axis=1)

    # Get z-coordinates along the cylinder axis
    z_coords = np.dot(vectors_to_points, cylinder_axis)

    # Define bins for z-coordinates
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    bins = np.linspace(z_min, z_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers

    # Compute variance in each bin
    binned_variances = []
    for i in range(num_bins):
        in_bin = (z_coords >= bins[i]) & (z_coords < bins[i + 1])
        if np.sum(in_bin) > 1:  # Ensure we have at least two points to compute variance
            variance = np.var((radial_distances[in_bin] - cylinder_radius))
        else:
            variance = np.nan  # Not enough points to compute variance
        binned_variances.append(variance)

    # Convert to numpy arrays for easier handling
    bin_centers = np.array(bin_centers)
    binned_variances = np.array(binned_variances)
    binned_stds = np.sqrt(binned_variances)
    binned_stds_rescale = binned_stds/8 # Convert units to m (8units ~ 1m)
    
    # Plot the variance as a function of z
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, binned_stds_rescale*1e6,#-1.3e3/np.sqrt(3), 
             marker='o', linestyle='-', color='g', label="Radial std")
    plt.plot(bin_centers, np.full(len(bin_centers),650), color='r', linestyle='--', label='Standard deviation for hex')
    plt.xlabel("Z Coordinate Along Cylinder Axis")
    plt.ylabel("Radial Standard Deviation (um)")
    plt.title("Radial Standard Deviation as a Function of Z")
    plt.legend()
    plt.grid(True)
    plt.show()

    return bin_centers, binned_stds_rescale

def compute_and_plot_binned_cylinder_center(geometry, cylinder_axis, cylinder_center, num_bins=20):
    """
    Computes and plots the estimated cylinder center (x, y) as a function of z.

    Parameters:
    - geometry: The input geometry (Open3D TriangleMesh or PointCloud object).
    - cylinder_axis: The unit vector along the cylinder's axis (numpy array).
    - cylinder_center: A point on the cylinder axis, defining its center (numpy array).
    - num_bins: The number of bins for z-coordinates.

    Returns:
    - bin_centers: The center z-values of the bins.
    - binned_x_centers: The estimated x-coordinates of the cylinder center per bin.
    - binned_y_centers: The estimated y-coordinates of the cylinder center per bin.
    """
    # Extract points from the input geometry
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        points = np.asarray(geometry.vertices)
    elif isinstance(geometry, o3d.geometry.PointCloud):
        points = np.asarray(geometry.points)
    else:
        raise TypeError("Input must be an Open3D TriangleMesh or PointCloud.")

    # Normalize the cylinder axis to ensure it's a unit vector
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)

    # Compute the vector from the cylinder center to each point
    vectors_to_points = points - cylinder_center

    # Compute the z-coordinates along the cylinder axis
    z_coords = np.dot(vectors_to_points, cylinder_axis)

    # Define bins for z-coordinates
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    bins = np.linspace(z_min, z_max, num_bins + 1)
    bin_z_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers

    # Compute the estimated cylinder center (x, y) in each bin
    binned_x_centers = []
    binned_y_centers = []

    for i in range(num_bins):
        in_bin = (z_coords >= bins[i]) & (z_coords < bins[i + 1])
        if np.sum(in_bin) > 0:  # Ensure we have points in the bin
            avg_x = np.mean(points[in_bin, 0])  # Compute mean x
            avg_y = np.mean(points[in_bin, 1])  # Compute mean y
        else:
            avg_x, avg_y = np.nan, np.nan  # Handle empty bins

        binned_x_centers.append(avg_x)
        binned_y_centers.append(avg_y)

    # Convert to numpy arrays
    bin_z_centers = np.array(bin_z_centers)
    binned_x_centers = np.array(binned_x_centers)
    binned_y_centers = np.array(binned_y_centers)

    # Plot x and y coordinates of the estimated cylinder center
    plt.figure(figsize=(8, 5))
    plt.plot(bin_z_centers, binned_x_centers-np.mean(binned_x_centers), marker='o', linestyle='-', color='r', label="Estimated X Center")
    plt.plot(bin_z_centers, binned_y_centers-np.mean(binned_y_centers), marker='s', linestyle='-', color='b', label="Estimated Y Center")
    # plt.plot(bin_centers, binned_x_centers-cylinder_center[0], marker='o', linestyle='-', color='r', label="Relative X Center")
    # plt.plot(bin_centers, binned_y_centers-cylinder_center[1], marker='s', linestyle='-', color='b', label="Relative Y Center")
    plt.xlabel("Z Coordinate Along Cylinder Axis")
    plt.ylabel("Estimated Cylinder Center (X, Y)")
    plt.title("Estimated Cylinder Center as a Function of Z")
    plt.legend()
    plt.grid(True)
    plt.show()

    return bin_z_centers, binned_x_centers, binned_y_centers


def compute_hexagonal_rod_variance(geometry, rod_axis, rod_center, rod_radius):
    """
    Computes the variance of the distances of points from the ideal hexagonal rod surface.

    Parameters:
    - geometry: The input geometry (Open3D TriangleMesh or PointCloud object).
    - rod_axis: The unit vector along the rod's axis (numpy array).
    - rod_center: A point on the rod axis, defining its center (numpy array).
    - rod_radius: The radius of the hexagonal rod (distance from the center to the edge of the hexagon).

    Returns:
    - variance: The variance of the distance errors from the ideal hexagonal rod surface.
    """
    # Extract points from the input geometry
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        points = np.asarray(geometry.vertices)
    elif isinstance(geometry, o3d.geometry.PointCloud):
        points = np.asarray(geometry.points)
    else:
        raise TypeError("Input must be an Open3D TriangleMesh or PointCloud.")

    # Normalize the rod axis to ensure it's a unit vector
    rod_axis = rod_axis / np.linalg.norm(rod_axis)

    # Compute the vector from the rod center to each point
    vectors_to_points = points - rod_center

    # Project these vectors onto the rod axis to get the axial component (z-component)
    axial_components = np.dot(vectors_to_points, rod_axis)[:, np.newaxis] * rod_axis

    # Compute the radial component (perpendicular to the rod axis)
    radial_components = vectors_to_points - axial_components

    # Now we need to compute the distance to the hexagonal surface in the plane orthogonal to the rod axis
    radial_distances = np.linalg.norm(radial_components, axis=1)

    # Project the points onto the plane orthogonal to the rod axis (this is the 2D hexagonal surface)
    points_2d = radial_components

    # Compute the angle of the radial points in polar coordinates
    angles = np.arctan2(points_2d[:, 1], points_2d[:, 0])

    # Normalize the angles to be between 0 and 2π
    angles = np.mod(angles, 2 * np.pi)

    # Compute the distance to the closest edge of a regular hexagon in 2D
    # The hexagon has 6 edges, so we need to check the distance to the nearest edge
    hex_edge_distances = np.minimum(np.abs(radial_distances - rod_radius), rod_radius)

    # Compute the squared error from the expected radius (ideal distance from the center to the edge)
    errors = (hex_edge_distances) ** 2

    # Compute variance
    variance = np.var(errors)

    return variance

def find_hexagonal_rod_axis_and_center_pcd(point_cloud):
    """
    Given a dense point cloud of a hexagonal rod, this function returns the axis direction, 
    center of the rod, and the angle of the rod's axis relative to the x-axis,
    as well as the fitted radius of the rod.
    
    Parameters:
    - point_cloud (open3d.geometry.PointCloud): The input point cloud.
    
    Returns:
    - tuple: (rod_axis, rod_center, rod_angle, rod_radius)
      - rod_axis (numpy.ndarray): The direction of the hexagonal rod's axis.
      - rod_center (numpy.ndarray): The center of the rod.
      - rod_angle (float): The angle of the rod's axis relative to the x-axis (in radians).
      - rod_radius (float): The fitted radius of the hexagonal rod.
    """
    # Extract points
    points = np.asarray(point_cloud.points)

    # Run PCA to find the main elongation axis (the rod axis)
    pca = PCA(n_components=2)
    pca.fit(points)

    # The first principal component is the rod axis direction
    rod_axis = pca.components_[0]

    # The rod center can still be approximated as the mean of the points
    rod_center = np.mean(points, axis=0)

    # Calculate the angle of the rod's axis relative to the x-axis (angle in radians)
    x_axis = np.array([1, 0, 0])  # Unit vector along the x-axis
    rod_angle = np.arccos(np.dot(rod_axis, x_axis) / (np.linalg.norm(rod_axis) * np.linalg.norm(x_axis)))

    # Compute radial distances from the center (project points to the plane orthogonal to the rod axis)
    vectors_to_points = points - rod_center
    axial_components = np.dot(vectors_to_points, rod_axis)[:, np.newaxis] * rod_axis
    radial_components = vectors_to_points - axial_components
    radial_distances = np.linalg.norm(radial_components, axis=1)

    # Fit the rod radius (we can take the mean or median of the radial distances)
    rod_radius = np.mean(radial_distances)

    # Optional visualization
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([rod_center, rod_center + 10 * rod_axis])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])

    o3d.visualization.draw_geometries([point_cloud, line_set])

    return rod_axis, rod_center, rod_angle, rod_radius

def compute_variance_binned_by_axis_distance(point_cloud, rod_axis, rod_center, num_bins=10):
    """
    Computes the variance of the radial distances from the rod's axis, binned by distance along the axis.

    Parameters:
    - point_cloud (open3d.geometry.PointCloud): The input point cloud of the hexagonal rod.
    - rod_axis (numpy.ndarray): The direction of the rod's axis.
    - rod_center (numpy.ndarray): The center of the rod.
    - num_bins (int): The number of bins to divide the distance along the rod axis.

    Returns:
    - binned_variance (list): List of variances for each bin.
    - bin_centers (numpy.ndarray): The center values of each bin (i.e., the distance along the axis).
    """
    # Extract the points from the point cloud
    points = np.asarray(point_cloud.points)

    # Compute the vector from the center to each point
    vectors_to_points = points - rod_center

    # Project the vectors onto the rod axis to get the distance along the rod axis
    axial_components = np.dot(vectors_to_points, rod_axis)[:, np.newaxis] * rod_axis
    distances_along_axis = np.linalg.norm(axial_components, axis=1)

    # Compute the radial components (perpendicular to the rod axis)
    radial_components = vectors_to_points - axial_components
    radial_distances = np.linalg.norm(radial_components, axis=1)

    # Bin the distances along the axis
    min_distance = np.min(distances_along_axis)
    max_distance = np.max(distances_along_axis)
    bin_edges = np.linspace(min_distance, max_distance, num_bins + 1)

    # Calculate the variance within each bin
    binned_variance = []
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Find the center of each bin

    for i in range(num_bins):
        # Find points within this bin
        bin_mask = (distances_along_axis >= bin_edges[i]) & (distances_along_axis < bin_edges[i + 1])
        bin_radial_distances = radial_distances[bin_mask]
        
        if len(bin_radial_distances) > 0:  # Avoid empty bins
            # Compute variance for points in this bin
            bin_variance = np.var(bin_radial_distances)
        else:
            bin_variance = np.nan  # If no points in the bin, set variance to NaN

        binned_variance.append(bin_variance)

    return binned_variance, bin_centers

''' hex plotting functions'''

def plot_xy_projection(point_cloud):
    """
    Plots the x and y components of a 3D point cloud in 2D, as if looking down the z-axis.

    Parameters:
    - point_cloud (open3d.geometry.PointCloud): The input point cloud.

    Returns:
    - None (displays a 2D scatter plot).
    """
    # Extract point coordinates
    points = np.asarray(point_cloud.points)

    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Create 2D scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=1, alpha=0.5)  # Adjust size and transparency for clarity
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Top-down View of Point Cloud (XY Projection)")
    plt.axis("equal")  # Keep aspect ratio equal for correct proportions
    plt.grid(True)

    # Show the plot
    plt.show()
def project_pcd_onto_plane(point_cloud, plane_normal, plane_point):
    """
    Projects a point cloud onto a plane defined by its normal and a point on the plane.

    Parameters:
    - point_cloud (o3d.geometry.PointCloud): The input 3D point cloud.
    - plane_normal (numpy.ndarray): The normal vector of the plane (should be a unit vector).
    - plane_point (numpy.ndarray): A point on the plane.

    Returns:
    - o3d.geometry.PointCloud: The projected 2D point cloud.
    """
    # Ensure normal is a unit vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Convert Open3D point cloud to NumPy array
    points = np.asarray(point_cloud.points)

    # Compute projection of each point onto the plane
    vectors_to_plane = points - plane_point  # Vector from plane_point to each point
    distances = np.dot(vectors_to_plane, plane_normal)  # Perpendicular distance to plane
    projected_points = points - np.outer(distances, plane_normal)  # Move points onto plane

    # Create new Open3D point cloud with projected points
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)

    return projected_pcd

def project_points_onto_plane(point_cloud, plane_normal, plane_point):
    """
    Projects a point cloud onto a plane and returns the projected 3D points as a NumPy array.
    """
    # Ensure normal is a unit vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Convert Open3D point cloud to NumPy array
    points = np.asarray(point_cloud.points)

    if points.size == 0:
        print("Warning: Input point cloud is empty.")
        return np.array([])  # Return an empty array to avoid indexing errors

    # Compute projection of each point onto the plane
    vectors_to_plane = points - plane_point
    distances = np.dot(vectors_to_plane, plane_normal)
    projected_points = points - np.outer(distances, plane_normal)

    return np.array(projected_points)  # Ensure it's a NumPy array

def hexagon_vertices_2D(center, radius, angle=0):
    n = 6  # A hexagon has 6 sides
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle  # Angles for each vertex
    vertices = np.column_stack((center[0] + radius * np.cos(angles), 
                                center[1] + radius * np.sin(angles)))  # Calculate (x, y) coordinates
    return vertices


# Function to compute the perpendicular distance from a point to a line segment
def point_to_segment_distance(p, v1, v2):
    # Vector from v1 to v2
    line_vec = v2 - v1
    # Vector from v1 to point p
    point_vec = p - v1
    # Project point_vec onto line_vec to get the projection scalar
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:  # avoid division by zero, in case v1 == v2 (shouldn't happen for hexagon)
        return np.linalg.norm(p - v1)
    proj = np.dot(point_vec, line_vec) / line_len_sq
    # Clamping the projection to the segment
    proj = np.clip(proj, 0, 1)
    # Find the closest point on the line segment
    closest_point = v1 + proj * line_vec
    # Return the distance from p to the closest point
    return np.linalg.norm(p - closest_point)

# Function to compute the minimum distance from a point to the hexagon edges
def distance_to_hexagon_edges(point, hexagon_vertices):
    min_dist = float('inf')  # Start with an infinitely large distance
    n = len(hexagon_vertices)
    # Check distances to all edges (line segments between consecutive vertices)
    for i in range(n):
        # Vertices of the current edge (v1, v2)
        v1 = hexagon_vertices[i]
        v2 = hexagon_vertices[(i + 1) % n]  # Next vertex, wrap around with modulus
        # Calculate the distance from the point to this edge
        dist = point_to_segment_distance(point, v1, v2)
        min_dist = min(min_dist, dist)
    return min_dist

def hexagon_vertices(center, radius, angle, axis_direction):
    """
    Generate the 3D vertices of a hexagonal rod.
    
    Parameters:
    - center: The center of the hexagonal rod (x, y, z).
    - radius: The radius of the hexagon.
    - angle: The rotation angle of the hexagon (around the axis direction).
    - axis_direction: A 3D vector (x, y, z) defining the axis of the hexagonal rod.
    
    Returns:
    - A list of 12 vertices (2 sets of 6 for top and bottom faces).
    """
    # Define the 6 vertices of a regular hexagon in the XY-plane (radius is the distance from the center)
    hexagon_2d = np.array([
        [radius * np.cos(np.pi / 3 * i), radius * np.sin(np.pi / 3 * i)] for i in range(6)
    ])

    # Normalize the axis direction to ensure it's a unit vector
    axis_direction = axis_direction / np.linalg.norm(axis_direction)

    # Create a rotation matrix based on the axis direction and angle
    # We will use the rotation matrix for rotating points in 3D space.
    def rotation_matrix(axis, angle):
        """
        Create a rotation matrix for rotating points by 'angle' around the given 'axis'.
        """
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        ux, uy, uz = axis
        
        return np.array([
            [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
            [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
            [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
        ])

    # Rotation matrix for rotating the hexagon points
    rot_matrix = rotation_matrix(axis_direction, angle)

    # Rotate and translate the 6 vertices
    hexagon_3d = []
    for point in hexagon_2d:
        # 3D coordinates (x, y, z) where z = 0 (since it's in the XY-plane)
        point_3d = np.append(point, 0)  # (x, y, 0)
        # Apply rotation
        rotated_point = np.dot(rot_matrix, point_3d)
        hexagon_3d.append(rotated_point)

    # Now we need to add the top and bottom face vertices (offset by the rod's height)
    # Assuming the center is at the midpoint between the top and bottom faces
    height = 10  # Assume unit height, this can be set to a different value if needed
    bottom_face = np.array(hexagon_3d) - axis_direction * height / 2
    top_face = np.array(hexagon_3d) + axis_direction * height / 2

    # Combine the vertices into a single list (12 vertices)
    hexagon_vertices = np.vstack((bottom_face, top_face))
    
    # Translate the vertices to the given center point
    hexagon_vertices += center

    return hexagon_vertices

def compute_normal_from_corners(corners):
    """
    Compute the normal vector of a plane defined by four corners.
    """
    P1, P2, P3, P4 = corners

    v1 = P3 - P1  # Change order
    v2 = P4 - P1  # Use P4 instead of P2

    print("v1:", v1)
    print("v2:", v2)

    normal = np.cross(v1, v2)
    print("Cross product:", normal)

    norm_val = np.linalg.norm(normal)
    if norm_val == 0:
        print("Warning: Zero normal vector!")
        return np.array([0, 0, 0])  # Avoid division by zero

    normal = normal / norm_val
    return normal
def point_to_plane_distance(point, plane_normal, plane_point):
    """
    Calculate the perpendicular distance from a point to a plane.

    Args:
    - point (np.array): The coordinates of the point (x0, y0, z0).
    - plane_normal (np.array): The normal vector to the plane (A, B, C).
    - plane_point (np.array): A point on the plane (x1, y1, z1).

    Returns:
    - float: The perpendicular distance from the point to the plane.
    """
    # Vector from the point on the plane to the point
    point_to_plane_vec = point - plane_point
    
    # The perpendicular distance is the projection of point_to_plane_vec onto the normal vector
    distance = np.abs(np.dot(point_to_plane_vec, plane_normal)) / np.linalg.norm(plane_normal)
    # print(f"The distance from the point to the plane is: {distance}")
    return distance