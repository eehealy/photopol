import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import functions as fun
import geoms
from scipy.spatial import distance

import scipy.optimize as opt
from scipy.spatial import ConvexHull

from scipy.optimize import least_squares
from sklearn.linear_model import RANSACRegressor

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

import importlib 
importlib.reload(fun)
importlib.reload(geoms)

def transform_to_new_coord_system(points, cylinder_axis_pcd, cylinder_center_pcd):
    # Step 1: Translate points to move the center of the rod to the origin
    translated_points = points - cylinder_center_pcd
    
    # Step 2: Normalize the cylinder axis (desired z-axis)
    cylinder_axis_pcd = cylinder_axis_pcd / np.linalg.norm(cylinder_axis_pcd)

    # Step 3: Compute the new x-axis (this should align with the original x-axis)
    original_x_axis = np.array([1, 0, 0])  # Assuming the original x-axis is [1, 0, 0]
    # Ensure that the x-axis is not parallel to the cylinder axis
    if np.abs(np.dot(cylinder_axis_pcd, original_x_axis)) > 0.999:
        raise ValueError("The original x-axis is nearly parallel to the cylinder axis.")
    
    # Compute the new y-axis by taking the cross product of the cylinder axis and original x-axis
    new_x_axis = original_x_axis
    new_y_axis = np.cross(cylinder_axis_pcd, new_x_axis)
    new_y_axis = new_y_axis / np.linalg.norm(new_y_axis)  # Normalize y-axis

    # Compute the new z-axis (which is just the cylinder axis)
    new_z_axis = cylinder_axis_pcd
    
    # Step 4: Construct the rotation matrix
    # The rotation matrix columns should be the new coordinate axes
    rotation_matrix = np.column_stack([new_x_axis, new_y_axis, new_z_axis])

    # Step 5: Rotate the translated points to the new coordinate system
    transformed_points = (rotation_matrix.T @ translated_points.T).T  # Apply rotation

    return transformed_points


def compute_corners(vertices):
    """
    Computes the corners from a set of vertices.
    
    Parameters:
    - vertices: A numpy array of shape (12, 3) or a list of 12 vertices, each with 3 coordinates.
    
    Returns:
    - corners: A numpy array of shape (6, 4, 3) representing 6 corners with 4 vertices each.
    """
    corners = np.empty((6, 4, 3))
    
    for i in range(6):
        corner1 = vertices[i % 6]
        corner2 = vertices[(i + 1) % 6]
        corner3 = vertices[i + 6]
        corner4 = vertices[((i + 7) - 6) % 6 + 6]
        corners[i] = [corner1, corner2, corner3, corner4]
    
    return corners

def compute_normal_from_corners(corners):
    """
    Compute the normal vector of a plane defined by four corners.
    """
    P1, P2, P3, P4 = corners

    v1 = P3 - P1  # Change order
    v2 = P4 - P1  # Use P4 instead of P2

    # print("v1:", v1)
    # print("v2:", v2)

    normal = np.cross(v1, v2)
    # print("Cross product:", normal)

    norm_val = np.linalg.norm(normal)
    if norm_val == 0:
        print("Warning: Zero normal vector!")
        return np.array([0, 0, 0])  # Avoid division by zero

    normal = normal / norm_val
    return normal


def point_to_plane_distance(point, normal, plane_point):
    """
    Computes the signed distance from a point to a plane.
    
    Parameters:
    - point: A numpy array representing the point (x, y, z).
    - normal: A numpy array representing the normal vector of the plane.
    - plane_point: A numpy array representing a point on the plane.
    
    Returns:
    - distance: The signed distance from the point to the plane.
    """
    # Vector from the point on the plane to the point of interest
    vector = point - plane_point
    # Dot product to get the signed distance
    distance = np.dot(vector, normal)
    return distance

def compute_min_distances(points, normals, vertices):
    """
    Computes the minimum signed distances from points to planes defined by
    vertices and normals. It returns the signed minimum distance while considering 
    the absolute minimum distance for selection.
    
    Parameters:
    - points: A numpy array of shape (N, 3) representing the points (x, y, z).
    - normals: A numpy array of shape (6, 3) representing the normals of the planes.
    - vertices: A numpy array of shape (6, 3) representing points on the planes.
    
    Returns:
    - min_dists: A list of the minimum signed distances for each point, 
                 considering the minimum absolute distance.
    """
    min_dists = []
    
    # Loop through all points
    for j in range(len(points)):
        distances = np.empty((len(normals)))
        
        # Loop through all planes (vertices and normals define the planes)
        for i in range(len(normals)):
            plane_point = vertices[i]
            normal = normals[i]
            distance = point_to_plane_distance(points[j], normal, plane_point)
            distances[i] = distance
        
        # Find the minimum absolute distance and its corresponding signed distance
        min_abs_dist = np.min(np.abs(distances))  # Find minimum absolute distance
        min_signed_dist = distances[np.argmin(np.abs(distances))]  # Get the signed value
        
        # Append the signed distance to min_dists
        min_dists.append(min_signed_dist)
    
    return min_dists

# def hexagon_vertices(center, radius, angle, axis_direction):
#     """
#     Generate the 3D vertices of a hexagonal rod.
    
#     Parameters:
#     - center: The center of the hexagonal rod (x, y, z).
#     - radius: The radius of the hexagon.
#     - angle: The rotation angle of the hexagon (around the axis direction).
#     - axis_direction: A 3D vector (x, y, z) defining the axis of the hexagonal rod.
    
#     Returns:
#     - A list of 12 vertices (2 sets of 6 for top and bottom faces).
#     """
#     # Define the 6 vertices of a regular hexagon in the XY-plane (radius is the distance from the center)
#     hexagon_2d = np.array([
#         [radius * np.cos(np.pi / 3 * i), radius * np.sin(np.pi / 3 * i)] for i in range(6)
#     ])

#     # Normalize the axis direction to ensure it's a unit vector
#     axis_direction = axis_direction / np.linalg.norm(axis_direction)

#     # Create a rotation matrix based on the axis direction and angle
#     # We will use the rotation matrix for rotating points in 3D space.
#     def rotation_matrix(axis, angle):
#         """
#         Create a rotation matrix for rotating points by 'angle' around the given 'axis'.
#         """
#         axis = axis / np.linalg.norm(axis)
#         cos_theta = np.cos(angle)
#         sin_theta = np.sin(angle)
#         ux, uy, uz = axis
        
#         return np.array([
#             [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
#             [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
#             [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
#         ])

#     # Rotation matrix for rotating the hexagon points
#     rot_matrix = rotation_matrix(axis_direction, angle)

#     # Rotate and translate the 6 vertices
#     hexagon_3d = []
#     for point in hexagon_2d:
#         # 3D coordinates (x, y, z) where z = 0 (since it's in the XY-plane)
#         point_3d = np.append(point, 0)  # (x, y, 0)
#         # Apply rotation
#         rotated_point = np.dot(rot_matrix, point_3d)
#         hexagon_3d.append(rotated_point)

#     # Now we need to add the top and bottom face vertices (offset by the rod's height)
#     # Assuming the center is at the midpoint between the top and bottom faces
#     height = 10  # Assume unit height, this can be set to a different value if needed
#     bottom_face = np.array(hexagon_3d) - axis_direction * height / 2
#     top_face = np.array(hexagon_3d) + axis_direction * height / 2

#     # Combine the vertices into a single list (12 vertices)
#     hexagon_vertices = np.vstack((bottom_face, top_face))
    
#     # Translate the vertices to the given center point
#     hexagon_vertices += center


def hexagon_vertices(center, radius, angle, axis_direction, height=10):
    """
    Generate the 3D vertices of a hexagonal rod.
    
    Parameters:
    - center: The center of the hexagonal rod (x, y, z).
    - radius: The radius of the hexagon.
    - angle: The rotation angle of the hexagon (around the axis direction).
    - axis_direction: A 3D vector (x, y, z) defining the axis of the hexagonal rod.
    - height: The height of the hexagonal rod (distance between top and bottom faces).
    
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
    bottom_face = np.array(hexagon_3d) - axis_direction * height / 2
    top_face = np.array(hexagon_3d) + axis_direction * height / 2

    # Combine the vertices into a single list (12 vertices)
    hexagon_vertices = np.vstack((bottom_face, top_face))
    
    # Translate the vertices to the given center point
    hexagon_vertices += center
    
    return hexagon_vertices


def points_to_axis(ply_file, cylinder_center, cylinder_radius, cylinder_height, visualization=True):
    pcd = o3d.io.read_point_cloud(ply_file)
    pcd_cyl = fun.filter_points_in_cylinder(pcd, np.array([0, 0, 1]), cylinder_center, cylinder_radius, 
                                            cylinder_height, visualization=visualization)
    points = np.asarray(pcd_cyl.points)
    hex_axis, hex_center = fun.find_cylinder_axis_and_center_pcd(pcd_cyl, visualization=visualization)

    return points, hex_axis, hex_center


def plot_room_plumbbob_projections(points, transformed_points):
    x = points[:, 0]
    y = points[:, 1]
    
    x2 = transformed_points[:, 0]
    y2 = transformed_points[:, 1]
    
    # Create a figure with two subplots in one row (1 row, 2 columns)
    plt.figure(figsize=(10, 5))  # Set the figure size if needed
    
    # First subplot (top-down view)
    plt.subplot(1, 2, 1)  # (rows, columns, index)
    plt.scatter(x, y, s=0.1)
    plt.axis("equal")
    plt.title("Room coordinate system")
    plt.xlabel("X (mm)")  # Customize as needed
    plt.ylabel("Y (mm)")  # Customize as needed
    
    # Second subplot (fill in with other plot details)
    plt.subplot(1, 2, 2)  # (rows, columns, index)
    plt.scatter(x2, y2, s=0.1, color='green', )  # Replace with your other plot details

    
    plt.axis('equal')
    plt.title("Plumbbob coordinate system")
    plt.xlabel("X (mm)")  # Customize as needed
    plt.ylabel("Y (mm)")  # Customize as needed
    # plt.legend()
    # Show the plots
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def hexagon_generation_parameters(radius, angle, center, perturbed_axis):
    """
    Generate hexagonal parameters: vertices, corners, and normals with perturbed axis.

    Parameters:
    - radius: The radius of the hexagon.
    - angle: Rotation angle to apply to the hexagon (in radians).
    - center: The center of the hexagon (x, y, z) coordinates.
    - perturbed_axis: The perturbation applied to the axis of rotation (e.g., a vector).

    Returns:
    - vertices: List of the hexagon's vertices in 3D space.
    - corners: Coordinates of the corners of the hexagon.
    - normals: Normal vectors at each vertex, considering the perturbed axis.
    """
    # Create hexagon vertices in 2D space
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 vertices for hexagon
    hexagon_vertices_2d = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])

    # Apply the rotation based on the given angle
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    # Rotate the hexagon vertices and then translate to the center
    hexagon_vertices_3d = []
    for v in hexagon_vertices_2d:
        rotated_vertex = np.dot(rotation_matrix, np.array([v[0], v[1], 0]))
        hexagon_vertices_3d.append(rotated_vertex + np.array(center))

    # Perturbing the axis of the hexagon by the given perturbed_axis
    perturbed_normals = []
    for v in hexagon_vertices_3d:
        # Perturb the normal vector (rotation axis) based on the perturbed axis
        normal = np.array([0, 0, 1])  # Initial normal vector
        perturbed_normal = np.dot(rotation_matrix, normal)  # Apply rotation
        perturbed_normal += perturbed_axis  # Apply the perturbation
        perturbed_normals.append(perturbed_normal)

    # Output the results
    vertices = np.array(hexagon_vertices_3d)
    corners = hexagon_vertices_3d  # Could be modified if needed
    normals = np.array(perturbed_normals)

    return vertices, corners, normals
