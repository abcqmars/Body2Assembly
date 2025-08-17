import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import bezier
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d

def compute_hodograph(nodes):
    degree = nodes.shape[1] - 1
    deriv_nodes = degree * (nodes[:, 1:] - nodes[:, :-1])
    return bezier.Curve(deriv_nodes, degree=degree - 1)

def frenet_frame(curve, t):
    # Evaluate point on the curve
    r_t = curve.evaluate(t).flatten()

    # First derivative (velocity)
    hodograph1 = compute_hodograph(curve.nodes)
    r_prime = hodograph1.evaluate(t).flatten()

    # Second derivative (acceleration)
    hodograph2 = compute_hodograph(hodograph1.nodes)
    r_double_prime = hodograph2.evaluate(t).flatten()

    # Tangent
    T = r_prime / np.linalg.norm(r_prime)

    # Normal (orthogonal component of second derivative)
    # T_dot = np.dot(r_prime, r_double_prime)
    # speed = np.linalg.norm(r_prime)

    T_prime = r_double_prime - np.dot(r_double_prime, T) * T
    # import pdb
    # pdb.set_trace()
    #(r_double_prime * speed**2 - r_prime * T_dot) / speed**4
    N = T_prime / np.linalg.norm(T_prime)

    # Binormal (only meaningful in 3D)
    if r_t.shape[0] == 3:
        B = np.cross(T, N)
        return T, N, B
    else:
        return T, N, None


def construct_cross_section(center, N, B, radius=0.2, num_points=100):

    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = np.array([
        center + radius * np.cos(t) * N + radius * np.sin(t) * B
        for t in theta
    ])
    return circle_points

def create_bezier_curve(nodes):
    # Create a Bézier curve from the given nodes
    curve = bezier.Curve(nodes, degree=nodes.shape[1] - 1)
    return curve

def create_cylinder_mesh_from_bezierCurve(curve, radius=0.2, num_points=10, num_segments=10):
    t_values = np.linspace(0, 1, num_segments)
    vertices = []
    faces = []

    # Generate cross-sections along the curve
    lastN = None
    for i, t in enumerate(t_values):
        center = curve.evaluate(t).flatten()
        T, N, B = frenet_frame(curve, t)

        # Ensure the normal vector N is tranitintg smoothly (Bug still exists)
        if lastN is None:
            lastN = N
        else:
            if np.dot(N, lastN) < 0:
                N = -N
                B = -B
            lastN = N

        circle = construct_cross_section(center, N, B, radius, num_points)
        vertices.append(circle)

    vertices = np.array(vertices)

    # Create faces by connecting adjacent cross-sections
    for i in range(num_segments - 1):
        for j in range(num_points - 1):
            # First triangle
            faces.append([
                (i + 1) * num_points + j + 1,
                (i + 1) * num_points + j,
                i * num_points + j,
            ])
            # Second triangle
            faces.append([
                i * num_points + j + 1,
                (i + 1) * num_points + j + 1,
                i * num_points + j,
            ])
        # Close the circle with triangles
        faces.append([
            (i + 1) * num_points,
            (i + 1) * num_points + (num_points - 1),
            i * num_points + (num_points - 1),
        ])
        faces.append([
            i * num_points,
            (i + 1) * num_points,
            i * num_points + (num_points - 1),
        ])

    # Close the cylinder.
    for j in range(num_points - 2): 
        faces.append([
            (num_segments - 1) * num_points,
            (num_segments - 1) * num_points + j + 1,
            (num_segments - 1) * num_points + j + 2,
        ])

        faces.append([
            j + 2,
            j + 1,
            0 
        ])

    # Convert vertices and faces to Open3D format
    vertices_flat = vertices.reshape(-1, 3)
    faces_flat = np.array(faces).reshape(-1, 3)
    # import pdb
    # pdb.set_trace()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_flat)
    mesh.triangles = o3d.utility.Vector3iVector(faces_flat)
    return mesh

def create_cylinder_mesh_from_2endPoints(p1, p2, radius=0.2, num_points=10, num_segments=10):
    t_values = np.linspace(0, 1, num_segments)
    vertices = []
    faces = []

    # Generate cross-sections along the curve
    N = np.array([1, 0, 0])
    B = np.array([0, 0, 1])
    for i, t in enumerate(t_values):
        center = p1 * (1 - t) + p2 * t
        circle = construct_cross_section(center, N, B, radius, num_points)
        vertices.append(circle)

    vertices = np.array(vertices)

    # Create faces by connecting adjacent cross-sections
    for i in range(num_segments - 1):
        for j in range(num_points - 1):
            # First triangle
            faces.append([
                (i + 1) * num_points + j + 1,
                (i + 1) * num_points + j,
                i * num_points + j,
            ])
            # Second triangle
            faces.append([
                i * num_points + j + 1,
                (i + 1) * num_points + j + 1,
                i * num_points + j,
            ])
        # Close the circle with triangles
        faces.append([
            (i + 1) * num_points,
            (i + 1) * num_points + (num_points - 1),
            i * num_points + (num_points - 1),
        ])
        faces.append([
            i * num_points,
            (i + 1) * num_points,
            i * num_points + (num_points - 1),
        ])

    # Close the cylinder.
    for j in range(num_points - 2): 
        faces.append([
            (num_segments - 1) * num_points,
            (num_segments - 1) * num_points + j + 1,
            (num_segments - 1) * num_points + j + 2,
        ])

        faces.append([
            j + 2,
            j + 1,
            0 
        ])

    # Convert vertices and faces to Open3D format
    vertices_flat = vertices.reshape(-1, 3)
    faces_flat = np.array(faces).reshape(-1, 3)
    # import pdb
    # pdb.set_trace()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_flat)
    mesh.triangles = o3d.utility.Vector3iVector(faces_flat)
    return mesh


if __name__ == "__main__":
    # Define control points for the Bézier curve
    control_points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.5],
        [2.0, 0.0, 1.0],
        [3.0, 1.0, 1.5]
    ]).T  # Transpose to match bezier library's format

    # Create the Bézier curve
    curve = create_bezier_curve(control_points)

    # Generate the cylinder mesh along the curve
    cylinder_mesh = create_cylinder_mesh_from_bezierCurve(curve, radius=0.1, num_points=10, num_segments=10)

    # Save the mesh to a file
    o3d.io.write_triangle_mesh("cylinder_mesh.obj", cylinder_mesh)

    print("Cylinder mesh saved to 'cylinder_mesh.ply'")