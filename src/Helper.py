#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from itertools import combinations

# -----------------
colors_map = np.array([
    [0.894, 0.102, 0.110],  # red
    [0.216, 0.494, 0.722],  # blue
    [0.302, 0.686, 0.290],  # green
    [0.596, 0.306, 0.639],  # purple
    [1.000, 0.498, 0.000],  # orange
    [1.000, 1.000, 0.200],  # yellow
    [0.651, 0.337, 0.157],  # brown
    [0.969, 0.506, 0.749],  # pink
    [0.600, 0.600, 0.600],  # gray
    [0.121, 0.470, 0.705],  # blue2
    [0.682, 0.780, 0.909],  # light blue
    [0.199, 0.627, 0.172],  # green2
    [0.992, 0.706, 0.384],  # light orange
    [0.894, 0.102, 0.110],  # red2
    [0.792, 0.698, 0.839],  # light purple
    [0.839, 0.152, 0.156],  # dark red
    [0.000, 0.588, 0.533],  # indigo
    [0.000, 0.000, 0.502],  # sienna
    [0.870, 0.796, 0.894],  # pale violet
    [0.498, 0.600, 0.600],  # slate gray
    [0.737, 0.741, 0.133],  # olive
    [0.090, 0.745, 0.811],  # cyan
    [0.960, 0.480, 0.286],  # coral
    [0.580, 0.580, 0.580],  # mid gray
    [0.7, 0.7, 0.7]
])

def is_segment_intersecting_triangle(p1, p2, tri):
    v0, v1, v2 = tri

    seg_dir = p2 - p1
    seg_len = np.linalg.norm(seg_dir)
    seg_dir /= seg_len

    epsilon = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(seg_dir, edge2)
    a = np.dot(edge1, h)
    if -epsilon < a < epsilon:
        return False

    f = 1.0 / a
    s = p1 - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(seg_dir, q)
    if v < 0.0 or u + v > 1.0:
        return False

    t = f * np.dot(edge2, q)
    if t < 0.0 or t > seg_len:
        return False

    return True

def rotation_matrix_to_euler_angles(R):
    theta = np.arcsin(-R[2, 0])
    psi = np.arctan2(R[1, 0], R[0, 0])
    phi = np.arctan2(R[2, 1], R[2, 2])
    
    return phi, theta, psi  # Roll, Pitch, Yaw


def rotation_matrix_to_axis_angle(R):
    """
    Convert batch of rotation matrices to axis-angle vectors.

    Args:
        R (torch.Tensor): shape (N, 3, 3)
    Returns:
        rot_vecs (torch.Tensor): shape (N, 3)
    """
    batch_size = R.shape[0]
    cos_theta = 0.5 * (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1)
    theta = torch.acos(torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6))  # avoid nan

    # Compute rotation axis from skew-symmetric part
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    r = torch.stack([rx, ry, rz], dim=1)

    sin_theta = torch.norm(r, dim=1, keepdim=True) / 2
    axis = r / (2 * sin_theta + 1e-8)

    rot_vecs = axis * theta.unsqueeze(1)
    return rot_vecs


def batch_rodrigues(
    rot_vecs,
    epsilon: float = 1e-8,
):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat



def euler_angles_to_rotation_matrix(angles):
    phi = angles[0]
    theta = angles[1]
    psi = angles[2]
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R



def is_coplanar(p1, p2, p3, p4, tol=1e-6):
    """Check if four points are coplanar using determinant method."""
    matrix = np.array([p2 - p1, p3 - p1, p4 - p1])
    volume = np.linalg.det(matrix)
    return abs(volume) < tol 

def get_cuboid_faces(vertices):
    """
    Given eight unordered vertices forming a bounding box, compute the 6 faces (represnted in indices ordered in CCW) of the cuboid.
    """
    assert isinstance(vertices, np.ndarray), "Ensure the vertices is numpy.ndarray."
    assert vertices.ndim==2 and vertices.shape[1]==3, "Ensure the vertices is of 3D."
    assert len(vertices)==8, "Ensure 8 vertices are given."

    # Compute centroid
    centroid = np.mean(vertices, axis=0)
    
    # Find the 6 faces by identifying 4 vertices that lie on the same plane
    faces = []
    normals = []
    face_candidates = []
    
    for comb in combinations(range(8), 4):  # All possible 4-vertex groups
        subset = vertices[list(comb)]

        # Ensure four vertices are coplanar
        if not is_coplanar(*subset):
            continue  # Skip non-coplanar sets
        
        # Compute normal using cross product and normalize
        v1, v2, v3 = subset[:3]  # Pick first 3 points
        normal = np.cross(v2 - v1, v3 - v1)
        normal /= np.linalg.norm(normal)  # Normalize

        # Check if the face normal is pointing outward
        center_of_face = np.mean(subset, axis=0)
        if np.dot(normal, center_of_face - centroid) < 0:
            normal = -normal  # Flip normal if inward

        # Ensure the order of the vertices
        ref_vector = subset[0] - subset.mean(axis=0)
        angles = np.arctan2(np.cross(ref_vector, subset - subset.mean(axis=0)) @ normal, (subset - subset.mean(axis=0)) @ ref_vector)
        sorted_indices = np.argsort(angles)
        ordered_face = [comb[i] for i in sorted_indices]

        # Store face candidate for parallelism check
        face_candidates.append((ordered_face, normal))

    # Add only faces that have a parallel counterpart
    used_faces = set()
    for i, (face1, normal1) in enumerate(face_candidates):
        for j, (face2, normal2) in enumerate(face_candidates):
            if i != j and (np.allclose(normal1, normal2) or np.allclose(normal1, -normal2)):
                ordered_face1 = list(face1)
                ordered_face2 = list(face2)

                if frozenset(ordered_face1) not in used_faces:
                    faces.append(ordered_face1)
                    used_faces.add(frozenset(ordered_face1))
                
                if frozenset(ordered_face2) not in used_faces:
                    faces.append(ordered_face2)
                    used_faces.add(frozenset(ordered_face2))

                if len(faces) == 6:
                    return faces

    return faces

def get_rectangle_frame(tgtVerts):
    """
    Frame: the orientation and the scales along the orientation.
    """
    c2 = tgtVerts - tgtVerts.mean(axis=0)
    a1 = (c2[1] - c2[0])
    s1 = np.linalg.norm(a1)
    a1 = a1/s1
    a2 = (c2[3] - c2[0])
    s2 = np.linalg.norm(a2)
    a2 = a2/s2
    a3 = np.cross(a1, a2)
    s3 = 1
    a3 = a3/np.linalg.norm(a3)
    c2Axes = np.array([a1, a2, a3])
    c2Scales = [s1, s2, s3]
    return c2Axes, c2Scales

def scales_to_scale_matrix(scales):
    assert len(scales)==3
    return np.array([
            [scales[0], 0, 0],
            [0, scales[1], 0],
            [0, 0, scales[2]]])

def pointLineDist(p, segment):
    """
    Computes the shortest distance from point p to an infinite line defined by two points.
    """
    a, b = segment
    ab = b - a
    ap = p - a
    ab_norm = np.linalg.norm(ab)
    if ab_norm < 1e-8:
        return np.linalg.norm(ap)
    # Project ap onto ab
    proj = np.dot(ap, ab) / ab_norm
    closest = a + proj * ab / ab_norm
    dist = np.linalg.norm(p - closest)
    return dist

def angles2Rot(rotAngles):
    rotX = np.array([
        [1, 0, 0],
        [0, np.cos(rotAngles[0]), -np.sin(rotAngles[0])],
        [0, np.sin(rotAngles[0]), np.cos(rotAngles[0])]
    ])
    rotY = np.array([
        [np.cos(rotAngles[1]), 0, np.sin(rotAngles[1])],
        [0, 1, 0],
        [-np.sin(rotAngles[1]), 0, np.cos(rotAngles[1])]
    ])
    rotZ = np.array([
        [np.cos(rotAngles[2]), -np.sin(rotAngles[2]), 0],
        [np.sin(rotAngles[2]), np.cos(rotAngles[2]), 0],
        [0, 0, 1]
    ])
    return rotX @ rotY @ rotZ