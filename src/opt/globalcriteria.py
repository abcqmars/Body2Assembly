import numpy as np
# Criteras for global optimization:
def parts_penetrion(part1Verts, part2Verts, threshold=0.03):
    """
    Calculate the distance between the vertices on two parts.
    """
    # Verify inputs:
    assert len(part1Verts.shape)==2 and part1Verts.shape[1] == 3, "part1Verts should be of shape (N, 3)"
    assert len(part2Verts.shape)==2 and part2Verts.shape[1] == 3, "part2Verts should be of shape (N, 3)"

    # Criterion:
    penalty = 0
    distMatrix = np.linalg.norm(part1Verts[:, np.newaxis, :] - part2Verts[np.newaxis, :, :], axis=2) # N x N
    penalty += max(0, threshold- np.min(distMatrix, axis=1).min())
    penalty += max(0, threshold- np.min(distMatrix, axis=0).min())
    return penalty

def parts_overlap(parts1Verts, part2Verts, bodyPart1Verts, bodyPart2Verts, bodyPart1Area, bodyPart2Area):
    """
    Calculate the projected overlapping area on body.
    """
    assert len(parts1Verts.shape)==2 and parts1Verts.shape[1] == 3, "part1Verts should be of shape (N, 3)"
    assert len(part2Verts.shape)==2 and part2Verts.shape[1] == 3, "part2Verts should be of shape (N, 3)"
    
    # Compute distance matrices
    dist_parts1_body1 = np.linalg.norm(parts1Verts[:, np.newaxis, :] - bodyPart1Verts[np.newaxis, :, :], axis=2)
    dist_parts1_body2 = np.linalg.norm(parts1Verts[:, np.newaxis, :] - bodyPart2Verts[np.newaxis, :, :], axis=2)
    dist_parts2_body1 = np.linalg.norm(part2Verts[:, np.newaxis, :] - bodyPart1Verts[np.newaxis, :, :], axis=2)
    dist_parts2_body2 = np.linalg.norm(part2Verts[:, np.newaxis, :] - bodyPart2Verts[np.newaxis, :, :], axis=2) 

    # For parts1Verts: find closest in body1 and body2
    min_dist_body1 = np.min(dist_parts1_body1, axis=1) 
    min_dist_body2 = np.min(dist_parts1_body2, axis=1)
    idx_body2 = np.argmin(dist_parts1_body2, axis=1) 
    overlap_area = np.sum(bodyPart2Area[idx_body2][min_dist_body2 < min_dist_body1])

    # For part2Verts: find closest in body1 and body2
    min_dist_body1_2 = np.min(dist_parts2_body1, axis=1)
    min_dist_body2_2 = np.min(dist_parts2_body2, axis=1) 
    idx_body1_2 = np.argmin(dist_parts2_body1, axis=1) 
    overlap_area += np.sum(bodyPart1Area[idx_body1_2][min_dist_body1_2 < min_dist_body2_2])

    return overlap_area



def parts_smoothness_penalty(part1Edges, part2Edges):
    """
    Calculate the smoothness by measuring distance between edge vertices between parts.
    """
    # Verify inputs:
    assert isinstance(part1Edges, list), "part1Edges should be a list of edges"
    assert part1Edges[0].shape[1] == 3, "part1Edges should be a list of edges vertices with shape (E, 3)"
    
    # Criterion:
    mindist = float("inf")
    for edge1Verts in part1Edges:
        for edge2Verts in part2Edges:
            distMat = np.linalg.norm(edge1Verts[:, np.newaxis, :] - edge2Verts[np.newaxis, :, :], axis=2) # E x E
            dist = min(np.min(distMat, axis=0).mean(), np.min(distMat, axis=1).mean()) 
            if dist < mindist:
                mindist = dist
    return mindist
