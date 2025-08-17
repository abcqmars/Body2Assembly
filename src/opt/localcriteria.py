import numpy as np
# Criteras for local optimization:
def part_seg_dist(partVerts, bodySegVerts, bodySegForces):
    """
    Calculate the distance between the vertices on body segment and furniturePart.
    """
    # Verify inputs:
    assert len(partVerts.shape)==2 and partVerts.shape[1] == 3, "partVerts should be of shape (N, 3)"
    assert len(bodySegVerts.shape)==2 and bodySegVerts.shape[1] == 3, "bodyPartVerts should be of shape (B, 3)"
    assert len(bodySegForces.shape)==1, "bodyPartForces should be of shape (B,)"

    # Criterion:
    distanceMatrix = np.linalg.norm(partVerts[:, np.newaxis, :] - bodySegVerts[np.newaxis, :, :], axis=2)# N x B
    cloestPointsIndices = np.argmin(distanceMatrix, axis=1) # N x 1
    weights = bodySegForces[cloestPointsIndices] # N x 1
    return (distanceMatrix.min(axis=1) * weights).mean(axis=0)

def penetration_penalty(partVerts, bodySegVerts, bodySegNormals, K =1):
    """
    Calculate the penetration by projecting the part vertices to the normal of cloest body segment vertices.
    """
    # Verify inputs:
    assert len(partVerts.shape)==2 and partVerts.shape[1] == 3, "partVerts should be of shape (N, 3)"   
    assert len(bodySegVerts.shape)==2 and bodySegVerts.shape[1] == 3, "bodyPartVerts should be of shape (B, 3)"
    assert len(bodySegNormals.shape)==2 and bodySegNormals.shape[1] == 3, "bodyPartNormals should be of shape (B, 3)"

    # Criterion:
    distanceMatrix = np.linalg.norm(partVerts[:, np.newaxis, :] - bodySegVerts[np.newaxis, :, :], axis=2)# N x B
    topKpatchesIndices = np.argsort(distanceMatrix, axis=1)[:, :K] # N x K
    normals = bodySegNormals[topKpatchesIndices] # N x K x 3
    cloestPoitns = bodySegVerts[topKpatchesIndices] # N x K x 3
    penalty = ((partVerts[:, None, :] - cloestPoitns) * normals)
    penalty = penalty.sum(axis=2)
    penalty[penalty>=0.01] = 0
    penalty = -penalty
    penalty = penalty.mean(axis=1)
    return penalty.mean()

def penetration_penalty2(partVerts, bodySegVerts, bodySegNormals, K =1):
    """
    Calculate the penetration by projecting the part vertices to the normal of cloest body segment vertices.
    """
    # Verify inputs:
    assert len(partVerts.shape)==2 and partVerts.shape[1] == 3, "partVerts should be of shape (N, 3)"   
    assert len(bodySegVerts.shape)==2 and bodySegVerts.shape[1] == 3, "bodyPartVerts should be of shape (B, 3)"
    assert len(bodySegNormals.shape)==2 and bodySegNormals.shape[1] == 3, "bodyPartNormals should be of shape (B, 3)"

    # Criterion:
    distanceMatrix = np.linalg.norm(partVerts[:, np.newaxis, :] - bodySegVerts[np.newaxis, :, :], axis=2)# N x B
    topKpatchesIndices = np.argsort(distanceMatrix, axis=1)[:, :K] # N x K
    normals = bodySegNormals[topKpatchesIndices] # N x K x 3
    cloestPoitns = bodySegVerts[topKpatchesIndices] # N x K x 3
    penalty = ((partVerts[:, None, :] - cloestPoitns) * normals)
    penalty = penalty.sum(axis=2)
    penalty = penalty.mean(axis=1)

    # Imp1:
    # penalty = np.abs(penalty-0.01)

    # Imp2:
    penalty[penalty>0] = 0
    penalty = np.abs(penalty)

    return penalty.mean()

def overfitting_penalty(originalpartVerts, partVerts, threshold=0.05):
    """
    Calculate the overfitting penalty by comparing the original part vertices and the current part vertices.
    """
    # Verify inputs:
    assert len(originalpartVerts.shape)==2 and originalpartVerts.shape[1] == 3, "originalpartVerts should be of shape (N, 3)"
    assert len(partVerts.shape)==2 and partVerts.shape[1] == 3, "partVerts should be of shape (N, 3)"

    # Criterion:
    penalty = np.linalg.norm(partVerts - originalpartVerts, axis=1).max()
    return penalty if penalty>threshold else 0

def pressure(partPatchesTriple, bodySegVerts, bodySegForces, bodySegNormals, K=1):
    """
    Pressure = force/(sum(dis * area))
    K: number of closest patches used to calculate the pressure.
    """
    # Verify inputs:
    assert len(partPatchesTriple)==3, "partPatchesTriple should consists of centers, normals and areas"
    centers, normals, areas = partPatchesTriple # S x 3, S x 3, S x 1
    assert len(centers.shape)==2 and centers.shape[1] == 3, "Patch centers should be of shape (S, 3)"
    assert len(normals.shape)==2 and normals.shape[1] == 3, "Patch normals should be of shape (S, 3)"
    assert len(areas.shape)==1, "Patch areas should be of shape (S,)"
    assert len(centers) == len(normals) == len(areas), "Patch centers, normals and areas should have the same length"
    assert len(bodySegForces.shape)==1, "bodyPartForces should be of shape (B,)"
    assert len(bodySegVerts.shape)==2 and bodySegVerts.shape[1] == 3, "bodyPartVerts should be of shape (B, 3)"
    assert len(bodySegNormals.shape)==2 and bodySegNormals.shape[1] == 3, "bodyPartNormals should be of shape (B, 3)"

    # Criterion:
    distanceMatrix = np.linalg.norm(bodySegVerts[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)# N X S
    topKpatchesIndices = np.argsort(distanceMatrix, axis=1)[:, :K]
    distanceMatrix = distanceMatrix[np.arange(distanceMatrix.shape[0])[:, None], topKpatchesIndices] # N x K
    centers = centers[topKpatchesIndices] # N x K x 3
    normals = normals[topKpatchesIndices] # N x K x 3
    areas = areas[topKpatchesIndices] # N x K
    distanceMatrix = 1/(distanceMatrix+1e-5)
    normalWeights = np.abs((bodySegNormals[:, np.newaxis, :] * normals).sum(axis=2)) # N x K
    areas = (distanceMatrix * normalWeights * areas).sum(axis =1)/areas.sum(axis=1)
    pressure = bodySegForces / areas
    return pressure.mean()

def pressure2(partPatchesTriple, bodySegVerts, bodySegForces, bodySegNormals, K=1):
    """
    Pressure = force/(sum(dis * area))
    K: number of closest patches used to calculate the pressure.
    """
    # Verify inputs:
    assert len(partPatchesTriple)==3, "partPatchesTriple should consists of centers, normals and areas"
    centers, normals, areas = partPatchesTriple # S x 3, S x 3, S x 1
    assert len(centers.shape)==2 and centers.shape[1] == 3, "Patch centers should be of shape (S, 3)"
    assert len(normals.shape)==2 and normals.shape[1] == 3, "Patch normals should be of shape (S, 3)"
    assert len(areas.shape)==1, "Patch areas should be of shape (S,)"
    assert len(centers) == len(normals) == len(areas), "Patch centers, normals and areas should have the same length"
    assert len(bodySegForces.shape)==1, "bodyPartForces should be of shape (B,)"
    assert len(bodySegVerts.shape)==2 and bodySegVerts.shape[1] == 3, "bodyPartVerts should be of shape (B, 3)"
    assert len(bodySegNormals.shape)==2 and bodySegNormals.shape[1] == 3, "bodyPartNormals should be of shape (B, 3)"

    # Criterion:
    distanceMatrix = np.linalg.norm(bodySegVerts[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)# N X S
    topKpatchesIndices = np.argsort(distanceMatrix, axis=1)[:, :K]
    distanceMatrix = distanceMatrix[np.arange(distanceMatrix.shape[0])[:, None], topKpatchesIndices] # N x K
    centers = centers[topKpatchesIndices] # N x K x 3
    normals = normals[topKpatchesIndices] # N x K x 3
    areas = areas[topKpatchesIndices] # N x K
    distanceMatrix = 1/(distanceMatrix+1e-5)
    normalWeights = np.abs((bodySegNormals[:, np.newaxis, :] * normals).sum(axis=2)) # N x K
    areas = (distanceMatrix * normalWeights * areas).sum(axis =1)/(distanceMatrix * normalWeights).sum(axis=1)
    pressure = bodySegForces / areas
    return pressure.mean()

def shearForce(partPatchesTriple, bodySegVerts, bodySegForces, bodySegNormals, bodySegJoints, lateralWeight=0.6):
    """
    Calculate the shear force on the body segment by removing the normal component of the force vector.
    """
    # Verify inputs:    
    assert len(partPatchesTriple)==3, "partPatchesTriple should consists of centers, normals and areas"
    centers, normals, areas = partPatchesTriple # S x 3, S x 3, S x 1
    assert len(centers.shape)==2 and centers.shape[1] == 3, "Patch centers should be of shape (S, 3)"
    assert len(normals.shape)==2 and normals.shape[1] == 3, "Patch normals should be of shape (S, 3)"
    assert len(areas.shape)==1, "Patch areas should be of shape (S,)"
    assert len(bodySegVerts.shape)==2 and bodySegVerts.shape[1] == 3, "bodyPartVerts should be of shape (B, 3)"
    assert len(bodySegForces.shape)==1, "bodyPartForces should be of shape (B,)"
    assert len(bodySegNormals.shape)==2 and bodySegNormals.shape[1] == 3, "bodyPartNormals should be of shape (B, 3)"
    assert len(bodySegJoints.shape)==2 and bodySegJoints.shape[1] == 3, "bodySegJoints should be of shape (2, 3)"

    # Criterion:
    segmentAxis  = bodySegJoints[1] - bodySegJoints[0]
    segmentAxis = segmentAxis/np.linalg.norm(segmentAxis) # 1 x 3
    distanceMatrix = np.linalg.norm(bodySegVerts[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)# N X S
    cloestPointsIndices = np.argmin(distanceMatrix, axis=1) # N x 1
    normals = normals[cloestPointsIndices] # N x 3
    perAxis = np.cross(normals, segmentAxis)
    perAxis = perAxis/np.linalg.norm(perAxis, axis=1, keepdims=True)
    forcesVector = bodySegForces[:,None]* (np.array([[0, -1., 0]] + lateralWeight * bodySegNormals))# * self.bodyPart.normals # N x 3
    shearForce = np.linalg.norm(forcesVector - (forcesVector * normals).sum(axis=1)[:, None] * normals, axis=1)
    return shearForce.mean()

def lateralShearForce(partPatchesTriple, bodySegVerts, bodySegForces, bodySegNormals, segmentAxis, lateralWeight=0.6):
    """
    Calculate the shear force on the body segment by removing the normal component of the force vector.
    """
    # Verify inputs:    
    assert len(partPatchesTriple)==3, "partPatchesTriple should consists of centers, normals and areas"
    centers, normals, areas = partPatchesTriple # S x 3, S x 3, S x 1
    assert len(centers.shape)==2 and centers.shape[1] == 3, "Patch centers should be of shape (S, 3)"
    assert len(normals.shape)==2 and normals.shape[1] == 3, "Patch normals should be of shape (S, 3)"
    assert len(areas.shape)==1, "Patch areas should be of shape (S,)"
    assert len(bodySegVerts.shape)==2 and bodySegVerts.shape[1] == 3, "bodyPartVerts should be of shape (B, 3)"
    assert len(bodySegForces.shape)==1, "bodyPartForces should be of shape (B,)"
    assert len(bodySegNormals.shape)==2 and bodySegNormals.shape[1] == 3, "bodyPartNormals should be of shape (B, 3)"
    assert len(segmentAxis.shape)==2 and segmentAxis.shape[1] == 3, "bodySegJoints should be of shape (2, 3)"

    # Criterion:
    # segmentAxis  = bodySegJoints[1] - bodySegJoints[0]
    segmentAxis = segmentAxis/np.linalg.norm(segmentAxis) # 1 x 3
    distanceMatrix = np.linalg.norm(bodySegVerts[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)# N X S
    cloestPointsIndices = np.argmin(distanceMatrix, axis=1) # N x 1
    normals = normals[cloestPointsIndices] # N x 3
    perAxis = np.cross(normals, segmentAxis)
    perAxis = perAxis/np.linalg.norm(perAxis, axis=1, keepdims=True) # N x 3
    forcesVector = bodySegForces[:,None]* (np.array([[0, -1., 0]] + lateralWeight * bodySegNormals))# * self.bodyPart.normals # N x 3
    #shearForce = np.linalg.norm(forcesVector - (forcesVector * normals).sum(axis=1)[:, None] * normals, axis=1)

    # Project the force onto the plane:
    shearForce = forcesVector - (forcesVector * normals).sum(axis=1)[:, None] * normals
    # lateralShearForce = (shearForce[:,1]>0) * shearForce
    lateralShearForce = (shearForce * perAxis).sum(axis=1)[:,None] * perAxis
    lateralShearForce = lateralShearForce * (shearForce[:,1][:,None]<0)
    return np.linalg.norm(lateralShearForce, axis=1).mean()