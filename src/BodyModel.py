import os
import gc
import numpy as np
import open3d as o3d
import torch
import pickle
import joblib
from smplx import SMPL
from .Helper import *

class BodyPart:
    def __init__(self, partId, segment, verts, normals, forces, area):
        self.partId = partId
        self.segment = segment # 2 X 3,  line segment between two joints.
        self.verts = verts # N X 3, corresponding body verts
        self.normals = normals # N X 3, normals for each vertices
        self.forces = forces
        self.area = area
        if self.forces is None:
            self.forces = -self.normals[:,1]
            self.forces[self.forces<0] = 0
    
    @property
    def avgNormal(self):
        return self.normals.mean(axis=0)

class BodyModel:
    def __init__(self, modelPth = "", gender='neutral', config=None):
        assert os.path.exists(modelPth), f"Model Path {modelPth} is not found."
        # assert config is not None, f"Config is not initialized."
        if config is not None:
            self.config = config
        else:
            self.config = {"segmentStrategy": "segment_butt_updown"}

        # Data
        self.bodyMesh = None
        self.bodyJoints = None
        self.bodyWeights=None
        self.partLabel=None
        self.bodyParts = None
        self.bodyForces = None
        self.bodyArea = None
        self.smplModel = SMPL(model_path=modelPth, gender=gender)

        # Body Params
        self.poseParams = (torch.rand(1, 72)*2-1) * 0
        self.poseParams[:, :3] = 0
        self.globalOrient = self.poseParams[:, :3]
        self.poseParams = self.poseParams[:, 3:]
        self.shapetensor = torch.rand(1, 10) * 0
        self.pose2rot = True
        # Spatial Params:
        self.rotationMat = np.eye(3)
        self._preprocess()

        # segment the smpl body based on skeleton.
        self._segment()

    @property
    def bodyVerts(self):
        assert self.bodyMesh is not None, f"self.bodyMesh is not initialized."
        return np.asarray(self.bodyMesh.vertices)

    @property
    def conobodyVerts(self):
        assert self.smplModel is not None, f"self.bodyMesh is not initialized."
        smplParser = self.smplModel()
        return smplParser.vertices.detach().cpu().numpy()[0]

    @property
    def connoJoints(self):
        assert self.smplModel is not None, f"self.bodyMesh is not initialized."
        smplParser = self.smplModel()
        return smplParser.joints.detach().cpu().numpy()[0]

    @property
    def bodyFaces(self):
        assert self.bodyMesh is not None, f"self.bodyMesh is not initialized."
        return np.asarray(self.bodyMesh.triangles)
    
    @property
    def bodyNormals(self):
        assert self.bodyMesh is not None, f"self.bodyMesh is not initialized."
        return np.asarray(self.bodyMesh.vertex_normals)

    @property
    def bodySegments(self):
        assert self.bodyJoints is not None,  f"self.bodyJoints is not initialized."
        body_joint_pairs = [
            (0, 1),  
            (0, 2),  
            (1, 4),  
            (2, 5),  
            (4, 7),  
            (5, 8),  
            (8, 11),  
            (7, 10),  
            (0, 3),  
            (3, 6),  
            (6, 9),  
            (9, 13),  
            (9, 14), 
            (17, 19),
            (16, 18),
        ]
        return [(self.bodyJoints[i], self.bodyJoints[j]) for i, j in body_joint_pairs]

    @property
    def partEdges(self):
        edges = [
            (0,1),
            (0,2),
            (1,2),
            # (1,4),
            # (2,5),
            # (1,2),
            (0,3),
            # (3,9),
            (8,13),
            (17,19)
        ]
        partIds = []
        for part in self.bodyParts:
            partIds.append(part.partId)
        def valid_edge(e):
            return (e[0] in partIds) and (e[1] in partIds)
        return list(filter(valid_edge, edges))

    def _joint2segment(self, jointId):
        j2s = {0: [3, 0],
               1: [1, 4],
               2: [2, 5],
               3: [5, 3],
               4: [4, 7],
               5: [5, 8],
               9: [9, 6],
               16:[16, 18],
               17:[17, 19],
               18:[18, 20],
               19:[19, 21]
               }
        assert jointId in j2s, f"JointId: {jointId} is not registered"
        return self.bodyJoints[j2s[jointId]]


    def _init_data(self):
        if not all([i is None for i in [self.bodyMesh, self.bodyJoints, self.bodyWeights, self.partLabel, self.bodyParts, self.bodyForces]]):
            del self.bodyMesh, self.bodyJoints, self.bodyWeights, self.partLabel, self.bodyParts, self.bodyForces
            gc.collect()
            self.bodyMesh = None
            self.bodyJoints = None
            self.bodyWeights=None
            self.partLabel=None
            self.bodyParts=None
            self.bodyForces=None

    def update_params(self, pose=None, shape=None):
        if pose is not None:
            self.globalOrient = pose[:, :3]
            self.poseParams = pose[:, 3:]
        if shape is not None:
            self.shapetensor = shape
        self._preprocess() # call to update the vertices.
        self._segment()

    def load_params(self, paramPth = "", rotAngles = (0., 0., 0.)):
        assert os.path.exists(paramPth), f"Param Path {paramPth} is not found."
        assert len(rotAngles) == 3, f"{len(rotAngles)} angles params are given."
        self._init_data()
        if  paramPth.endswith('.pkl'):
            try:
                with open(paramPth, 'rb') as f:
                    data = pickle.load(f)
                self.globalOrient = (data['global_orient'][0][0] @ torch.from_numpy(euler_angles_to_rotation_matrix([np.pi, 0, 0])).float()).unsqueeze(0).unsqueeze(0)
                self.poseParams = data['body_pose']# 1 x 23 x 3 x 3
                self.shapetensor = data['betas']
                self.pose2rot = False
            except:
                data = joblib.load(paramPth)
                self.globalOrient = data['global_orient'][0]
                self.poseParams = data['body_pose'][0].view(1,-1)
                self.shapetensor = data['betas']
                self.pose2rot = True


        elif paramPth.endswith('.txt'):
            self.poseParams  = torch.from_numpy(np.loadtxt(paramPth, max_rows=24, usecols=[0, 1, 2])).view(1, -1).float()
            self.poseParams[:, :3] = 0
            self.globalOrient = self.poseParams[:, :3]# 1x3
            self.poseParams = self.poseParams[:, 3:] # 1x69
            self.shapetensor = torch.from_numpy(np.loadtxt(paramPth, skiprows=24, usecols=range(10))).view(1, -1).float()
            self.pose2rot=True

        self.rotationMat = angles2Rot(rotAngles)
        self._preprocess()
        self._segment()

        # Concatenate self.globalOrient and self.poseParams into a 1 x 72 tensor and return
        return torch.cat([self.globalOrient, self.poseParams], dim=1)

    @property
    def _isPreprocessed(self):
        return all([i is not None for i in [self.bodyMesh, self.bodyJoints, self.bodyWeights, self.partLabel]])

    def _preprocess(self):
        smplParser = self.smplModel(global_orient=self.globalOrient, body_pose=self.poseParams, betas=self.shapetensor, pose2rot=self.pose2rot)
        bodyVerts = smplParser.vertices.detach().cpu().numpy()[0] @ self.rotationMat
        bodyFaces = self.smplModel.faces
        self.bodyMesh = o3d.geometry.TriangleMesh()
        self.bodyMesh.vertices = o3d.utility.Vector3dVector(bodyVerts)
        self.bodyMesh.triangles = o3d.utility.Vector3iVector(bodyFaces)
        self.bodyMesh.compute_vertex_normals()
        self.bodyJoints = smplParser.joints.detach().cpu().numpy()[0] @ self.rotationMat
        self.bodyWeights = self.smplModel.lbs_weights.detach().cpu().numpy()

        # Force model:
        vertAreas = np.zeros(len(bodyVerts), dtype=np.float32)
        vertPressure = -self.bodyNormals[:, 1]
        vertPressure[vertPressure<0] = 0
        for face in bodyFaces:
            vId1 = face[0]
            vId2 = face[1]
            vId3 = face[2]
            p1, p2, p3 = bodyVerts[vId1], bodyVerts[vId2], bodyVerts[vId3]
            v1 = p2 - p1
            v2 = p3 - p2
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            vertAreas[vId1] += area / 3.0
            vertAreas[vId2] += area / 3.0
            vertAreas[vId3] += area / 3.0
        self.bodyForces = vertAreas * vertPressure
        self.bodyArea = vertAreas

    def _segment(self):
        self.partLabel = np.argmax(self.bodyWeights, axis=1)
        if self.config["segmentStrategy"] =="segment_butt_updown":
            partIndices = (self.partLabel == 0) & (np.asarray(self.conobodyVerts)[:, 1] > self.connoJoints[0][1])
            self.partLabel[partIndices] = 3
        self.partLabel[self.partLabel == 6] = 3
        self.partLabel[self.partLabel == 13] = 9
        self.partLabel[self.partLabel == 14] = 9

        # Filtering vertices with negative normals.
        self.partLabel[np.asarray(self.bodyMesh.vertex_normals)[:, 1]>-0.2] = -1
        # Filtering vertices at front body.
        self.partLabel[np.asarray(self.conobodyVerts)[:, 2] > self.connoJoints[13][2]] = -1

        parts2skip = [-1, 7, 8, 10,11, 22, 23, 15, 16, 17, 12, 20, 21]
        self.bodyParts = []
        for partId in np.unique(self.partLabel):
            if partId in parts2skip:
                continue
            partVerts = self.bodyVerts[np.where(self.partLabel == partId)[0]]
            if len(partVerts)<20:
                continue
            partNormals = self.bodyNormals[np.where(self.partLabel == partId)[0]]
            partForces = self.bodyForces[np.where(self.partLabel == partId)[0]]
            area = self.bodyArea[np.where(self.partLabel == partId)[0]]
            self.bodyParts.append(BodyPart(partId, self._joint2segment(partId), partVerts, partNormals, partForces, area))
        