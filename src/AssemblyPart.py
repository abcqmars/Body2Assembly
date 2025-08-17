import numpy as np
import open3d as o3d
from itertools import permutations
from .BodyModel import BodyModel
from .Helper import *
from .opt import ParametricModel, part_seg_dist, penetration_penalty, overfitting_penalty, pressure, shearForce
from .opt.localcriteria import penetration_penalty2

class AssemblyPart(ParametricModel):

    def __init__(self, alpha=0, a=1., b=1., beta=0., gamma=0., \
                     position=(0.,0.,0.), rotation=(0.,0.,0.), scales=(0.1, 0.1, 1), bodyPart=None, config=None, body=None):
        #Initialization before the first optimization. 
        super().__init__()
        all_params = (alpha, a, b, beta, gamma, *position, *rotation, *scales)
        self.params_copies = []
        self.params_copies.append(all_params)
        self.strategy = None
        # States.
        self.bodyPart = bodyPart
        self.config = config
        self.body = body

    # -----------------------Abstract methods-----------------------
    def setup_optimization(self, strategy=None):
        """
        Start a new optimization process based on the latest params copy
        """
        self.strategy = strategy if strategy is not None else self.strategy
        self.init_params = self._filter_params(self.get_latest_params()) #(alpha, a, b, beta, gamma, 0, 0, 0)
        self.optimized_params = None
        self.res = 10

    def _parametric_fn(self, x, y, params=None):# calculate samples over the parametric models.
        if params is None:
            all_params = self.get_original_params()
        elif len(params)==len(self.get_latest_params()):
            all_params = params
        else:
            all_params = self._revert_params(params, latest=True)
        shapeParams = self._get_shape_params(all_params)
        z = shapeParams[0] * ((x/shapeParams[1])**2 - (y/shapeParams[2])**2) + shapeParams[3] * (x**4) + shapeParams[4] * (y**4)
        rotMat = euler_angles_to_rotation_matrix(self._get_rotation_params(all_params))
        scaleMat = scales_to_scale_matrix(self._get_scale_params(all_params))
        if isinstance(x, np.ndarray):
            p = np.array([x.ravel(), y.ravel(), z.ravel()]).T @ scaleMat @ rotMat + np.array(self._get_position_params(all_params))
        else:
            p = np.array([[x, y, z]]) @ scaleMat @ rotMat + np.array(np.array(self._get_position_params(all_params)))
        return p

    def _objective_fn(self, params):
        E = 0

        # Prepare necessary inputs for criteria:
        xy = self._sample_xy(self.res)
        xyz = self._parametric_fn(xy[:,0], xy[:,1], params=params)
        init_xyz = self._parametric_fn(xy[:,0], xy[:,1])
        bodySegVerts = self.bodyPart.verts
        bodySegForces = self.bodyPart.forces
        bodySegNormals = self.bodyPart.normals
        bodySegjoints = self.bodyPart.segment
        partPatchesTriple = self._get_patches_triple(params)

        # Energy computation:
        E += self.config.localParams["bodyPlaneDist"] * part_seg_dist(xyz, bodySegVerts, bodySegForces)
        E += self.config.localParams["pressure"] * pressure(partPatchesTriple, bodySegVerts, bodySegForces, bodySegNormals, K=int(self.res * self.res / 4))
        E += self.config.localParams["penetration"] * penetration_penalty(xyz, bodySegVerts, bodySegNormals, K=1)
        E += self.config.localParams["shear"] * shearForce(partPatchesTriple, bodySegVerts, bodySegForces, bodySegNormals, bodySegjoints, lateralWeight=0.6)
        E += self.config.localParams["overfit"] * overfitting_penalty(xyz, init_xyz, threshold=0.05)

        return E

    def _update_params(self, params=None):
        """
        Post process to the optimized params.
        """
        if params is None:
            assert self.optimized_params is not None
            self.params_copies.append(self._revert_params(self.optimized_params, latest=True))
        elif len(params)==len(self.get_original_params()):
            self.optimized_params = params
            self.params_copies.append(params)
        else:
            self.optimized_params = params
            newParams = self._revert_params(params, latest=True)
            self.params_copies.append(newParams)

    # -----------------------Helper functions-----------------------
    def num_results(self):
        return len(self.params_copies)

    def set_strategy(self, strategy):
        self.strategy = strategy

    def get_original_params(self):
        return self.params_copies[0][:]

    def get_latest_params(self):
        return self.params_copies[-1][:]

    def _filter_params(self, params):
        """
        Takes all params of a single part and return the learnable params.
        """
        if self.strategy is None:
            return params
        if self.strategy=="All":
            return params
        if self.strategy=="ShapeOnly":
            return self._get_shape_params(params)
        if self.strategy=="PositionOnly":
            return self._get_position_params(params)
        if self.strategy=="ScaleOnly":
            return self._get_scale_params(params)
        if self.strategy =="RotationOnly":
            return self._get_rotation_params(params)
        if self.strategy=="ScalePosition":
            return self._get_scale_params(params) + self._get_position_params(params)
        if self.strategy=="ShapeRotation":
            return self._get_shape_params(params) + self._get_rotation_params(params)
        if self.strategy=="ShapePosition":
            return self._get_shape_params(params) + self._get_position_params(params)
        else:
            assert False, f"{self.strategy} do not exist."

    def _revert_params(self, params, latest=True):
        """
        Takes filtered params of a single part and return all params.
        """
        fullParams = self.get_latest_params() if latest else self.get_original_params()
        if self.strategy=="All":
            return params
        if self.strategy=="ShapeOnly":
            fullParams = list(fullParams)
            fullParams[:5] = params
            return tuple(fullParams)
        if self.strategy=="PositionOnly":
            fullParams = list(fullParams)
            fullParams[5:8] = params
            return tuple(fullParams)
        if self.strategy=="ScaleOnly":
            fullParams = list(fullParams)
            fullParams[11:] = params
            return tuple(fullParams)
        if self.strategy =="RotationOnly":
            fullParams = list(fullParams)
            fullParams[8:11] = params
            return tuple(fullParams)
        if self.strategy=="ScalePosition":
            fullParams = list(fullParams)
            fullParams[11:] = params[:3]
            fullParams[5:8] = params[3:]
            return tuple(fullParams)
        if self.strategy=="ShapeRotation":
            fullParams = list(fullParams)
            fullParams[:5] = params[:5]
            fullParams[8:11] = params[-3:]
            return tuple(fullParams)
        if self.strategy=="ShapePosition":
            fullParams = list(fullParams)
            fullParams[:5] = params[:5]
            fullParams[8:11] = params[-3:]
            return tuple(fullParams)
        else:
            assert False, f"{self.strategy} do not exist."

    def _get_shape_params(self, params):
        return params[:5]

    def _get_position_params(self, params):
        return params[5:8]

    def _get_rotation_params(self, params):
        return params[8:11]

    def _get_scale_params(self, params):
        p = params[11:-1]
        return (*p, 1)

    def _get_faces(self, res):
        faces = []
        for i in range(res-1):
            for j in range(res-1):
                idx1 = i * res + j
                idx2 = idx1 + 1
                idx3 = idx1 + res
                idx4 = idx3 + 1
                faces.append([idx1, idx2, idx3])
                faces.append([idx3, idx2, idx4])
        return faces

    def _get_patches_triple(self, params):
        xy = self._sample_xy(self.res)
        xyz = self._parametric_fn(xy[:,0], xy[:,1], params=params)
        faces = self._get_faces(self.res)
        centers = np.zeros((len(faces), 3))
        normals = np.zeros((len(faces), 3))
        areas = np.zeros(len(faces))
        for i, f in enumerate(faces):
            p1 = xyz[f[0]]
            p2 = xyz[f[1]]
            p3 = xyz[f[2]]
            center = (p1 + p2 + p3)/3
            normal = np.cross(p3-p1, p2-p1)
            area = np.linalg.norm(normal)
            normal = normal / area * (-1 if (normal * self.bodyPart.avgNormal).sum() > 0 else 1)
            centers[i] = center
            normals[i] = normal
            areas[i] = area
        return centers, normals, areas

    
    # -----------------------visualization related-----------------------
    def __repr__(self):
        if len(self.params_copies)>0:
            latest_params = self.get_latest_params()
            offset = [l-o for l, o in zip(self._get_position_params(self.get_latest_params()), self._get_position_params(self.get_original_params()))]
            rotattionOffset = [l-o for l, o in zip(self._get_rotation_params(self.get_latest_params()), self._get_rotation_params(self.get_original_params()))]
            return f"z={self._get_shape_params(latest_params)[0]:.2e}((x/{self._get_shape_params(latest_params)[1]:.2e})^2 - (y/{self._get_shape_params(latest_params)[2]:.2e})^2)" + \
            f"+ {self._get_shape_params(latest_params)[3]:.2e}x^4 + {self._get_shape_params(latest_params)[4]:.2e}y^4 \n" + \
            f"Position Offset: {offset[0]:.2e}, {offset[1]:.2e}, {offset[2]:.2e} \n" + \
            f"Rotation Offset: {rotattionOffset[0]:.2e}, {rotattionOffset[1]:.2e}, {rotattionOffset[2]:.2e} \n"
        return ""

    def get_edge_vertIndices(self, res=None):
        if res is None:
            res = self.res
        edges = []
        edges.append([j for j in range(res)])
        edges.append([i * res +  res - 1 for i in range(1, res)])
        edges.append([(res - 1) * res + j for j in range(res - 2, -1, -1)] if res > 1 else [])
        edges.append([i * res for i in range(res - 2, 0, -1)] if res > 1 else [])
        return edges


    def get_open3d_mesh(self, init=False, stage=None):
        """Convert generated vertices and faces into an Open3D mesh."""
        visRes = 20
        xy = self._sample_xy(visRes)
        x = xy[:,0]
        y = xy[:,1]
        if stage is None:
            all_params = self.get_original_params() if init else self.get_latest_params()
        else:
            assert stage>=0 and stage<self.num_results() and isinstance(stage, int), f"stage: {stage}, num_results: {self.num_results()}"
            all_params = self.params_copies[stage]
        shapeParams = self._get_shape_params(all_params)
        z = shapeParams[0] * ((x/shapeParams[1])**2 - (y/shapeParams[2])**2) + shapeParams[3] * (x**4) + shapeParams[4] * (y**4)
        thinckness = 0.01 * 3
        z2 = shapeParams[0] * ((x/shapeParams[1])**2 - (y/shapeParams[2])**2) + shapeParams[3] * (x**4) + shapeParams[4] * (y**4) + thinckness

        rotMat = euler_angles_to_rotation_matrix(self._get_rotation_params(all_params))
        scales = list(self._get_scale_params(all_params))
        maxIdx = np.argmax(scales[:2])
        minIdx = np.argmin(scales[:2])
        if self.bodyPart.partId == 4 or self.bodyPart.partId==5:
            scales[maxIdx] *= 0.75
        else:
            scales[maxIdx] *= 0.85

        scaleMat = scales_to_scale_matrix(scales)
        x = np.concatenate([x.flatten(), x.flatten()])
        y = np.concatenate([y.flatten(), y.flatten()])
        z = np.concatenate([z.flatten(), z2.flatten()])
        p = np.array([x, y, z]).T @ scaleMat @ rotMat + np.array(self._get_position_params(all_params))
        faces = []
        for i in range(visRes-1):
            for j in range(visRes-1):
                # Top surface
                idx1 = i * visRes + j
                idx2 = idx1 + 1
                idx3 = idx1 + visRes
                idx4 = idx3 + 1
                faces.append([idx3, idx2, idx1])
                faces.append([idx4, idx2, idx3])

                # Bottom surface
                idx1 = i * visRes + j + visRes*visRes
                idx2 = idx1 + 1
                idx3 = idx1 + visRes
                idx4 = idx3 + 1
                faces.append([idx1, idx2, idx3])
                faces.append([idx3, idx2, idx4])

        # Side surfaces
        edgeVertIndices = []
        edgeVertIndices += [j for j in range(visRes)]
        edgeVertIndices += [i * visRes +  visRes - 1 for i in range(1, visRes)]
        edgeVertIndices += [(visRes - 1) * visRes + j for j in range(visRes - 2, -1, -1)] if visRes > 1 else []
        edgeVertIndices += [i * visRes for i in range(visRes - 2, 0, -1)] if visRes > 1 else []
        for i, idx in enumerate(edgeVertIndices):
            idx1 = idx
            idx2 = edgeVertIndices[(i + len(edgeVertIndices) + 1)%len(edgeVertIndices)]
            idx3 = idx + visRes * visRes
            idx4 = edgeVertIndices[(i + len(edgeVertIndices) - 1)%len(edgeVertIndices)] + visRes*visRes
            faces.append([idx1, idx2, idx3])
            faces.append([idx1, idx3, idx4])

        assert all([all([i>=0 and i <800 for i in f])for f in faces])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(p)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.8, 0.8, 0.8]] * len(p)))
        mesh.compute_vertex_normals()
        mesh = mesh.subdivide_loop(number_of_iterations=2)
        mesh.compute_vertex_normals()
        return mesh

    def get_center_point(self, init=False):
        """
        Get the center point of the part.
        """
        visRes = 20
        xy = self._sample_xy(visRes)
        x = xy[:,0]
        y = xy[:,1]
        if init:
            all_params = self.get_original_params()
        else:
            all_params = self.get_latest_params()
        shapeParams = self._get_shape_params(all_params)
        thinckness = 0.01 * 3 * 0.6
        z = shapeParams[0] * ((x/shapeParams[1])**2 - (y/shapeParams[2])**2) + shapeParams[3] * (x**4) + shapeParams[4] * (y**4)+ thinckness

        rotMat = euler_angles_to_rotation_matrix(self._get_rotation_params(all_params))
        scales = list(self._get_scale_params(all_params))
        maxIdx = np.argmax(scales[:2])
        scales[maxIdx] *= 0.85
        scaleMat = scales_to_scale_matrix(scales)
        p = np.array([x, y, z]).T @ scaleMat @ rotMat + np.array(self._get_position_params(all_params))
        return p[visRes//2 * visRes + visRes//2]

    def get_edge_centerPoints(self, init=False):
        """
        Get the center point of the part edges.
        """
        visRes = 20
        xy = self._sample_xy(visRes)
        x = xy[:,0]
        y = xy[:,1]
        if init:
            all_params = self.get_original_params()
        else:
            all_params = self.get_latest_params()
        shapeParams = self._get_shape_params(all_params)
        thinckness = 0.01 * 3 / 2 
        z = shapeParams[0] * ((x/shapeParams[1])**2 - (y/shapeParams[2])**2) + shapeParams[3] * (x**4) + shapeParams[4] * (y**4)+ thinckness

        rotMat = euler_angles_to_rotation_matrix(self._get_rotation_params(all_params))
        scales = list(self._get_scale_params(all_params))
        maxIdx = np.argmax(scales[:2])
        scales[maxIdx] *= 0.85
        scaleMat = scales_to_scale_matrix(scales)
        p = np.array([x, y, z]).T @ scaleMat @ rotMat + np.array(self._get_position_params(all_params))

        edgePoints = []
        for edgeIndieces in self.get_edge_vertIndices(res=visRes):
            edgePoints.append(p[edgeIndieces[len(edgeIndieces)//2]])

        edgePoints = sorted(edgePoints, key=lambda p: pointLineDist(p, self.bodyPart.segment), reverse=True)[:2]
        return edgePoints

    def get_vis_meshes(self, init=False):
        """
        Get the surface mesh, part mesh, rounded mesh, for a part.
        """
        surface_mesh = None
        part_mesh = None
        roundedPart_Mesh = None

        visRes = 20
        xy = self._sample_xy(visRes)
        x = xy[:,0]
        y = xy[:,1]
        if init:
            all_params = self.get_original_params()
        else:
            all_params = self.get_latest_params()

        shapeParams = self._get_shape_params(all_params)
        z = shapeParams[0] * ((x/shapeParams[1])**2 - (y/shapeParams[2])**2) + shapeParams[3] * (x**4) + shapeParams[4] * (y**4)
        thinckness = 0.0
        z2 = shapeParams[0] * ((x/shapeParams[1])**2 - (y/shapeParams[2])**2) + shapeParams[3] * (x**4) + shapeParams[4] * (y**4) + thinckness

        rotMat = euler_angles_to_rotation_matrix(self._get_rotation_params(all_params))
        scales = list(self._get_scale_params(all_params))
        scaleMat = scales_to_scale_matrix(scales)
        x = np.concatenate([x.flatten(), x.flatten()])
        y = np.concatenate([y.flatten(), y.flatten()])
        z = np.concatenate([z.flatten(), z2.flatten()])
        p = np.array([x, y, z]).T @ scaleMat @ rotMat + np.array(self._get_position_params(all_params))
        faces = []
        surface_faces = []
        for i in range(visRes-1):
            for j in range(visRes-1):
                # Top surface
                idx1 = i * visRes + j
                idx2 = idx1 + 1
                idx3 = idx1 + visRes
                idx4 = idx3 + 1
                faces.append([idx3, idx2, idx1])
                faces.append([idx4, idx2, idx3])
                surface_faces.append([idx3, idx2, idx1])
                surface_faces.append([idx4, idx2, idx3])
                surface_faces.append([idx1, idx2, idx3])
                surface_faces.append([idx3, idx2, idx4])

                # Bottom surface
                idx1 = i * visRes + j + visRes*visRes
                idx2 = idx1 + 1
                idx3 = idx1 + visRes
                idx4 = idx3 + 1
                faces.append([idx1, idx2, idx3])
                faces.append([idx3, idx2, idx4])

        # Side surfaces
        edgeVertIndices = []
        edgeVertIndices += [j for j in range(visRes)]
        edgeVertIndices += [i * visRes +  visRes - 1 for i in range(1, visRes)]
        edgeVertIndices += [(visRes - 1) * visRes + j for j in range(visRes - 2, -1, -1)] if visRes > 1 else []
        edgeVertIndices += [i * visRes for i in range(visRes - 2, 0, -1)] if visRes > 1 else []
        for i, idx in enumerate(edgeVertIndices):
            idx1 = idx
            idx2 = edgeVertIndices[(i + len(edgeVertIndices) + 1)%len(edgeVertIndices)]
            idx3 = idx + visRes * visRes
            idx4 = edgeVertIndices[(i + len(edgeVertIndices) - 1)%len(edgeVertIndices)] + visRes*visRes
            faces.append([idx1, idx2, idx3])
            faces.append([idx1, idx3, idx4])

        assert all([all([i>=0 and i <800 for i in f])for f in faces])
        part_mesh = o3d.geometry.TriangleMesh()
        part_mesh.vertices = o3d.utility.Vector3dVector(p)
        part_mesh.triangles = o3d.utility.Vector3iVector(faces)
        part_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.8, 0.8, 0.8]] * len(p)))
        part_mesh.compute_vertex_normals()
        roundedPart_Mesh = part_mesh.subdivide_loop(number_of_iterations=2)
        roundedPart_Mesh.compute_vertex_normals()

        surface_mesh = o3d.geometry.TriangleMesh()
        surface_mesh.vertices = o3d.utility.Vector3dVector(p[:(len(p)//2)])
        surface_mesh.triangles = o3d.utility.Vector3iVector(surface_faces)
        surface_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.8, 0.8, 0.8]] * (len(p)//2)))
        surface_mesh.compute_vertex_normals()
        return surface_mesh, part_mesh, roundedPart_Mesh
    
    def get_partSize(self, init=False):
        params = self.get_original_params() if init else self.get_latest_params()
        scales = self._get_scale_params(params)
        return scales[0] * scales[1]