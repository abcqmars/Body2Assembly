import numpy as np
import open3d as o3d
from itertools import permutations
from .BodyModel import BodyModel
from .Helper import *
from .opt import ParametricModel
from .AssemblyPart import AssemblyPart
from .Config import Config
from .SmplGraph import SMPLGraph
from .Helper_bezier import create_cylinder_mesh_from_bezierCurve, create_bezier_curve, create_cylinder_mesh_from_2endPoints
from .opt import parts_penetrion, parts_smoothness_penalty, overfitting_penalty, parts_overlap

class Furniture(ParametricModel):
    def __init__(self, bodyModel:BodyModel, config=None):
        super().__init__()
        self.config = config
        # Data:
        self.bodyModel = bodyModel
        self.furnParts = []
        self.bodyParts = []
        self.furnEdges = []
        # Preprocess:
        self._preprocess_body()

        # Optimization:
        self.strategy = ""
        self.paramsOrder = []
        self._optimize()

    # -----------------------Abstract methods-----------------------
    def setup_optimization(self, strategy=None):
        self.strategy = strategy
        partCandidateIndices = set()
        for edge in self.furnEdges:
            partCandidateIndices.add(edge[0])
            partCandidateIndices.add(edge[1])
        self.paramsOrder = list(partCandidateIndices)
        self.init_params = []
        for partIdx in self.paramsOrder:
            self.furnParts[partIdx].set_strategy(self.strategy)
            self.init_params.extend(self._get_filtered_params(partIdx))
        self.optimized_params = None
        self.res = 10

    def _parametric_fn(self):# calculate samples over the parametric models.
        assert False, "_parametric_fn is not excutable for whole furniture."

    def _objective_fn(self, params=None):# takes parameters as input to return result.
        E = 0
        paramsSize = int(len(params)/len(self.paramsOrder))
        for edge in self.furnEdges:
            # Prepare inputs:
            partIdx1 = edge[0]
            partIdx2 = edge[1]
            orderIdx1 = self._get_orderIdx(partIdx1)
            orderIdx2 = self._get_orderIdx(partIdx2)
            part1 = self.furnParts[partIdx1]
            part2 = self.furnParts[partIdx2]
            params1 = self._get_revert_params(params[orderIdx1*paramsSize:(orderIdx1+1)*paramsSize], partIdx1)
            params2 = self._get_revert_params(params[orderIdx2*paramsSize:(orderIdx2+1)*paramsSize], partIdx2)
            xy = self._sample_xy(self.res)
            part1xyz = part1._parametric_fn(xy[:,0], xy[:,1], params=params1) # N x 3
            part2xyz = part2._parametric_fn(xy[:,0], xy[:,1], params=params2) # N x 3
            part1EdgesIndices = part1.get_edge_vertIndices(self.res)
            part2EdgesIndices = part2.get_edge_vertIndices(self.res)
            part1EdgesVerts = [part1xyz[edgeindices] for edgeindices in part1EdgesIndices]
            part2EdgesVerts = [part2xyz[edgeindices] for edgeindices in part2EdgesIndices]

            # Criterion
            E += parts_penetrion(part1xyz, part2xyz) * self.config.globalParams["partDist"]
            E += parts_smoothness_penalty(part1EdgesVerts, part2EdgesVerts) * self.config.globalParams["smoothness"]
            E += parts_overlap(part1xyz, part2xyz, part1.bodyPart.verts, part2.bodyPart.verts, part1.bodyPart.area, part2.bodyPart.area)* self.config.globalParams["overlap"]

        for i, partId in enumerate(self.paramsOrder):
            p = self._get_revert_params(params[i*paramsSize:(i+1)*paramsSize], partId)
            xy = self._sample_xy(self.res)
            xyz = self.furnParts[partId]._parametric_fn(xy[:,0], xy[:,1], params=p)
            last_params = self.furnParts[partId].get_latest_params()
            last_xyz = self.furnParts[partId]._parametric_fn(xy[:,0], xy[:,1], params=last_params)
            E += overfitting_penalty(last_xyz, xyz) * self.config.globalParams["divergence"]

        return E
    

    def _update_params(self, params=None):
        paramsSize = int(len(params)/len(self.paramsOrder))
        for i, part in enumerate(self.furnParts):
            if i in self.paramsOrder:
                orderIdx = self._get_orderIdx(i)
                partParams = params[orderIdx*paramsSize:(orderIdx+1)*paramsSize]
                part._update_params(partParams)
            else:
                latestParams = part.get_latest_params()
                part._update_params(latestParams)

    # ----------------------Some Helper functions----------------------

    def __repr__(self):
        s = "---------------------\n"
        for p in self.furnParts:
            s += p.__repr__()
            s += "\n"
        s +="---------------------\n"
        return s

    def _preprocess_body(self):
        for bodyPart in self.bodyModel.bodyParts:
            # Parse the body part.
            o3dVerts = o3d.geometry.PointCloud()
            o3dVerts.points = o3d.utility.Vector3dVector(bodyPart.verts)
            bbox = o3dVerts.get_oriented_bounding_box()
            boxVerts = np.asarray(bbox.get_box_points())
            faces = get_cuboid_faces(boxVerts)

            def align(f):
                p1 = boxVerts[f[0]]
                p2 = boxVerts[f[1]]
                p3 = boxVerts[f[2]]
                p4 = boxVerts[f[3]]
                pc = (p1 + p2 + p3 + p4)/4
                return pc.dot(-bodyPart.avgNormal)

            def distance_align(f):
                p1 = boxVerts[f[0]]
                p2 = boxVerts[f[1]]
                p3 = boxVerts[f[2]]
                p4 = boxVerts[f[3]]
                pc = (p1 + p2 + p3 + p4)/4
                bodyVerts = bodyPart.verts
                dist = np.linalg.norm(bodyVerts - pc[np.newaxis, :], axis=1)
                return dist.mean()

            faces = sorted(faces, key=distance_align)
            candiFace = faces[0]
            
            # if downwardFace is not None:
            p1 = boxVerts[candiFace[0]]
            p2 = boxVerts[candiFace[1]]
            p3 = boxVerts[candiFace[2]]
            p4 = boxVerts[candiFace[3]]
            center = (p1 + p2 + p3 + p4)/4
            normal = np.cross(p2-p1, p3-p1)
            normal /= np.linalg.norm(normal)
            rotMat, scales = get_rectangle_frame(np.array([p1, p2, p3, p4]))
            segmentAxis = bodyPart.segment[1] - bodyPart.segment[0]
            segmentLength = np.linalg.norm(segmentAxis)

            # Initial tolerance for parts &  calibrate body segment.
            if np.abs((segmentAxis * rotMat[0]).sum()) > np.abs((segmentAxis * rotMat[1]).sum()):
                scales[0] *= 0.92
                scales[1] *= 1.1
                bodyPart.segment[1] = bodyPart.segment[0] + rotMat[0] * segmentLength

            else:
                scales[1] *= 0.92
                scales[0] *= 1.1
                bodyPart.segment[1] = bodyPart.segment[0] + rotMat[1] * segmentLength

            part = AssemblyPart(position = (center[0], center[1], center[2]), scales=scales, rotation=rotation_matrix_to_euler_angles(rotMat), bodyPart=bodyPart, config=self.config, body=self.bodyModel)
            self.furnParts.append(part)

        for edge in self.bodyModel.partEdges:
            pId1, pId2 = edge
            self.furnEdges.append((self._getPartIdx(pId1), self._getPartIdx(pId2)))

    def _getPartIdx(self, pId):
        for i, p in enumerate(self.furnParts):
            if p.bodyPart.partId==pId:
                return i
        assert False, "Get partIdx failed"

    #--------------------------------visualization related--------------------------------
    def get_connector_meshes(self, init=False):
        validPartIds = []
        for part in self.furnParts:
            validPartIds.append(part.bodyPart.partId)

        furnGraph = SMPLGraph(valid_joints=validPartIds)
        meshes = []
        edges = furnGraph.reduced_graph.get_edges()
        armPartsId = [18, 19, 16, 17]
        # ***** Create rods between parts (except arm parts) *****
        for edge in edges:
            if edge[0] in armPartsId or edge[1] in armPartsId:
                continue
            partIdx1 = self._getPartIdx(edge[0])
            partIdx2 = self._getPartIdx(edge[1])
            part1 = self.furnParts[partIdx1]
            part2 = self.furnParts[partIdx2]
            p_start = part1.get_center_point(init)
            p_end = part2.get_center_point(init)
            partNormal1 = part1.bodyPart.avgNormal
            partNormal2 = part2.bodyPart.avgNormal
            dist = 0.2
            p1 = p_start + partNormal1 * dist
            p2 = p_end + partNormal2 * dist
            # Create a Bézier curve from the two points
            nodes = np.array([p_start, p1, p2, p_end]).T
            curve = create_bezier_curve(nodes)
            # Create a cylinder mesh along the curve
            cylinder_mesh = create_cylinder_mesh_from_bezierCurve(curve, radius=0.02, num_points=20, num_segments=20)
            cylinder_mesh = cylinder_mesh.subdivide_loop(number_of_iterations=2)
            meshes.append(cylinder_mesh)

        # ***** Create rods for arm parts *****
        meshes += self._create_armrods()

        # ***** Create pillars to ground *****
        meshes += self._create_pillars()

        # ***** Create ground mesh *****
        ground_size = 0.75
        groundy = self.bodyModel.bodyVerts[:, 1].min() - 0.01
        ground_mesh = o3d.geometry.TriangleMesh()
        ground_vertices = [
            [-ground_size, groundy, -ground_size],
            [ground_size, groundy, -ground_size],
            [ground_size, groundy, ground_size],
            [-ground_size, groundy, ground_size],
        ]
        ground_faces = [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 2],
            [0, 2, 3],
        ]
        ground_mesh.vertices = o3d.utility.Vector3dVector(ground_vertices)
        ground_mesh.triangles = o3d.utility.Vector3iVector(ground_faces)
        ground_mesh.compute_vertex_normals()
        ground_mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Set ground color

        return meshes, ground_mesh

    def _get_MassCenter(self, init=False):
        weights = [part.get_partSize() for part in self.furnParts]
        weights = [w/sum(weights) for w in weights]
        return sum([w * part.get_center_point(init) for w, part in zip(weights, self.furnParts)])

    def _create_pillars(self, init=False):
        
        meshes = []
        groundy = self.bodyModel.bodyVerts[:, 1].min() - 0.01
        buttPart = self.furnParts[self._getPartIdx(0)]
        buttCenter = buttPart.get_center_point(init)
        buttProjCenter = np.array([buttCenter[0], groundy, buttCenter[2]])

        # **** ensure mass center inside area ****
        # b1    b2          b
        #  \   /            |
        #    c              c
        #  /   \           /  \
        # lt    rt       lt    rt
        leftThighPart = self.furnParts[self._getPartIdx(1)]
        rightThignPart = self.furnParts[self._getPartIdx(2)]
        lThighCenter = leftThighPart.get_center_point(init)
        lThighProjCenter  = np.array([lThighCenter[0], groundy, lThighCenter[2]])
        rThighCenter = rightThignPart.get_center_point(init)
        rThighProjCenter = np.array([rThighCenter[0], groundy, rThighCenter[2]])

        axis = buttProjCenter - (lThighProjCenter + rThighProjCenter)/2
        massCenter = self._get_MassCenter()
        massProjCenter = np.array([massCenter[0], groundy, massCenter[2]])
        supportPart = None
        numBackPillar = 1
        if (massProjCenter * axis).sum() > (buttProjCenter * axis).sum():
            # the projected center is outof the enclosed triangle.
            supportPart = self.furnParts[self._getPartIdx(3)]
            numBackPillar = 2
        else:
            supportPart = self.furnParts[self._getPartIdx(0)]
        supPartCenter = supportPart.get_center_point(init)
        supPartProjCenter = np.array([supPartCenter[0], groundy, supPartCenter[2]])

        # **** connect thigh parts to ground ****
        frontAreaCenter= (buttProjCenter + lThighProjCenter + rThighProjCenter) / 3
        lAttachPoint = 2 * (lThighProjCenter - frontAreaCenter) + frontAreaCenter
        rAttachPoint = 2 * (rThighProjCenter - frontAreaCenter) + frontAreaCenter
        mesh = create_cylinder_mesh_from_2endPoints(lThighCenter, lAttachPoint, radius=0.03, num_points=20, num_segments=20)
        meshes.append(mesh)
        mesh = create_cylinder_mesh_from_2endPoints(rThighCenter, rAttachPoint, radius=0.03, num_points=20, num_segments=20)
        meshes.append(mesh)

        # **** connect butt/back parts to ground ****
        if numBackPillar==1:
            supAttachpoint = (supPartProjCenter - (lAttachPoint+rAttachPoint)/2) * 0.2 + supPartProjCenter
            mesh = create_cylinder_mesh_from_2endPoints(supPartCenter, supAttachpoint , radius=0.03, num_points=20, num_segments=20)
            meshes.append(mesh)
        elif numBackPillar==2:
            lsupAttachPoint = (frontAreaCenter - lThighProjCenter) + supPartProjCenter
            ratio =  np.linalg.norm(frontAreaCenter - lThighProjCenter) / np.linalg.norm(frontAreaCenter - rThighProjCenter) # make it symmetric.
            rsupAttachPoint = (frontAreaCenter - rThighProjCenter) * ratio + supPartProjCenter
            mesh = create_cylinder_mesh_from_2endPoints(supPartCenter, lsupAttachPoint, radius=0.03, num_points=20, num_segments=20)
            meshes.append(mesh)
            mesh = create_cylinder_mesh_from_2endPoints(supPartCenter, rsupAttachPoint, radius=0.03, num_points=20, num_segments=20)
            meshes.append(mesh)

        return meshes

    def _create_armrods(self, init=False):
        meshes = []
        # ***** Determine candidates attachment point, on back, thighs, butt *****
        leftarmPart = self.furnParts[self._getPartIdx(18)]
        rightarmPart = self.furnParts[self._getPartIdx(19)]

        # only select points at the side.
        l = self.bodyModel.bodyJoints[14]
        r = self.bodyModel.bodyJoints[13]
        lraxis = (r - l) / np.linalg.norm(r - l)
        lraxis[1] = 0
        def edgePointfilter(points):
            points = sorted(points, key=lambda p: (p*lraxis).sum())
            return [points[0], points[-1]]

        attachPoints = []
        leftThighPart = self.furnParts[self._getPartIdx(1)]
        leftThighCenter = leftThighPart.get_center_point(init)
        attachPoints+=[(leftThighCenter, ep) for ep in edgePointfilter(leftThighPart.get_edge_centerPoints())]

        rightThignPart = self.furnParts[self._getPartIdx(2)]
        rightTignCenter = rightThignPart.get_center_point()
        attachPoints+=[(rightTignCenter, ep) for ep in edgePointfilter(rightThignPart.get_edge_centerPoints())]

        buttPart = self.furnParts[self._getPartIdx(0)]
        buttCenter =buttPart.get_center_point()
        attachPoints+=[(buttCenter, ep) for ep in edgePointfilter(buttPart.get_edge_centerPoints())]

        # backPart = self.furnParts[self._getPartIdx(3)]
        # backCenter =buttPart.get_center_point()
        # attachPoints+=[(backCenter, ep) for ep in edgePointfilter(backPart.get_edge_centerPoints())]

        # ***** Modeling of rods *****
        for armPart in [leftarmPart, rightarmPart]:
            start = armPart.get_center_point(init)
            # Connect to closest attach point that is below the arm part.
            closestPoint = min(filter(lambda p:p[1][1]<start[1], attachPoints), key=lambda p: np.linalg.norm(p[1]-start))
            closestCenter = closestPoint[0]
            closestPoint = closestPoint[1]
            partNormal = armPart.bodyPart.avgNormal
            dist = 0.05
            middle1 = start + partNormal * dist
            middle2 = closestCenter + (closestPoint - closestCenter) * 2
            end = closestPoint*0.9 + 0.1*closestCenter - np.array([0, 0.015, 0])
            # Create a cylinder mesh based on control points.
            nodes = np.array([start, middle1, middle2, end]).T
            curve = create_bezier_curve(nodes)
            cylinder_mesh = create_cylinder_mesh_from_bezierCurve(curve, radius=0.02, num_points=20, num_segments=20)
            cylinder_mesh = cylinder_mesh.subdivide_loop(number_of_iterations=2)
            meshes.append(cylinder_mesh)

        return meshes

    #------------------------------Optimization related----------------------------------
    def _optimize(self):
        for schedule in self.config.pipeline:
            if schedule['stage'] == "local":
                if "params" in schedule:
                    for k, v in schedule["params"].items():
                        self.config.__getattr__("localParams")[k] = v
                for furnPart in self.furnParts:
                    furnPart.setup_optimization(schedule["strategy"])
                    furnPart.optimize()

            if schedule['stage'] == "global":
                if "params" in schedule:
                    for k, v in schedule["params"].items():
                        self.config.__getattr__("globalParams")[k] = v
                self.setup_optimization(strategy=schedule["strategy"])
                if len(self.paramsOrder)==0:
                    continue
                self.optimize()

    def _get_filtered_params(self, partIdx, latest=True):
        """
        Takes all params of a single part and return the learnable params.
        """
        part = self.furnParts[partIdx]
        params = part.get_latest_params() if latest else part.get_original_params()
        return part._filter_params(params)
    
    def _get_revert_params(self, params, partIdx, latest=True):
        """
        Takes filtered params of a single part and return all params.
        """
        part = self.furnParts[partIdx]
        return part._revert_params(params, latest=latest)

    def _get_orderIdx(self, id):
        for i, idxc in enumerate(self.paramsOrder):
            if idxc == id:
                return i
        assert False
