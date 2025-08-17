import os
import open3d as o3d # MUST be imported before torch. (Otherwise segmentation fault on mac m3)
import torch 
torch.set_num_threads(1) 
from src import BodyModel, Furniture, Config


if __name__ == "__main__":
    input_dir = "./data/input"
    input_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file)) \
                    and (file.endswith('.txt') or file.endswith('.pkl'))]
    config = Config("./configs/baseline.yaml")

    for input_file in input_files:
        print("---------------------------------")
        print("Start optimizing: ", input_file)
        # Start the Experiment
        import time
        tic = time.time()
        body = BodyModel('./data/smpl', config = config.bodyModelParams)
        body.load_params(input_file)
        furn = Furniture(body, config)
        toc = time.time()
        print(f"Time elapsed: {toc-tic} s")

        # Save the config
        expName = config.expName
        intputName = os.path.basename(input_file).split(".")[0]
        output_dir = "./data/results"
        os.makedirs(os.path.join(output_dir, intputName, expName), exist_ok=True)
        output_config_path = os.path.join(output_dir, intputName, expName, "config.yaml")
        config.save(output_config_path)

        # Save the body mesh
        body_output_path = os.path.join(output_dir, intputName, expName, "body.obj")
        o3d.io.write_triangle_mesh(body_output_path, body.bodyMesh)

        # Save the supporting parts mesh
        for i, part in enumerate(furn.furnParts):
            output_path = os.path.join(output_dir, intputName, expName, f"part_{i}.obj")
            mesh = part.get_open3d_mesh()
            o3d.io.write_triangle_mesh(output_path, mesh)

        # Save the rods and pillars.
        connectorMeshes, groundMesh = furn.get_connector_meshes()
        for i, connectorMesh in enumerate(connectorMeshes):
            output_path = os.path.join(output_dir, intputName, expName, f"connectorPart_{i}.obj")
            o3d.io.write_triangle_mesh(output_path, connectorMesh)

        # Save the ground
        output_path = os.path.join(output_dir, intputName, expName, f"ground.obj")
        o3d.io.write_triangle_mesh(output_path, groundMesh)
