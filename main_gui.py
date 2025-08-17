import pdb
import sys
import os
import gc
import argparse
import open3d as o3d # MUST be imported before torch. (Otherwise segmentation fault)
import torch 
torch.set_num_threads(1) # MUST be setted to run over macbook m3. (Otherwise programe got freezed)
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QCheckBox, QDialog, QFileDialog, QLabel, QLineEdit
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtCore import Qt
from src import BodyModel, Furniture, Config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Body Model Analysis GUI")
    parser.add_argument("--config", type=str, default="./configs/baseline.yaml", help="Path to the configuration file")
    parser.add_argument("--smpl_path", type=str, default="./data/smpl", help="Path to the SMPL model directory")
    return parser.parse_args()

args = parse_arguments()
 
class Programe(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Control Panel")
        self.setGeometry(100, 100, 300, 150)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout of the GUI
        layout1 = QVBoxLayout()
        layout1_1 = QHBoxLayout()
        layout1_2 = QHBoxLayout()
        layout1_3 = QHBoxLayout()

        # I/O
        self._loadRun_button = QPushButton("Load & Run")
        self._loadRun_button.clicked.connect(self.load_run)
        self._save_button = QPushButton("Save")
        self._save_button.clicked.connect(self.save)
        txt = QLabel("I/O Settings:")
        layout1.addWidget(txt)
        layout1_1.addWidget(self._loadRun_button)
        layout1_1.addWidget(self._save_button)
        layout1.addLayout(layout1_1)
        # Vis settings:
        self._showBody_checkBox = QCheckBox("Show body")
        self._showBody_checkBox.stateChanged.connect(self.update_rendering)
        self._showFurn_checkBox = QCheckBox("Show furn")
        self._showFurn_checkBox.stateChanged.connect(self.update_rendering)
        txt = QLabel("Visualization Settings:")
        layout1.addWidget(txt)
        layout1_2.addWidget(self._showBody_checkBox)
        layout1_2.addWidget(self._showFurn_checkBox)
        layout1.addLayout(layout1_2)

        self._resetView_button = QPushButton("Show next")
        self._resetView_button.clicked.connect(self.showNextFurn)
        layout1_3.addWidget(self._resetView_button)
        layout1.addLayout(layout1_3)

        central_widget.setLayout(layout1)
        self.stage = 0

        #open3d states:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Open3D Viewer", width=800, height=600)
        self.body = None
        self.furn = None

    def load_run(self):
        file_filter = "All Files (*);;TXT Files (*.txt);;"
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter)
        if not os.path.exists(file_path): 
            return

        if self.body is not None or self.furn is not None:
            del self.body, self.furn
            gc.collect()

        self.body = BodyModel(args.smpl_path)
        angles = self.body.load_params(file_path)
        if angles is not None and angles.shape == (1, 72):
            angles = angles.reshape(24, 3)

        self.furn = Furniture(self.body, config)
        self.update_rendering()
    
    def save(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory to Save Mesh")
        if dir_path:
            if self.body is not None:
                body_output_path = os.path.join(dir_path, "body.obj")
                o3d.io.write_triangle_mesh(body_output_path, self.body.bodyMesh)

            if self.furn is not None:
                for i, part in enumerate(self.furn.furnParts):
                    output_path = os.path.join(dir_path, f"part_{i}.obj")
                    mesh = part.get_open3d_mesh(stage=self.stage)
                    o3d.io.write_triangle_mesh(output_path, mesh)

                connectorMeshes, groundMesh = self.furn.get_connector_meshes()
                for i, connectorMesh in enumerate(connectorMeshes):
                    output_path = os.path.join(dir_path, f"connectorPart_{i}.obj")
                    o3d.io.write_triangle_mesh(output_path, connectorMesh)

                output_path = os.path.join(dir_path, f"ground.obj")
                o3d.io.write_triangle_mesh(output_path, groundMesh)

    def create_groudMesh(self):
        min_bound = np.min(self.body.bodyVerts, axis=0) 
        max_bound = np.max(self.body.bodyVerts, axis=0) 
        ground_size = max(max_bound[0] - min_bound[0], max_bound[2] - min_bound[2]) * 1.5  
        ground_center = [(min_bound[0] + max_bound[0]) / 2, min_bound[1] - 0.01, (min_bound[2] + max_bound[2]) / 2] 
        ground = o3d.geometry.TriangleMesh.create_box(width=ground_size, height=0.001, depth=ground_size)
        ground.translate([ground_center[0] - ground_size / 2, ground_center[1], ground_center[2] - ground_size / 2])
        ground.paint_uniform_color([0.8, 0.8, 0.8])
        return ground

    def update_rendering(self):
        self.vis.clear_geometries()
        if self._showBody_checkBox.isChecked() and self.body is not None:
            self.vis.add_geometry(self.body.bodyMesh)
        # Furn
        if self._showFurn_checkBox.isChecked() and self.furn is not None:
            for part in self.furn.furnParts:
                self.vis.add_geometry(part.get_open3d_mesh(stage=self.stage))
        # Ground
        if self.body is not None:
            self.vis.add_geometry(self.create_groudMesh())

    def showNextFurn(self):
        if self.furn is not None:
            numStages = self.furn.furnParts[0].num_results()
            self.stage = (self.stage + 1) % numStages
            print(f"Show stage: {self.stage}")
            self.update_rendering()

if __name__ == "__main__":
    config = Config(args.config)
    app = QApplication(sys.argv)
    window = Programe()
    window.show()
    window.vis.run()
    window.vis.destroy_window()
    sys.exit(app.exec())