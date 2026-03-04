import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

from . import morphdg_core

class Mesh:
    def __init__(self, filename=None):
        self._cpp_mesh = morphdg_core.AggMesh()
        if filename:
            self.load(filename)
            
    def load(self, filename):
        self._cpp_mesh.load(filename)
        
    def agglomerate(self, n_polygons, seed=42):
        self._cpp_mesh.random_agglomerate(n_polygons, seed)
        self._cpp_mesh.push_to_device()
        
    @property
    def num_elements(self):
        return self._cpp_mesh.num_elements

    def plot(self, filename="mesh.png"):
        fig, ax = plt.subplots(figsize=(8, 8))
        nodes = self._cpp_mesh.nodes.reshape(-1, 2)
        triangles = self._cpp_mesh.triangles.reshape(-1, 3)
        faces = self._cpp_mesh.faces.reshape(-1, 4)
        bnd_faces = self._cpp_mesh.bnd_faces.reshape(-1, 3)
        
        # Draw Triangles
        patches = [Polygon(nodes[tri], closed=True) for tri in triangles]
        p = PatchCollection(patches, facecolor='none', edgecolors='gray', linewidths=0.5, zorder=1)
        ax.add_collection(p)
        
        # Draw Internal Faces
        for f in faces:
            nA, nB = f[2], f[3]
            ax.plot([nodes[nA, 0], nodes[nB, 0]], [nodes[nA, 1], nodes[nB, 1]], color='black', linewidth=1.5, zorder=3)
            
        # Draw Boundary Faces 
        for f in bnd_faces:
            nA, nB = f[1], f[2]
            ax.plot([nodes[nA, 0], nodes[nB, 0]], [nodes[nA, 1], nodes[nB, 1]], color='black', linewidth=1.5, zorder=3)

        ax.autoscale_view()
        ax.set_aspect('equal')
        plt.axis('off')
        plt.title(f'morphdG Mesh: {self.num_elements} Elements')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to '{filename}'")
        plt.close()
