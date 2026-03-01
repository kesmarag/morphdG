import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np


def _plot_mesh(self):
    """plots the morphdG mesh using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    nodes = self.nodes.reshape(-1, 2)         # [N_nodes, 2]
    triangles = self.triangles.reshape(-1, 3) # [N_tri, 3]
    faces = self.faces.reshape(-1, 4)         # [N_faces, 4]
    bnd_faces = self.bnd_faces.reshape(-1, 4) # [N_bnd, 4]
    
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
        nA, nB = f[2], f[3]
        ax.plot([nodes[nA, 0], nodes[nB, 0]], [nodes[nA, 1], nodes[nB, 1]], color='black', linewidth=1.5, zorder=3)

    # Format and display
    ax.autoscale_view()
    ax.set_aspect('equal')
    plt.axis('off')
    plt.title(f'morphdG Mesh: {self.num_elements} Elements')
    plt.savefig("mesh.png", dpi=300, bbox_inches='tight')
    print("Plot saved successfully to 'mesh.png'")
