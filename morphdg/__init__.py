import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

# Import the compiled C++ engine
from . import morphdg_core

# Expose classes to the user
KokkosManager = morphdg_core.KokkosManager
AggMesh = morphdg_core.AggMesh

def _plot_mesh(self):
    """Natively plots the C++ morphdG mesh using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Grab the C++ memory and shape it perfectly!
    nodes = self.nodes.reshape(-1, 2)         # [N_nodes, 2]
    triangles = self.triangles.reshape(-1, 3) # [N_tri, 3]
    faces = self.faces.reshape(-1, 4)         # [N_faces, 4]
    bnd_faces = self.bnd_faces.reshape(-1, 4) # [N_bnd, 4]
    
    # 2. Draw Fine Triangles (Gray lines)
    patches = [Polygon(nodes[tri], closed=True) for tri in triangles]
    p = PatchCollection(patches, facecolor='none', edgecolors='gray', linewidths=0.5, zorder=1)
    ax.add_collection(p)
    
    # 3. Draw Internal Faces (Black lines)
    for f in faces:
        nA, nB = f[2], f[3]
        ax.plot([nodes[nA, 0], nodes[nB, 0]], [nodes[nA, 1], nodes[nB, 1]], color='black', linewidth=2.5, zorder=3)
        
    # 4. Draw Boundary Faces (Black lines)
    for f in bnd_faces:
        nA, nB = f[2], f[3]
        ax.plot([nodes[nA, 0], nodes[nB, 0]], [nodes[nA, 1], nodes[nB, 1]], color='black', linewidth=2.5, zorder=3)

    # 5. Draw Labels (Polygon IDs at Centroids)
    for p_idx in range(self.num_elements):
        start = self.t_offsets[p_idx]
        end = self.t_offsets[p_idx + 1]
        
        # Get all nodes belonging to the triangles in this polygon
        poly_tris = triangles[start:end]
        poly_nodes = nodes[poly_tris]
        
        # Calculate centroid natively in Numpy
        cx = np.mean(poly_nodes[:, :, 0])
        cy = np.mean(poly_nodes[:, :, 1])
        
        # ax.text(cx, cy, str(p_idx), color='darkred', fontsize=14, ha='center', va='center', weight='bold', zorder=5)

    # 6. Format and display
    ax.autoscale_view()
    ax.set_aspect('equal')
    plt.axis('off')
    plt.title(f'morphdG Mesh: {self.num_elements} Elements')
    plt.savefig("mesh.png", dpi=300, bbox_inches='tight')
    print("Plot saved successfully to 'mesh.png'")

# Bind the Python plot method to the C++ class
AggMesh.plot = _plot_mesh
