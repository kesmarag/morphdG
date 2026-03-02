#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

/**
 * AggMesh: A polygonal mesh structure built by agglomerating
 * a dense grid of triangles.
 */
struct AggMesh {
  // --- Background Grid (Fine Data) ---
  std::vector<double> h_nodes;
  std::vector<int> h_triangles; // Flattened [n0, n1, n2, ...]
  std::vector<int>
      tri_neighbors; // [t0_n0, t0_n1, t0_n2, ...] (-1 for boundary)

  // --- Agglomerated Structure (Coarse Data) ---
  std::vector<int> h_t_offsets; // Triangle range for each polygon
  std::vector<double> h_bboxes; // [minx, miny, maxx, maxy] per polygon

  std::vector<int> h_faces; // Flattened [eL, eR, nA, nB, ...]
  std::vector<int> h_bnd_faces;

  // --- Simple Accessors ---
  int num_elements() const { return h_t_offsets.size() - 1; }
  int num_triangles() const { return h_triangles.size() / 3; }
  int num_nodes() const { return h_nodes.size() / 2; }

  // --- Core Workflow Functions ---

  // Load nodes and triangles from our simple .dat format
  void load_base_mesh(const std::string &filename) {
    std::ifstream in(filename);
    if (!in) {
      std::cerr << "Error: Cannot open " << filename << "\n";
      exit(1);
    }

    int n_nodes, n_tri, dummy;
    in >> n_nodes >> n_tri >> dummy;

    h_nodes.resize(n_nodes * 2);
    h_triangles.resize(n_tri * 3);

    for (int i = 0; i < n_nodes * 2; ++i)
      in >> h_nodes[i];
    for (int i = 0; i < n_tri * 3; ++i)
      in >> h_triangles[i];

    // Default: 1 Polygon = 1 Triangle
    h_t_offsets.resize(n_tri + 1);
    std::iota(h_t_offsets.begin(), h_t_offsets.end(), 0);

    std::cout << "initial mesh: " << n_tri << " triangles.\n";
    build_adjacency();

    compute_bboxes();
  }

  void build_adjacency() {
    int n_tri = num_triangles();
    tri_neighbors.assign(n_tri * 3, -1);

    // Edge -> {TriangleID, LocalEdgeIndex}
    std::map<std::pair<int, int>, std::pair<int, int>> edge_map;

    for (int t = 0; t < n_tri; ++t) {
      for (int k = 0; k < 3; ++k) {
        int n1 = h_triangles[3 * t + k];
        int n2 = h_triangles[3 * t + (k + 1) % 3];
        std::pair<int, int> edge =
            (n1 < n2) ? std::make_pair(n1, n2) : std::make_pair(n2, n1);

        if (edge_map.count(edge)) {
          auto neigh = edge_map[edge];
          tri_neighbors[3 * t + k] = neigh.first;
          tri_neighbors[3 * neigh.first + neigh.second] = t;
        } else {
          edge_map[edge] = {t, k};
        }
      }
    }
  }

  void create_faces(const std::vector<int> &tri_to_elem) {
    int n_tri = num_triangles();

    h_faces.clear();
    h_bnd_faces.clear();
    h_faces.reserve(n_tri * 3 * 4);
    h_bnd_faces.reserve(n_tri * 3 * 4);

    std::map<std::pair<int, int>, std::vector<int>> edge_to_tris;

    for (int t = 0; t < n_tri; ++t) {
      for (int i = 0; i < 3; ++i) {
        int n1 = h_triangles[t * 3 + i];
        int n2 = h_triangles[t * 3 + ((i + 1) % 3)];
        int min_n = std::min(n1, n2);
        int max_n = std::max(n1, n2);
        edge_to_tris[{min_n, max_n}].push_back(t);
      }
    }

    int face_count = 0;
    int bnd_face_count = 0;

    for (const auto &pair : edge_to_tris) {
      if (pair.second.size() == 2) {
        // ==========================================
        // INTERNAL FACES
        // ==========================================
        int tri1 = pair.second[0];
        int tri2 = pair.second[1];

        int ePlus = tri_to_elem[tri1];  // T+
        int eMinus = tri_to_elem[tri2]; // T-

        // Only save if it is an interface between DIFFERENT polygons
        if (ePlus != eMinus) {
          h_faces.push_back(ePlus);             // Index 0: T+
          h_faces.push_back(eMinus);            // Index 1: T-
          h_faces.push_back(pair.first.first);  // Index 2: nA
          h_faces.push_back(pair.first.second); // Index 3: nB
          face_count++;
        }
      } else if (pair.second.size() == 1) {
        // ==========================================
        // EXTERNAL BOUNDARY FACES
        // ==========================================
        int tri1 = pair.second[0];
        int ePlus =
            tri_to_elem[tri1]; // T+ is the interior element owning the boundary
        int nA = pair.first.first;
        int nB = pair.first.second;

        // 1. Get face coordinates
        double ax = h_nodes[2 * nA], ay = h_nodes[2 * nA + 1];
        double bx = h_nodes[2 * nB], by = h_nodes[2 * nB + 1];

        // 2. Calculate unoriented normal vector
        double dx = bx - ax;
        double dy = by - ay;
        double h_f = std::sqrt(dx * dx + dy * dy);
        double nx = dy / h_f;
        double ny = -dx / h_f;

        // Ensure normal points OUTWARD from T+
        double cx = 0.5 * (h_bboxes[4 * ePlus + 0] + h_bboxes[4 * ePlus + 2]);
        double cy = 0.5 * (h_bboxes[4 * ePlus + 1] + h_bboxes[4 * ePlus + 3]);
        double face_cx = 0.5 * (ax + bx);
        double face_cy = 0.5 * (ay + by);

        if ((nx * (face_cx - cx) + ny * (face_cy - cy)) < 0.0) {
          nx = -nx;
          ny = -ny;
        }

        // Calculate the Inner Product
        // double inn_prod = vx * nx + vy * ny;
        // int tag = 1; // Default to Wall

        // Tag dynamically based on wind direction
        // if (inn_prod < -1e-6) {
        // tag = 1; // Inflow (Flow entering T+)
        // } else if (inn_prod > 1e-6) {
        // tag = 2; // Outflow (Flow leaving T+)
        // }

        h_bnd_faces.push_back(ePlus); // Index 0: T+
        // h_bnd_faces.push_back(tag);   // Index 1: Boundary Tag
        h_bnd_faces.push_back(nA); // Index 1: nA
        h_bnd_faces.push_back(nB); // Index 2: nB
        bnd_face_count++;
      }
    }

    h_faces.shrink_to_fit();
    h_bnd_faces.shrink_to_fit();

    // std::cout << "Extracted " << face_count << " internal element faces.\n";
    // std::cout << "Extracted " << bnd_face_count << " boundary faces.\n";
  }

  void reorder_by_agglomeration(const std::vector<int> &tri_to_elem,
                                int n_elem) {
    int n_tri = num_triangles();
    std::vector<int> counts(n_elem, 0);
    for (int pid : tri_to_elem)
      counts[pid]++;

    // Update offsets
    h_t_offsets.resize(n_elem + 1);
    h_t_offsets[0] = 0;
    for (int i = 0; i < n_elem; ++i)
      h_t_offsets[i + 1] = h_t_offsets[i] + counts[i];

    // Map: OldIndex -> NewIndex
    std::vector<int> old_to_new(n_tri);
    std::vector<int> write_ptr = h_t_offsets;

    std::vector<int> new_tri(h_triangles.size());
    std::vector<int> new_neigh(tri_neighbors.size());

    // Phase 1: Move geometry and record new locations
    for (int t_old = 0; t_old < n_tri; ++t_old) {
      int pid = tri_to_elem[t_old];
      int t_new = write_ptr[pid]++;
      old_to_new[t_old] = t_new;

      for (int k = 0; k < 3; ++k)
        new_tri[3 * t_new + k] = h_triangles[3 * t_old + k];
    }

    // Phase 2: Update neighbors to point to new indices
    for (int t_old = 0; t_old < n_tri; ++t_old) {
      int t_new = old_to_new[t_old];
      for (int k = 0; k < 3; ++k) {
        int nb_old = tri_neighbors[3 * t_old + k];
        new_neigh[3 * t_new + k] = (nb_old == -1) ? -1 : old_to_new[nb_old];
      }
    }

    h_triangles = new_tri;
    tri_neighbors = new_neigh;
    compute_bboxes();
    // std::cout << "AggMesh reordered: " << n_elem << " elements formed.\n";
  }

  void compute_bboxes() {
    int n_elem = num_elements();
    h_bboxes.assign(n_elem * 4, 0.0);
    for (int p = 0; p < n_elem; ++p) {
      double xmin = 1e9, ymin = 1e9, xmax = -1e9, ymax = -1e9;
      for (int t = h_t_offsets[p]; t < h_t_offsets[p + 1]; ++t) {
        for (int k = 0; k < 3; ++k) {
          double x = h_nodes[h_triangles[3 * t + k] * 2];
          double y = h_nodes[h_triangles[3 * t + k] * 2 + 1];
          if (x < xmin)
            xmin = x;
          if (y < ymin)
            ymin = y;
          if (x > xmax)
            xmax = x;
          if (y > ymax)
            ymax = y;
        }
      }
      h_bboxes[4 * p + 0] = xmin;
      h_bboxes[4 * p + 1] = ymin;
      h_bboxes[4 * p + 2] = xmax;
      h_bboxes[4 * p + 3] = ymax;
    }
  }

  void print_mesh_info(int print_elem = 1) const {
    std::cout << "\n=== AggMesh Info ===\n";

    // Nodes
    std::cout << "\n--- Nodes (" << num_nodes() << ") ---\n";
    for (int i = 0; i < num_nodes(); ++i) {
      std::cout << "Node " << i << ": (" << h_nodes[2 * i] << ", "
                << h_nodes[2 * i + 1] << ")\n";
    }

    // Triangles
    std::cout << "\n--- Triangles (" << num_triangles() << ") ---\n";
    for (int t = 0; t < num_triangles(); ++t) {
      std::cout << "Tri " << t << ": [" << h_triangles[3 * t + 0] << ", "
                << h_triangles[3 * t + 1] << ", " << h_triangles[3 * t + 2]
                << "]\n";
    }

    // Neighbors
    std::cout << "\n--- Triangle Neighbors ---\n";
    for (int t = 0; t < num_triangles(); ++t) {
      std::cout << "Tri " << t << " : [";
      for (int k = 0; k < 3; ++k) {
        int n = tri_neighbors[3 * t + k];
        if (n == -1)
          std::cout << "-"; // Boundary
        else
          std::cout << n;

        if (k < 2)
          std::cout << ", ";
      }
      std::cout << "]\n";
    }

    // Elements
    if (print_elem == 1) {
      std::cout << "\n--- Elements (" << num_elements() << ") ---\n";
      for (int p = 0; p < num_elements(); ++p) {
        int start = h_t_offsets[p];
        int end = h_t_offsets[p + 1];

        std::cout << "Element " << p << " (Size " << (end - start) << "): { ";
        for (int t = start; t < end; ++t) {
          std::cout << t << " ";
        }
        std::cout << "}\n";

        // Print Bounding Box
        std::cout << "  BBox: [" << h_bboxes[4 * p + 0] << ", "
                  << h_bboxes[4 * p + 2] << "] x [" << h_bboxes[4 * p + 1]
                  << ", " << h_bboxes[4 * p + 3] << "]\n";
      }
    }

    // Internal Faces
    int num_faces = h_faces.size() / 4;
    if (num_faces > 0) {
      std::cout << "\n--- Internal Faces (" << num_faces << ") ---\n";
      for (int f = 0; f < num_faces; ++f) {
        int ePlus = h_faces[4 * f + 0];
        int eMinus = h_faces[4 * f + 1];
        int nA = h_faces[4 * f + 2];
        int nB = h_faces[4 * f + 3];
        std::cout << "Face " << f << ": Element T+ (" << ePlus
                  << ") | Element T- (" << eMinus << ") | Nodes [" << nA << ", "
                  << nB << "]\n";
      }
    }

    // Boundary Faces
    int num_bnd_faces = h_bnd_faces.size() / 3;
    if (num_bnd_faces > 0) {
      std::cout << "\n--- Boundary Faces (" << num_bnd_faces << ") ---\n";
      for (int f = 0; f < num_bnd_faces; ++f) {
        int ePlus = h_bnd_faces[3 * f + 0];
        // int tag = h_bnd_faces[3 * f + 1];
        int nA = h_bnd_faces[3 * f + 1];
        int nB = h_bnd_faces[3 * f + 2];

        // std::string tag_name;
        // if (tag == 1)
        //   tag_name = "Inflow ";
        // else if (tag == 2)
        //   tag_name = "Outflow";
        // else
        //   tag_name = "Wall   ";

        // std::cout << "Face " << f << ": Element (" << ePlus
        //           << ") | Type: " << tag_name << " | Nodes [" << nA << ", "
        //           << nB << "]\n";
      }
    }

    std::cout << "==========================\n";
  }

  // --- 5. Visualization Export ---
  // void export_python_plot(const std::string &filename = "plot_mesh.py") {
  //   std::ofstream out(filename);
  //   if (!out) {
  //     std::cerr << "Error writing export file.\n";
  //     return;
  //   }

  //   out << "import matplotlib.pyplot as plt\n";
  //   out << "from matplotlib.patches import Polygon\n";
  //   out << "from matplotlib.collections import PatchCollection\n";
  //   out << "import numpy as np\n\n";

  //   out << "fig, ax = plt.subplots(figsize=(8, 8))\n";

  //   // 1. Export Fine Triangles (Thin Gray Lines, No Fill)
  //   out << "# Draw Fine Triangles\n";
  //   out << "patches = []\n";
  //   int n_tris = num_triangles();
  //   for (int t = 0; t < n_tris; ++t) {
  //     double x[3], y[3];
  //     for (int k = 0; k < 3; ++k) {
  //       int node_idx = h_triangles[3 * t + k];
  //       x[k] = h_nodes[2 * node_idx];
  //       y[k] = h_nodes[2 * node_idx + 1];
  //     }
  //     out << "patches.append(Polygon([[" << x[0] << "," << y[0] << "], ["
  //         << x[1] << "," << y[1] << "], [" << x[2] << "," << y[2]
  //         << "]], closed=True))\n";
  //   }
  //   out << "p = PatchCollection(patches, facecolor='none', edgecolors='gray',
  //   "
  //          "linewidths=0.5, zorder=1)\n";
  //   out << "ax.add_collection(p)\n\n";

  //   // 2. Export Coarse Polygon Boundaries (Bold Black Lines)
  //   out << "# Draw Internal Polygon Faces\n";
  //   int num_faces = h_faces.size() / 4;
  //   for (int f = 0; f < num_faces; ++f) {
  //     int nA = h_faces[4 * f + 2];
  //     int nB = h_faces[4 * f + 3];
  //     double ax_coord = h_nodes[2 * nA], ay_coord = h_nodes[2 * nA + 1];
  //     double bx_coord = h_nodes[2 * nB], by_coord = h_nodes[2 * nB + 1];
  //     out << "ax.plot([" << ax_coord << ", " << bx_coord << "], [" <<
  //     ay_coord
  //         << ", " << by_coord << "], color='black', linewidth=2.5,
  //         zorder=3)\n";
  //   }

  //   out << "\n# Draw External Boundary Faces\n";
  //   int num_bnd_faces = h_bnd_faces.size() / 4;
  //   for (int f = 0; f < num_bnd_faces; ++f) {
  //     int nA = h_bnd_faces[4 * f + 2];
  //     int nB = h_bnd_faces[4 * f + 3];
  //     double ax_coord = h_nodes[2 * nA], ay_coord = h_nodes[2 * nA + 1];
  //     double bx_coord = h_nodes[2 * nB], by_coord = h_nodes[2 * nB + 1];
  //     out << "ax.plot([" << ax_coord << ", " << bx_coord << "], [" <<
  //     ay_coord
  //         << ", " << by_coord << "], color='black', linewidth=2.5,
  //         zorder=3)\n";
  //   }

  //   // 3. Draw Labels (Polygon IDs at Centroids)
  //   out << "\n# Draw Polygon ID Labels\n";
  //   int n_polys = num_elements();
  //   for (int p = 0; p < n_polys; ++p) {
  //     int start = h_t_offsets[p];
  //     int end = h_t_offsets[p + 1];

  //     double cx = 0.0, cy = 0.0;
  //     double area_sum = 0.0;

  //     for (int t = start; t < end; ++t) {
  //       for (int k = 0; k < 3; ++k) {
  //         int node_idx = h_triangles[3 * t + k];
  //         cx += h_nodes[2 * node_idx];
  //         cy += h_nodes[2 * node_idx + 1];
  //       }
  //       area_sum += 3.0;
  //     }

  //     if (area_sum > 0) {
  //       cx /= area_sum;
  //       cy /= area_sum;
  //     }

  //     // Plot the ID text
  //     out << "ax.text(" << cx << ", " << cy << ", '" << p
  //         << "', color='darkred', fontsize=14, ha='center', va='center', "
  //            "weight='bold', zorder=5)\n";
  //   }

  //   // 4. Finalize Plot
  //   out << "\n# Auto-scale and Show\n";
  //   out << "ax.autoscale_view()\n";
  //   out << "ax.set_aspect('equal')\n";
  //   out << "plt.axis('off')\n";
  //   out << "plt.title('Discontinuous Galerkin Agglomerated Mesh')\n";
  //   out << "plt.savefig('mesh.png')\n";
  //   out.close();
  // }

  void generate_random_polygons(int q_elem, int seed) {
    std::mt19937 gen(seed);
    int n_tri = num_triangles();

    // 1. Initialize Map
    std::vector<int> tri_to_elem(n_tri, -1);
    std::vector<std::vector<int>> frontiers(q_elem);
    int created_elems = 0;

    // 2. Pick Random Seeds (Initial Triangles)
    if (q_elem > n_tri)
      q_elem = n_tri;

    std::uniform_int_distribution<> dis(0, n_tri - 1);
    while (created_elems < q_elem) {
      int r = dis(gen);
      if (tri_to_elem[r] == -1) {
        tri_to_elem[r] = created_elems;
        // Add valid neighbors to frontier
        for (int k = 0; k < 3; ++k) {
          int n = tri_neighbors[3 * r + k];
          if (n != -1)
            frontiers[created_elems].push_back(n);
        }
        created_elems++;
      }
    }

    // 3. Grow Polygons (Round-Robin Frontier Expansion)
    int unassigned_count = n_tri - q_elem;
    bool progress = true;

    while (unassigned_count > 0 && progress) {
      progress = false;
      for (int p = 0; p < q_elem; ++p) {
        if (frontiers[p].empty())
          continue;

        // Pick random candidate from frontier
        std::uniform_int_distribution<> dis_front(0, frontiers[p].size() - 1);
        int idx = dis_front(gen);
        int candidate = frontiers[p][idx];

        // Remove from frontier (swap with back for O(1) removal)
        frontiers[p][idx] = frontiers[p].back();
        frontiers[p].pop_back();

        // If still available, claim it
        if (tri_to_elem[candidate] == -1) {
          tri_to_elem[candidate] = p;
          unassigned_count--;
          progress = true;

          // Add its neighbors to frontier
          for (int k = 0; k < 3; ++k) {
            int n = tri_neighbors[3 * candidate + k];
            if (n != -1 && tri_to_elem[n] == -1) {
              frontiers[p].push_back(n);
            }
          }
        }
      }
    }

    // 4. Robust Cleanup (Iterative Sweep) [FIXED]
    // We keep sweeping until no more triangles change.
    // This fills "islands" that might have been missed by a single pass.
    bool changed = true;
    while (changed) {
      changed = false;
      for (int t = 0; t < n_tri; ++t) {
        if (tri_to_elem[t] == -1) {
          // Check all neighbors
          for (int k = 0; k < 3; ++k) {
            int n = tri_neighbors[3 * t + k];
            // If neighbor has a valid poly, join it
            if (n != -1 && tri_to_elem[n] != -1) {
              tri_to_elem[t] = tri_to_elem[n];
              changed = true;
              break; // Move to next triangle once assigned
            }
          }
        }
      }
    }

    // Final Failsafe: If any triangles are STILL -1 (totally isolated
    // component), force assign them to Polygon 0 to prevent memory
    // corruption.
    int stranded_count = 0;
    for (int t = 0; t < n_tri; ++t) {
      if (tri_to_elem[t] == -1) {
        tri_to_elem[t] = 0;
        stranded_count++;
      }
    }

    if (stranded_count > 0) {
      std::cout << "Warning: Force-assigned " << stranded_count
                << " stranded triangles to Poly 0.\n";
    }

    // 5. Commit changes
    std::cout << "Generated " << q_elem << " random polygons (Seed: " << seed
              << ")\n";

    create_faces(tri_to_elem);

    reorder_by_agglomeration(tri_to_elem, q_elem);
  }

  // ==============================================================================
  // KOKKOS DEVICE MEMORY (GPU/OpenMP)
  // ==============================================================================
  Kokkos::View<double **, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_nodes;
  Kokkos::View<int **, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_triangles;
  Kokkos::View<int *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_faces;
  Kokkos::View<int *, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_t_offsets;
  Kokkos::View<double **, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace::memory_space>
      d_bboxes;

  void push_to_device() {
    int n_nodes = num_nodes();
    int n_tris = num_triangles();
    int n_polys = num_elements();

    // 1. Allocate Device Memory (Forcing exact type match using decltype)
    d_nodes = decltype(d_nodes)("d_nodes", n_nodes, 2);
    d_triangles = decltype(d_triangles)("d_triangles", n_tris, 3);
    d_faces = decltype(d_faces)("d_faces", h_faces.size());
    auto mirror_faces = Kokkos::create_mirror_view(d_faces);
    // Populate the mirror
    for (size_t i = 0; i < h_faces.size(); ++i) {
      mirror_faces(i) = h_faces[i];
    }

    d_t_offsets = decltype(d_t_offsets)("d_t_offsets", h_t_offsets.size());
    d_bboxes = decltype(d_bboxes)("d_bboxes", n_polys, 4);

    // 2. Create Host Mirrors
    auto mirror_nodes = Kokkos::create_mirror_view(d_nodes);
    auto mirror_tris = Kokkos::create_mirror_view(d_triangles);
    auto mirror_toff = Kokkos::create_mirror_view(d_t_offsets);
    auto mirror_bb = Kokkos::create_mirror_view(d_bboxes);

    // 3. Populate Mirrors from standard std::vectors
    for (int i = 0; i < n_nodes; ++i) {
      mirror_nodes(i, 0) = h_nodes[2 * i + 0];
      mirror_nodes(i, 1) = h_nodes[2 * i + 1];
    }
    for (int i = 0; i < n_tris; ++i) {
      mirror_tris(i, 0) = h_triangles[3 * i + 0];
      mirror_tris(i, 1) = h_triangles[3 * i + 1];
      mirror_tris(i, 2) = h_triangles[3 * i + 2];
    }
    for (size_t i = 0; i < h_t_offsets.size(); ++i) {
      mirror_toff(i) = h_t_offsets[i];
    }
    for (int i = 0; i < n_polys; ++i) {
      mirror_bb(i, 0) = h_bboxes[4 * i + 0];
      mirror_bb(i, 1) = h_bboxes[4 * i + 1];
      mirror_bb(i, 2) = h_bboxes[4 * i + 2];
      mirror_bb(i, 3) = h_bboxes[4 * i + 3];
    }

    // 4. Deep copy the Host Mirrors to the Device Views
    Kokkos::deep_copy(d_nodes, mirror_nodes);
    Kokkos::deep_copy(d_triangles, mirror_tris);
    Kokkos::deep_copy(d_faces, mirror_faces);
    Kokkos::deep_copy(d_t_offsets, mirror_toff);
    Kokkos::deep_copy(d_bboxes, mirror_bb);
  }
};
