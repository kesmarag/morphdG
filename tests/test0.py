import morphdg

print("Setting up Kokkos Engine...")
engine = morphdg.KokkosManager()

print("Initializing AggMesh...")
mesh = morphdg.AggMesh()

mesh.load_base_mesh("./mesh.dat") 
mesh.generate_random_polygons(15, 42)

print("Plotting natively...")
mesh.plot()
