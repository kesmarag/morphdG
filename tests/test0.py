import morphdg

print("Setting up Kokkos Engine...")
engine = morphdg.KokkosManager()

print("Initializing AggMesh...")
mesh = morphdg.AggMesh()

mesh.load("./mesh.dat") 
mesh.random_agglomerate(32, 42)

print("Plotting...")
# mesh.plot()

solver = morphdg.DGSolver()

# Set the coefficients interactively
solver.coeffs.vx = 2.5
solver.coeffs.alpha = 5.0


print(solver.coeffs.vx)
