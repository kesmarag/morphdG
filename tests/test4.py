import numpy as np
import morphdg as mdg
import os

def main():
    # Create a directory to store the animation frames
    os.makedirs("frames", exist_ok=True)

    print("1. Loading and generating mesh...")
    mesh = mdg.Mesh()
    mesh.load("mesh.dat")   
    
    # We use a bit more polygons here so the advection looks smooth!
    mesh.agglomerate(n_polygons=1024, seed=42) 

    print("2. Setting up DGSolver...")
    # Using P=2 for high-order quadratic curves
    solver = mdg.DGSolver(mesh)


    p_array = 3 * np.ones(mesh.num_elements, dtype=np.int32)
    # p_array[1] = 3 
    solver.update_p_orders(p_array)

    
    # Wind blowing left to right (vx=1.0). 
    # Very low diffusion (K=0.01) so the heat wave stays sharp!
    solver.set_params(
        vx=1.0, vy=1.0, 
        Kxx=0.1, Kyy=0.1, 
        alpha=5.0
    )

    # Left wall is a boiling hot heat source. Everything else is cold.
    solver.set_dirichlet_bc(loc=lambda x, y: x < 0.01, dirichlet_input=50.0)
    solver.set_dirichlet_bc(loc=lambda x, y: x > 0.99, dirichlet_input=0.0)
    solver.set_neumann_bc(loc=lambda x, y: y > 0.99, neumann_input=0.0)
    solver.set_dirichlet_bc(loc=lambda x, y: y < 0.01, dirichlet_input=0.0)


    # solver.set_source("100.0*Kokkos::exp(-15.0*(x-0.5)*(x-0.5) - 15.0*(y-0.5)*(y-0.5))")
    solver.set_source(0.0)
    
    solver.assemble_system()


    # dt = solver.calculate_stable_dt(v_max=0.1, K_max=0.0001, cfl=0.1)

    
    # exit(0)
    print("\n--- Starting RK4 Evolution ---")
    dt = 0.01
    num_steps = 100
    
    for step in range(num_steps):
        # The GPU crunches the 4 RK4 math steps natively
        # current_state = solver.advance_time(dt)
        current_state = solver.advance_time_implicit(dt)
        
        # Render a frame every 10 steps
        if step % 10 == 0:
            print(f" -> Time Step {step:03d} completed. Saving frame...")
            solver.plot_solution(current_state, f"frames/frame_{step:03d}.png", vlim=(0.0, 100.0))

    print("\nSimulation complete! Check the 'frames' folder for the animation.")
    
    # Safely free GPU memory before Python/Kokkos shuts down
    del solver, mesh

if __name__ == "__main__":
    main()
