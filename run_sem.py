# run_sem.py
import sys
import numpy as np

# Import our SEM system class and our solver wrapper.
from sem_sys_mda import sem_sys
from solver import SEMSolver

if __name__ == '__main__':
    
    # Create and configure the solver.
    solver_obj = SEMSolver(rel_tol=1e-6, max_it=30000)
    solver = solver_obj.get_solver()
    
    # Initialize the system using the first grid file.
    system = sem_sys("wrap.psf", "wrap.dcd", sigma=1.12, Volts=0.15, margin = 38.0)
    
    for I in range(system.n_frames):
        # Update the conductivity field on the mesh.
        system.update_conductivity(I)
    
        # Solve the system.
        system.u1 = system.solve(solver)
    
        # Calculate the current (fluxes at boundaries).
        ft, fb = system.calculate_current()
        print("Flux through top boundary:", ft)
        print("Flux through bottom boundary:", fb)
    
