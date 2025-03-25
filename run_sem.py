# run_sem.py
import sys
import numpy as np

# Import our SEM system class and our solver wrapper.
from sem_sys import sem_sys
from solver import SEMSolver

# Set the FEniCS log level to suppress lower-priority messages.
set_log_level(30)

if __name__ == '__main__':
    # The first argument is the comfile listing grids.
    comfile = sys.argv[1]
    
    infiles = []
    outfiles = []
    
    # Read grid file names and set output file names.
    with open(comfile, 'r') as infile:
        count = 0
        for line in infile:
            count += 1
            if count % 2 == 0:
                infiles.append(line.split()[0] + ".bin")
                outfiles.append(f'{line.split()[0]}.2.dat')
    
    # Create and configure the solver.
    solver = SEMSolver(rel_tol=1e-6, max_it=30000)
    
    # Initialize the system using the first grid file.
    system = sem_sys(infiles[0])
    
    for I in range(len(infiles)):
        # Update the conductivity field on the mesh.
        system.update_conductivity(infiles[I])
    
        # Solve the system.
        system.u1 = system.solve(solver)
    
        # Calculate the current (fluxes at boundaries).
        ft, fb = system.calculate_current()
        print("Flux through top boundary:", ft)
        print("Flux through bottom boundary:", fb)
    
        # Save the top boundary flux to an output file.
        np.savetxt(f'{outfiles[I]}', np.array([ft]))

