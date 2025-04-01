import argparse
import numpy as np
from sem_sys_mda import sem_sys
from solver import SEMSolver

def main():
    parser = argparse.ArgumentParser(
        description="Run SEM analysis on an MD trajectory with specified parameters."
    )
    # Positional arguments
    parser.add_argument("psf_file", help="Path to the PSF file.")
    parser.add_argument("dcd_file", help="Path to the concatenated DCD file.")
    # Optional arguments with defaults
    parser.add_argument("--sigma", type=float, default=11.2,
                        help="Conductivity in S/m (default: 11.2).")
    parser.add_argument("--volt", type=float, default=0.15,
                        help="Voltage applied to the system (default: 0.15).")
    parser.add_argument("--xymargin", type=float, default=5.0,
                        help="Margin for x and y dimensions to prevent leakage of current (default: 5.0 Å).")
    parser.add_argument("--zmargin", type=float, default=5.0,
                        help="Margin for z dimension to reduce computation (default: 5.0 Å).")
    parser.add_argument("--str_file", default="radius_mapping.json",
                        help="Path to the JSON file for radius mapping (default: radius_mapping.json).")

    '''
    consider_conc and bulk_conc are the argument for SEM2
    '''

    parser.add_argument("--consider_conc", action="store_true",
                        help="If set, ion concentration correction is considered.")
    parser.add_argument("--bulk_conc", type=float, default=1.0,
                        help="Bulk concentration for ion concentration correction (default: 1.0).")



    parser.add_argument("--o", "--output", dest="output_file", default="output.dat",
                        help="Output file to record all the current values in nA (default: output.dat).")
    
    args = parser.parse_args()
    
    # Create and configure the solver.
    solver_obj = SEMSolver(rel_tol=1e-6, max_it=30000)
    solver = solver_obj.get_solver()

    # Initialize the SEM system using the provided files and parameters.
    system = sem_sys(
        psf_file=args.psf_file,
        dcd_file=args.dcd_file,
        sigma=args.sigma,
        Volts=args.volt,
        xymargin=args.xymargin,
        zmargin=args.zmargin,
        str_file=args.str_file,
        consider_conc=args.consider_conc,
        bulk_conc=args.bulk_conc
    )
    
    # List to store the flux for each frame.
    fluxes = []

    # Loop over all frames in the trajectory.
    for I in range(system.n_frames):
        print(f"Processing frame {I}")
        # Update the conductivity field on the mesh for the current frame.
        system.update_conductivity(I)
        # Solve the system.
        system.u1 = system.solve(solver)
        # Calculate the current (fluxes at boundaries).
        ft, fb = system.calculate_current()
        print(f"Frame {I} - Ionic Flux (nA): {ft}")
        fluxes.append(ft)
    
    # Save all flux values to the specified output file.
    np.savetxt(args.output_file, np.array(fluxes), fmt="%.6f")
    print(f"All ionic flux values saved to {args.output_file}")

if __name__ == '__main__':
    main()

