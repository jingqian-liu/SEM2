# sem_sys.py

# Standard imports
import os
import numpy as np
from pathlib import Path
import MDAnalysis as mda

# FEniCS imports
from fenics import *
from fenics import ds, dx, assemble_system, Constant, TrialFunction, TestFunction, Function, BoxMesh, FunctionSpace, MeshFunction, Point, lhs, rhs, dot, grad, nabla_grad, assemble

# Scipy import for interpolation
from scipy.interpolate import RegularGridInterpolator

# Import cKDTree for fast nearest-neighbor search
from scipy.spatial import cKDTree

# Import helper functions from utils.py
from utils import condfrac, loadFunc, readbinGrid

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        # 'sizez' should be attached externally
        return on_boundary and near(x[2], self.sizez/2.)

class BotBoundary(SubDomain):
    def inside(self, x, on_boundary):
        # 'sizez' should be attached externally
        return on_boundary and near(x[2], -self.sizez/2.)

# The main class that encapsulates the SEM system
class sem_sys:
    def __init__(self, psf_file, dcd_file, sigma=1.12, Volts=0.15, margin=5.0):
        """
        Initialize the SEM system by reading the trajectory, setting up the mesh,
        function spaces, and boundary conditions.
        
        Parameters:
            psf_file (str): Path to the PSF file.
            dcd_file (str): Path to the DCD file.
            sigma (float): Conductivity scaling parameter.
            Volts (float): Voltage applied at the top boundary.
            margin (float): Margin to adjust the system boundaries.
        """
        self.sigma = sigma
        self.Volts = Volts

        # Load the MD trajectory and store it as 'universe'
        self.universe = mda.Universe(psf_file, dcd_file)
        self.n_frames = len(self.universe.trajectory)
        self.universe.trajectory[0]  # Set the trajectory to the first frame

        # Extract atom positions including water (in Å) from the entire system.
        all_positions = self.universe.atoms.positions

        # Determine the extents of the system from atom coordinates
        x_min, y_min, z_min = np.min(all_positions, axis=0)
        x_max, y_max, z_max = np.max(all_positions, axis=0)

        # Adjust extents by margin
        x_min += margin
        y_min += margin
        x_max -= margin
        y_max -= margin

        # With a resolution of 1Å, use np.ceil to cover the entire system.
        self.nx = int(np.ceil(x_max - x_min)) + 1
        self.ny = int(np.ceil(y_max - y_min)) + 1
        self.nz = int(np.ceil(z_max - z_min)) + 1

        # Define the physical dimensions of the grid (n-1 because spacing is between points)
        self.Lm, self.Wm, self.Hm = self.nx - 1.0, self.ny - 1.0, self.nz - 1.0

        # Adjust sizes for mesh construction (if needed)
        self.sizex = self.Lm - 5
        self.sizey = self.Wm - 5
        self.sizez = self.Hm
        self.numx = int(self.sizex)
        self.numy = int(self.sizey)
        self.numz = int(self.sizez)

        # Create coordinate arrays for the grid points with 1Å spacing.
        x_coords = np.arange(x_min, x_min + self.nx, 1)
        y_coords = np.arange(y_min, y_min + self.ny, 1)
        z_coords = np.arange(z_min, z_min + self.nz, 1)

        # Generate a 3D meshgrid of points with 'ij' indexing.
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        self.grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Create the FEniCS mesh and define function spaces
        self.mesh = BoxMesh(Point(-self.sizex/2., -self.sizey/2., -self.sizez/2.),
                            Point(self.sizex/2., self.sizey/2., self.sizez/2.),
                            self.numx, self.numy, self.numz)
        self.V = FunctionSpace(self.mesh, 'P', 1)
        self.F = FunctionSpace(self.mesh, 'CG', 1)

        # Setup boundary conditions:
        # Attach the domain size to the boundary objects
        self.top_bndr = TopBoundary()
        self.top_bndr.sizez = self.sizez
        self.bot_bndr = BotBoundary()
        self.bot_bndr.sizez = self.sizez

        self.VTop = DirichletBC(self.V, Constant(self.Volts), self.top_bndr)
        self.VBot = DirichletBC(self.V, Constant(0), self.bot_bndr)

        # Mark boundaries for current calculation
        self.boundary_parts = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.top_bndr.mark(self.boundary_parts, 1)
        self.bot_bndr.mark(self.boundary_parts, 2)

        # Create FEniCS functions for solution and conductivity.
        # Rename the trial function to avoid conflict with the Universe.
        self.u_trial = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.sig = Function(self.F)
        self.u1 = Function(self.V)
        # For storing the weak form's RHS (here zero)
        self.l = Constant(0.0) * self.v * dx

    def update_conductivity(self, frame_idx):
        """
        Update the conductivity field 'sig' using an interpolator based on the current frame.
        
        Parameters:
            frame_idx (int): The index of the trajectory frame to use.
        """
        # Set the universe to the specified frame.
        self.universe.trajectory[frame_idx]

        # Select atoms that are not water or ions.
        # Adjust the selection string according to your system's residue names.
        sel = self.universe.select_atoms("not (resname TIP3 WAT CLA POT SOD)")
        positions = sel.positions  # positions of the selected atoms

        # Build a KDTree from the selected atom positions for fast nearest-neighbor lookup.
        tree = cKDTree(positions)

        # Query the KDTree: for each grid point, find the distance to the nearest atom.
        distances, _ = tree.query(self.grid_points)

        # Cap any distance larger than 5Å to 5Å.
        distances[distances > 5.0] = 5.0

        # Reshape the distances back into a 3D array with shape (nx, ny, nz)
        distance_grid = distances.reshape((self.nx, self.ny, self.nz))

        # Compute conductivity scaling based on the grid distances.
        calcSig = self.sigma * condfrac(distance_grid)
        interpfunction = RegularGridInterpolator(
            (np.linspace(-self.Lm/2., self.Lm/2., num=self.nx),
             np.linspace(-self.Wm/2., self.Wm/2., num=self.ny),
             np.linspace(-self.Hm/2., self.Hm/2., num=self.nz)),
            calcSig, bounds_error=False, fill_value=self.sigma)
        loadFunc(self.mesh, self.F, self.sig, interpfunction)

    def solve(self, solver):
        """
        Set up the variational form and solve the system.
        Assumes that self.sig has been updated (e.g. via update_conductivity).
        
        Parameters:
            solver: A configured KrylovSolver (or similar) to solve the linear system.
            
        Returns:
            Function: The computed solution stored in self.u1.
        """
        # Build the variational form: a(u,v) = (sig * dot(grad(u), grad(v))) dx
        FF = self.sig * dot(grad(self.u_trial), grad(self.v)) * dx
        a_form, _ = lhs(FF), rhs(FF)

        # Assemble system with boundary conditions applied
        A, bb = assemble_system(a_form, self.l, [self.VTop, self.VBot])
        solver.set_operator(A)
        solver.solve(self.u1.vector(), bb)
        return self.u1

    def calculate_current(self):
        """
        Calculate the current through the top and bottom boundaries.
        
        Returns:
            tuple: (ft, fb) where ft is the flux (current) through the top boundary,
                   and fb is the flux through the bottom boundary.
        """
        ds_measure = ds(subdomain_data=self.boundary_parts)
        flux_top = dot(Constant((0, 0, 1)), self.sig * nabla_grad(self.u1)) * ds_measure(1)
        flux_bot = dot(Constant((0, 0, 1)), self.sig * nabla_grad(self.u1)) * ds_measure(2)
        ft = assemble(flux_top)
        fb = assemble(flux_bot)
        return ft, fb

