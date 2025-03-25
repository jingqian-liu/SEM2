# sem_sys.py

# Standard imports
import os
import numpy as np
from pathlib import Path

# FEniCS imports
from fenics import *
from fenics import ds, dx, assemble_system, Constant, TrialFunction, TestFunction, Function, BoxMesh, FunctionSpace, MeshFunction, Point, lhs, rhs, dot, grad, nabla_grad, assemble

# Scipy import for interpolation
from scipy.interpolate import RegularGridInterpolator

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
    def __init__(self, input_binfile, sigma=1.12, Volts=0.15):
        """
        Initialize the SEM system by reading a binary grid file, setting up the mesh,
        function spaces, and boundary conditions.
        
        Parameters:
            input_binfile (str): Path to the binary grid file.
            sigma (float): Conductivity scaling parameter.
            Volts (float): Voltage applied at the top boundary.
        """
        self.sigma = sigma
        self.Volts = Volts

        # Read first grid file to set up domain sizes
        val3d, dims, nums = readbinGrid(input_binfile)
        self.Lm, self.Wm, self.Hm = dims
        self.nx, self.ny, self.nz = nums

        print("Domain dimensions:", dims, "Grid points:", nums)

        # Adjust sizes if needed
        self.sizex = self.Lm - 5
        self.sizey = self.Wm - 5
        self.sizez = self.Hm
        self.numx = int(self.sizex)
        self.numy = int(self.sizey)
        self.numz = int(self.sizez)

        # Create the mesh and define function spaces
        self.mesh = BoxMesh(Point(-self.sizex/2., -self.sizey/2., -self.sizez/2.),
                            Point(self.sizex/2., self.sizey/2., self.sizez/2.),
                            self.numx, self.numy, self.numz)
        self.V = FunctionSpace(self.mesh, 'P', 1)
        self.F = FunctionSpace(self.mesh, 'CG', 1)

        # Setup boundary conditions:
        # (Attach the domain size to the boundary objects for use in near() calls)
        self.top_bndr = TopBoundary()
        self.top_bndr.sizez = self.sizez  # pass sizez for use in boundary test
        self.bot_bndr = BotBoundary()
        self.bot_bndr.sizez = self.sizez

        self.VTop = DirichletBC(self.V, Constant(self.Volts), self.top_bndr)
        self.VBot = DirichletBC(self.V, Constant(0), self.bot_bndr)

        # Mark boundaries for current calculation
        self.boundary_parts = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.top_bndr.mark(self.boundary_parts, 1)
        self.bot_bndr.mark(self.boundary_parts, 2)

        # Pre-create functions for solution and conductivity
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.sig = Function(self.F)
        self.u1 = Function(self.V)
        # For storing the weak form's RHS (here zero)
        self.l = Constant(0.0) * self.v * dx

    def update_conductivity(self, input_binfile):
        """
        Update the conductivity field 'sig' using an interpolator based on a new binary grid.
        
        Parameters:
            input_binfile (str): Path to the new binary grid file.
        """
        val3d, _, _ = readbinGrid(input_binfile)
        calcSig = self.sigma * condfrac(val3d)
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
        FF = self.sig * dot(grad(self.u), grad(self.v)) * dx
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

