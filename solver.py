# solver.py
from fenics import KrylovSolver, set_log_level

class SEMSolver:
    """
    A wrapper for configuring and returning a KrylovSolver for the SEM system.
    """
    def __init__(self, method="gmres", preconditioner="amg", rel_tol=1e-6, max_it=30000, log_level=30):
        self.solver = KrylovSolver(method, preconditioner)
        self.solver.parameters["relative_tolerance"] = rel_tol
        self.solver.parameters["maximum_iterations"] = max_it
        set_log_level(log_level)
    
    def get_solver(self):
        """
        Return the configured KrylovSolver.
        """
        return self.solver

