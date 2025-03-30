import os
import numpy as np
from scipy.interpolate import interp1d

def condfrac(invec):
    """
    Interpolate ion mobility according to the distance to the nearest neighbor atom.
    
    Parameters:
        invec (numpy.ndarray): Array of distances to the nearest neighbor atom.
    
    Returns:
        numpy.ndarray: Interpolated values between 0 and 1 based on the distance.
    """
    # Points on line (min,0) (max,1)
    minr = 1.3
    maxr = 4.1
    slope = 1.0 / (maxr - minr)
    intercept = -minr * slope
    result = slope * invec + intercept
    result[result < 0] = 1e-7
    result[result > 1] = 1.0
    return result

def loadFunc(mesh, F, sig, interpfunction):
    """
    Load conductivity values onto a FEniCS function by interpolating 
    using the provided interpolation function.
    
    Parameters:
        mesh (Mesh): FEniCS mesh.
        F (FunctionSpace): FEniCS function space for the conductivity field.
        sig (Function): FEniCS Function where the conductivity values will be loaded.
        interpfunction (callable): Interpolator function (e.g. from scipy.interpolate.RegularGridInterpolator)
                                   that returns conductivity values given coordinates.
    """
    vec = sig.vector()
    values = vec.get_local()
    dofmap = F.dofmap()
    my_first, my_last = dofmap.ownership_range()

    n = F.dim()
    d = mesh.geometry().dim()
    F_dof_coordinates = F.tabulate_dof_coordinates()
    F_dof_coordinates.resize((n, d))

    unowned = dofmap.local_to_global_unowned()
    dofs = list(filter(lambda dof: dofmap.local_to_global_index(dof) not in unowned,
                       range(my_last - my_first)))
    coords = F_dof_coordinates[dofs]
    values[:] = interpfunction(coords)
    vec.set_local(values)
    vec.apply('insert')

def readbinGrid(name, maskRad=-1):
    """
    Read a binary grid file and return the 3D values along with domain dimensions and grid numbers.
    
    Parameters:
        name (str): Filename of the binary grid.
        maskRad (float, optional): If greater than zero, apply a mask based on distance.
    
    Returns:
        tuple:
            - val3d (numpy.ndarray): 3D grid values.
            - dims (list): Physical dimensions [Lm, Wm, Hm] of the grid.
            - nums (list): Number of grid points [nx, ny, nz] in each dimension.
    """
    if not os.path.isfile(name):
        print(name + " doesn't exist, EXITING")
        exit()
    with open(name, 'rb') as f:
        val1d = np.fromfile(f, dtype=np.float32)

    delta = np.array([val1d[6], val1d[6], val1d[6]])
    assert np.abs(delta[0] - 1) < 1e-5, "WE ASSUME the DELTA IS 1!!!!, DELTA IS NOT 1"
    origin = np.array([val1d[3], val1d[4], val1d[5]])
    shape = (int(np.ceil(val1d[0])), int(np.ceil(val1d[1])), int(np.ceil(val1d[2])))
    val3d = np.reshape(val1d[7:], shape, order='F')
    if maskRad > 0:
        x_ = np.arange(origin[0], origin[0] + shape[0] * delta[0], delta[0])
        y_ = np.arange(origin[1], origin[1] + shape[1] * delta[1], delta[1])
        z_ = np.arange(origin[2], origin[2] + shape[2] * delta[2], delta[2])
        assert len(x_) == val3d.shape[0], "x is wrong size"
        assert len(y_) == val3d.shape[1], "y is wrong size"
        assert len(z_) == val3d.shape[2], "z is wrong size"
        xx, yy, zz = np.meshgrid(x_, y_, z_, indexing='ij')
        msk = xx * xx + yy * yy > maskRad * maskRad
        val3d[msk] = 1e-5

    L = delta[0] * shape[0]
    W = delta[1] * shape[1]
    H = delta[2] * shape[2]
    nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
    Lm = L - delta[0]
    Wm = W - delta[1]
    Hm = H - delta[2]
    return val3d, [Lm, Wm, Hm], [nx, ny, nz]



def get_radius(atom_type, rules, default=1.5):
    """
    Given an atom type and a list of rules, return the corresponding radius.
    If no rule applies, return the default value.

    Parameters:
        atom_type (str): The atom type string.
        rules (list): A list of dicts with keys "prefix" and "radius".
        default (float): The default radius if no rule matches.

    Returns:
        float: The radius for the given atom type.
    """
    for rule in rules:
        if atom_type.startswith(rule["prefix"]):
            return rule["radius"]
    return default




def read_dist_to_conc(csv_file):
    """
    Given a CSV file with two columns (distance in Å, scaled ion concentration),
    return an interpolation function that maps any distance to its corresponding
    scaled ion concentration using cubic interpolation.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        interp_func (function): A function that takes a distance (or array of distances)
                                  and returns the interpolated scaled ion concentration.
    """
    # Load data from CSV; adjust delimiter if needed (default here is comma).
    data = np.loadtxt(csv_file, delimiter=',')

    # Extract distances and concentrations.
    distances = data[:, 0]
    concentrations = data[:, 1]

    # Optional: sort the data by distances (in case they are not sorted)
    sorted_indices = np.argsort(distances)
    distances = distances[sorted_indices]
    concentrations = concentrations[sorted_indices]

    # Create an interpolation function with cubic interpolation.
    # 'fill_value="extrapolate"' allows extrapolation beyond the provided data.
    interp_func = interp1d(distances, concentrations, kind='cubic', fill_value="extrapolate")

    return interp_func
