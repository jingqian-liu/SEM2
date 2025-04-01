import os
import numpy as np
from scipy.interpolate import interp1d
import MDAnalysis as mda
import matplotlib.pyplot as plt

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




def compute_avg_ion_conc_map(universe, grid_points, fine_grid_shape, resolution=1.0, coarse_resolution=3.0):
    """
    Compute an averaged ion concentration map over the trajectory on a coarse grid 
    (e.g. 3 Å resolution), then upsample to the high-resolution grid (e.g. 1 Å resolution).
    Handles ions that fall just outside the coarse grid (within one fine grid cell).
    
    Parameters:
        universe (MDAnalysis.Universe): The MD trajectory.
        grid_points (np.ndarray): Array of shape (N, 3) with coordinates of each fine grid point.
        fine_grid_shape (tuple): The fine grid dimensions (nx, ny, nz) corresponding to resolution (default=1 Å).
        ion_selection (str): MDAnalysis selection string for the ions (e.g. "resname NA or resname CL").
        resolution (float): The fine grid resolution in Å (default 1.0).
        coarse_resolution (float): The resolution for coarse gridding (e.g. 3.0).
    
    Returns:
        fine_conc_map (np.ndarray): A 3D array (shape fine_grid_shape) containing the ion concentration (M)
                                    averaged over the trajectory.
    """
    # Determine the grid minimum from the fine grid points.
    grid_min = np.min(grid_points, axis=0)  # (x_min, y_min, z_min)

    # Calculate the coarse factor
    coarse_factor = int(coarse_resolution / resolution)
    
    # Compute coarse grid shape assuming fine grid shape is divisible by coarse_factor.
    nx_fine, ny_fine, nz_fine = fine_grid_shape
    nx_coarse = nx_fine // coarse_factor
    ny_coarse = ny_fine // coarse_factor
    nz_coarse = nz_fine // coarse_factor
    coarse_shape = (nx_coarse, ny_coarse, nz_coarse)
    
    # Initialize accumulators for coarse and fine counts.
    coarse_counts = np.zeros(coarse_shape, dtype=float)
    fine_counts = np.zeros(fine_grid_shape, dtype=float)
    frame_count = 0

    # Loop over frames in the trajectory.
    for ts in universe.trajectory:
        # Select ions using the provided selection string.
        sel = universe.select_atoms("name POT SOD CLA")
        if len(sel) == 0:
            continue
        ion_positions = sel.positions
        
        # Compute coarse grid cell indices for each ion.
        coarse_indices = np.floor((ion_positions - grid_min) / coarse_resolution).astype(int)
        
        # Check if ions are within coarse grid or just outside (within one fine cell)
        valid_coarse = ((coarse_indices[:, 0] >= 0) & (coarse_indices[:, 0] < nx_coarse) &
                       (coarse_indices[:, 1] >= 0) & (coarse_indices[:, 1] < ny_coarse) &
                       (coarse_indices[:, 2] >= 0) & (coarse_indices[:, 2] < nz_coarse))
        
        valid_fine = ((coarse_indices[:, 0] >= 0) & (coarse_indices[:, 0] <= nx_coarse) &
                      (coarse_indices[:, 1] >= 0) & (coarse_indices[:, 1] <= ny_coarse) &
                      (coarse_indices[:, 2] >= 0) & (coarse_indices[:, 2] <= nz_coarse))
        
        # Separate ions into those in coarse grid and those just outside
        ions_coarse = ion_positions[valid_coarse]
        ions_fine = ion_positions[valid_fine & ~valid_coarse]
        
        # Process ions in coarse grid
        coarse_indices_valid = coarse_indices[valid_coarse]
        for idx in coarse_indices_valid:
            coarse_counts[idx[0], idx[1], idx[2]] += 1
        
        # Process ions just outside coarse grid (within one fine cell)
        if len(ions_fine) > 0:
            fine_indices = np.floor((ions_fine - grid_min) / resolution).astype(int)
            # Ensure fine indices are within bounds
            valid_fine_idx = ((fine_indices[:, 0] >= 0) & (fine_indices[:, 0] < nx_fine) &
                              (fine_indices[:, 1] >= 0) & (fine_indices[:, 1] < ny_fine) &
                              (fine_indices[:, 2] >= 0) & (fine_indices[:, 2] < nz_fine))
            fine_indices = fine_indices[valid_fine_idx]
            for idx in fine_indices:
                fine_counts[idx[0], idx[1], idx[2]] += 1
        
        frame_count += 1

    if frame_count == 0:
        raise ValueError("No frames processed or no ions found with the selection: " + ion_selection)
    
    # Average the counts over frames.
    avg_coarse_counts = coarse_counts / frame_count
    avg_fine_counts = fine_counts / frame_count

    # Convert counts to concentration.
    # For coarse grid:
    cell_volume_L = (coarse_resolution ** 3) * 1e-27  # in liters
    NA = 6.022e23  # Avogadro's number
    coarse_conc_map = (avg_coarse_counts / NA) / cell_volume_L

    # For fine grid (edge cases):
    fine_cell_volume_L = (resolution ** 3) * 1e-27  # in liters
    fine_conc_map = (avg_fine_counts / NA) / fine_cell_volume_L

    # Create combined concentration map
    # First upsample the coarse map
    fine_conc_map[:int(coarse_factor*nx_coarse), :int(coarse_factor*ny_coarse), :int(coarse_factor*nz_coarse)] = np.repeat(np.repeat(np.repeat(coarse_conc_map, coarse_factor, axis=0),
                                      coarse_factor, axis=1),
                                      coarse_factor, axis=2)
    
    return fine_conc_map / 2.0



def plot_ion_conc_slice(ion_conc_map, nx, ny, nz, resolution=1.0):
    """
    Plot a mid-plane slice of the ion concentration map.

    Parameters:
        ion_conc_map (np.ndarray): A 3D array with shape (nx, ny, nz) of ion concentration (M).
        nx, ny, nz (int): Dimensions of the grid.
        resolution (float): Grid resolution in Å (default 1.0).
    """
    # Select a mid-plane slice along the x-axis.
    mid_index = nx // 2
    slice_data = ion_conc_map[mid_index, :, :]

    # Determine the physical extent of the y and z axes.
    # For a 1 Å grid, extent in y is from 0 to (ny-1)*resolution, similarly for z.
    extent = [0, (ny - 1) * resolution, 0, (nz - 1) * resolution]

    plt.figure(figsize=(8, 6))
    plt.imshow(slice_data, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    plt.title(f"Ion Concentration Map (Slice at x = {mid_index})")
    plt.xlabel("y (Å)")
    plt.ylabel("z (Å)")
    plt.colorbar(label="Ion Concentration (M)")
    plt.show()

