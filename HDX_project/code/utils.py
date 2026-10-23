from scipy.spatial.distance import pdist
import mdtraj as md
import numpy as np
import os

def find_xtc_filenames( path_to_dir, suffix=".xtc" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def find_npy_filenames( path_to_dir, suffix=".npy" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

# functions to calculate statistics
def RMSE(actual, predicted):
    
    # Convert inputs to numpy arrays 
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate the squared differences
    squared_diff = (actual - predicted) ** 2
    
    # Compute the mean of squared differences
    mean_squared_diff = np.mean(squared_diff)
    
    # Take the square root of the mean squared difference
    return np.sqrt(mean_squared_diff)

# function to filter out frames that have overlapping atoms (SASA calculation will fail otherwise)
def filter_frames(pdb_file, xtc_file, threshold=0.001):
    print(f"Loading trajectory: {xtc_file}")
    traj = md.load(xtc_file, top=pdb_file)
    print(f"Trajectory loaded with {traj.n_frames} frames and {traj.n_atoms} atoms.")

    ok_frames = []
    min_dists = []

    for i, xyz in enumerate(traj.xyz):
        # Compute all pairwise distances (upper triangle only)
        min_dist = pdist(xyz).min()
        min_dists.append(min_dist)
        if min_dist > threshold:
            ok_frames.append(i)
        else:
            print(f"Frame {i}: Minimum distance = {min_dist:.6f} nm (overlap detected)")

    print("\nSummary:")
    print(f"  Threshold: {threshold} Å")
    print(f"  Frames checked: {traj.n_frames}")
    print(f"  Frames OK: {len(ok_frames)}")
    print(f"  Frames failing: {traj.n_frames - len(ok_frames)}")

    print("\nReturning frames passing (min_dist > threshold):")

    return np.array(ok_frames).astype(int)

def find_elbow(x, y):
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Sort x (and y) if needed, but keep track of original indices
    if not np.all(np.diff(x) >= 0):
        idx_sort = np.argsort(x)
        x_sorted = x[idx_sort]
        y_sorted = y[idx_sort]
    else:
        idx_sort = np.arange(len(x))
        x_sorted, y_sorted = x, y

    # Line between first and last point
    p1 = np.array([x_sorted[0], y_sorted[0]])
    p2 = np.array([x_sorted[-1], y_sorted[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    # Distances from each point to the line
    vec_from_p1 = np.vstack([x_sorted - p1[0], y_sorted - p1[1]]).T
    proj_len = np.dot(vec_from_p1, line_vec_norm)
    proj_point = np.outer(proj_len, line_vec_norm) + p1
    dist = np.linalg.norm(vec_from_p1 - (proj_point - p1), axis=1)

    # Elbow = point with maximum distance
    idx_elbow_sorted = np.argmax(dist)
    idx_elbow_original = idx_sort[idx_elbow_sorted]

    return idx_elbow_original, x[idx_elbow_original], y[idx_elbow_original]