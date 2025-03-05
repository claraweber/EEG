# housekeeping
import numpy as np
import pandas as pd 
import numpy as np
from nilearn import plotting
import scipy.io as sio
from brainspace.mesh import mesh_io
from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_fsa5
from brainstat.datasets import fetch_parcellation
import os
from scipy.spatial.distance import cdist
import nibabel as nib
from nilearn.datasets import fetch_surf_fsaverage
from mne.channels import make_standard_montage
from mne.io import read_raw_eeglab
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from cmcrameri import cm

data = read_raw_eeglab('sub-032304/sub-032304_EC.set')
lemon_ch_names = data.ch_names

ch_names = [name for name in montage.ch_names if name in lemon_ch_names]

montage = make_standard_montage('standard_1020')
pos_3d = np.array([montage.get_positions()['ch_pos'][ch] for ch in lemon_ch_names])*800

fsaverage = fetch_surf_fsaverage()
surf_lh = mesh_io.read_surface(fsaverage['pial_left'])
surf_rh = mesh_io.read_surface(fsaverage['pial_right'])
vertices_lh = surf_lh.GetPoints()
vertices_rh = surf_rh.GetPoints()

distances_lh = cdist(vertices_lh, pos_3d)
distances_rh = cdist(vertices_rh, pos_3d)
dist_both = np.concatenate((distances_lh, distances_rh))

s200 = fetch_parcellation('fsaverage5', 'schaefer',200, join=True)
mask = s200 ==0

def weighted_mapping(matrix, distances):
    """
    matrix is n_electrodes x n_electrodes matrix of eeg data, symmetric

    distances needs to have shape n_vertices x n_electrodes and contain distance values (cdist) 
    from each vertex to each electrode
    """
    max_dist = np.max(distances)
    rel_dist = distances / max_dist

    out_values = rel_dist * np.mean(matrix, axis = 0)

    out_values_norm = np.mean(out_values, axis = 1) / np.mean(distances, axis = 1)

    return out_values_norm

# load data 
mat = np.load('coh_connectivity_matrix.npy')
mat_sym = (mat + mat.T) - np.diag(np.diag(mat)) # make symmetric 

weighted_eeg = weighted_mapping(mat_sym, dist_both)
weighted_eeg[mask] = np.nan
