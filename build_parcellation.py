"""
Plot EEG data to surface
Basic function that DOES NOT accurately reflect volume conduction effects, 
but can be used without imaging or freesurfer outputs

Create parcellation here - 61 electrodes in standard 2010 notation and map to fsaverage 5 space

March 2025, Clara Weber webercl@cbs.mpg.de
"""

# housekeeping
import numpy as np
import pandas as pd 
import mne
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

# get eeg data labels (here from LEMON dataset)
data = read_raw_eeglab('sub-032304/sub-032304_EC.set')
lemon_ch_names = data.ch_names

# get eeg electrode coordinates
montage = make_standard_montage('standard_1020')

# or choose from other montages 
#mne.channels.get_builtin_montages()

# Get the 3D positions of electrodes
pos_3d = np.array([montage.get_positions()['ch_pos'][ch] for ch in lemon_ch_names])*1000

# Fetch fsaverage surface using nilearn
fsaverage = fetch_surf_fsaverage()

# Load the surface files
surf_lh = mesh_io.read_surface(fsaverage['pial_left'])
surf_rh = mesh_io.read_surface(fsaverage['pial_right'])
vertices_lh = surf_lh.GetPoints()
vertices_rh = surf_rh.GetPoints()

# Find closest electrode for each vertex
distances_lh = cdist(vertices_lh, pos_3d)
closest_electrode_lh = np.argmin(distances_lh, axis = 1) + 1
distances_rh = cdist(vertices_rh, pos_3d)
closest_electrode_rh = np.argmin(distances_rh, axis = 1) + 1

parcellation = np.concatenate((closest_electrode_lh, closest_electrode_rh))
np.save('closest_parcellation_fsa5_62eeg2010.npy', parcellation)

# load surface and schaefer parcellation for midline mask
s200 = fetch_parcellation('fsaverage5', 'schaefer',200, join=True)
mask = s200 == 0

# also can load fsa5 surface from brainspace here because it looks smoother :)
#surf_lh, surf_rh = load_fsa5()
plot_parcels = np.asarray(parcellation, dtype ='f')
plot_parcels[mask] = np.nan
plot_hemispheres(surf_lh, surf_rh, plot_parcels, size=(1200, 400), nan_color=(0.7,0.7,0.7,1), zoom = 1.2, cmap='tab20', color_bar=True, embed_nb = True)

# create dataframe as dictionary between parcel # and EEG electrode
if len(lemon_ch_names)==61:
    lemon_ch_names.insert(0, 'midline mask')
df = pd.DataFrame({'electrode no': lemon_ch_names, 'parcels': range(0,62)})
df.to_csv('electrodename_parcel_dict.csv')
