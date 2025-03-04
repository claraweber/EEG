"""
Plot EEG data to surface
Basic function that DOES NOT accurately reflect volume conduction effects, 
but can be used without imaging or freesurfer outputs

This function uses parcellation from build_parcellation script
then takes data and maps it onto the surface for plotting

March 2025, Clara Weber webercl@cbs.mpg.de
"""

# housekeeping
import numpy as np
import pandas as pd 
import numpy as np
from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_fsa5
from brainstat.datasets import fetch_parcellation
import os
import nibabel as nib

def map_to_labels(in_eeg, labels):
    """
    this function takes a vector of EEG data and maps it to the cortical surface
    here 59 channel data is considered
    in_eeg = one dimensional (shape 59,) numpy array of eeg data
    out_data = mapped to 20484 vertices of fsaverage5 surface
    """
    out_data = np.empty((20484,))

    for i in range(len(np.unique(labels))):
        out_data[labels == i+1] = in_eeg[i]

    return out_data

# load data
mat = np.load('sub-032304_EC_coh_matrix.npy')
mat_sym = (mat + mat.T) - np.diag(np.diag(mat))

# load surface and schaefer parcellation for midline mask
s200 = fetch_parcellation('fsaverage5', 'schaefer',200, join=True)
mask = s200 == 0

eeg_parcellation = np.load('closest_parcellation_fsa5_62eeg2010.npy')

eeg_mapped = map_to_labels(np.mean(mat_sym, axis = 0), eeg_parcellation)
eeg_mapped[mask] = np.nan # remove midline zeros

surf_lh, surf_rh = load_fsa5()

plot_hemispheres(surf_lh, surf_rh, 
    eeg_mapped, 
    size = (1200, 400), 
    nan_color = (0.7, 0.7, 0.7, 1),  
    zoom = 1.2, 
    cmap='viridis', 
    color_bar = True, 
    embed_nb = True, 
    color_range=(0.2, 0.7) # adjust this to your data!
    )


# handle missing electrodes
# print missing electrodes
for i in range(len(lemon_ch_names)):
    if lemon_ch_names[i] in lemon_ch_names_59:
        #print(i, lemon_ch_names[i])
        pass
    else:
        print(i, lemon_ch_names[i], 'missing')

# compare channel names, create a key to map 59 channel labels to 61 space
key_59 = np.empty((60,))
for i in range(len(np.unique(lemon_ch_names_59))):
    p = df.loc[df['electrode no'] == lemon_ch_names_59[i], 'parcels'].values
    key_59[i] = p

key_59 = key_59[1:]

def resize_matrix(in_matrix, key):
    """
    Resizes martix with <61 entries to that size (to fit parcellation and allow for plotting)
    in_matrix = smaller data
    key = 61, shaped array that replaces integers at missing position with zeros (see above)
    """
    out_matrix = np.zeros((61, 61))

    for i in range(61):
        for j in range(61):
            try:
                out_matrix[i, j] = in_matrix[key==i, key==j]
            except:
                pass
    
    return out_matrix

mat_sym_61 = resize_matrix(mat_sym, key_59)
eeg_mapped_59 = map_to_labels(np.mean(mat_sym_61, axis =0), eeg_parcellation)
eeg_mapped_59[mask] = np.nan

plot_hemispheres(surf_lh, surf_rh, 
    eeg_mapped_59, 
    size = (1200, 400), 
    nan_color = (0.7, 0.7, 0.7, 1),  
    zoom = 1.2, 
    cmap='viridis', 
    color_bar = True, 
    embed_nb = True, 
    color_range=(0, 0.5) # adjust this to your data!
    )