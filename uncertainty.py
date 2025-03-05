unique_parcels = np.unique(closest_parcellation)

for parcel in unique_parcels:
    
    parcel_mask = closest_parcellation == parcel
    parcel_distances = dist_both[parcel_mask, parcel-1]
    
    # Calculate absolute distances
    dist_array_abs[parcel_mask] = parcel_distances
    
    # Calculate relative distances (normalized within the parcel)
    dist_range = np.max(parcel_distances)
    dist_array_rel[parcel_mask] = parcel_distances / dist_range if dist_range > 0 else 0
