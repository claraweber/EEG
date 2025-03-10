# EEG to surface mapping

This script builds a parcellation that allows to map EEG data to surface. ! Spatial resolution is highly limited ! This is only an approximation that allows for easy visualization and does not require MRI.
Uses a closest electrode approach to create parcels. Script originally written for LEMON data - 61 electrodes in 2010 space. Maps to fsaverage5.

Parcellation:
![image](https://github.com/user-attachments/assets/2ce6c489-4553-4f1e-8deb-64732ac77045)

EEG mapped to labels:
![Image](https://github.com/user-attachments/assets/56a2c71c-fd38-4bf1-b818-1dc91eb84cb0)

Below are two uncertainty maps (i.e., distance to assigned electrode within each parcel). Upper panel shows relative values normalized to maximum distance to assigned electrode, lower panel shows absolute values. 
![image](https://github.com/user-attachments/assets/7030f52e-f7d0-4725-aff5-b813e9d8dc52)

When correcting for uncertainty (map * (1-relative distance)), results follow a slightly different but plausible pattern:
![image](https://github.com/user-attachments/assets/46fb6681-eb4d-4616-bfe9-d4cecb5b09ea)

