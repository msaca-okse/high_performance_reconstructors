# Goal ofthis scriptis to handle the 3D volume as individual 2D slices. The projections are normalized (flat + dose corrected), then loaded in smaller batches to save memory and then saved as sinograms.
# The projections are NOT dark current corrected.

## Select packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from cil.processors import Normaliser
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.utilities.display import show2D, show_geometry
from cil.io import TIFFWriter
import imageutils as iu

# Specify settings
upper_path = '/work3/msaca/Mars_data_1/02_rawdata/'
subfolder_projections = 'ct_3x1126_60s/'
subfolder_flats = 'OB2_DC2/'
folder_normalizations = '/work3/msaca/Mars_data_1/02_filtereddata/normalizations/'
folder_filter = '/work3/msaca/Mars_data_1/02_filtereddata/filtered_projections/'
filename_prefix_projections = 'ct_3x1126_60s_'
filename_prefix_flats = 'OB2_DC2_'
filename_suffix = '.fits'
N_pixels = 2048
N_slices = 64
slice_offset = 1000
OB_reg_TL = (1061, 1570)
OB_reg_BR = (1274 , 1714)
N_projections = 1125 # Number of projections
batch_size = 32

N_OB = 240 # Number of flats
file_count = 2 # flat index offset
OB = np.zeros((N_pixels, N_pixels))
for i_OB in range(N_OB):
    file_count = file_count + 1
    path_load_OB = upper_path + subfolder_flats + filename_prefix_flats + f"{file_count:05}" + '.fits'
    OB = OB + fits.open(path_load_OB)[0].data.astype(float)/N_OB # Load data
    if not np.mod(file_count,50):
        string_progress = '. Loaded file ' + str(file_count) + ' out of ' + str(N_OB)
        print(string_progress)
print('Done filtering OB-files')

D0 = np.median(OB[OB_reg_TL[0]:OB_reg_BR[0], OB_reg_TL[1]:OB_reg_BR[1]])
OB_mean = OB

# Settings for the projection file loader
N_batches = int(np.ceil(N_slices/batch_size))

D = np.zeros((N_projections))
OB_mean = np.clip(OB_mean,1,1000000) # Make sure there are no zero or negative values.
for i_batch in range(N_batches):
    if i_batch == N_batches - 1:
        updated_batch_size = N_slices - (N_batches - 1)*batch_size
    else:
        updated_batch_size = batch_size
    # Set the geometry. Data is saved as 2D slices...
    ag = AcquisitionGeometry.create_Parallel3D()  \
    .set_panel(num_pixels=[N_pixels,updated_batch_size])        \
    .set_angles(angles=np.linspace(0,360,N_projections,endpoint=False))
    ag.set_labels(['angle','horizontal','vertical'])
    i_from = i_batch*batch_size + slice_offset
    i_to = i_batch*batch_size+updated_batch_size + slice_offset

    OB_batch = OB_mean[:,i_from:i_to] # Select the OB means for the current batch

    A = np.zeros((N_projections, N_pixels,updated_batch_size)) # Allocate sinograms
    file_count = 0
    for i_proj in range(N_projections):
        file_count = file_count + 1
        path_1 = upper_path + subfolder_projections + filename_prefix_projections + f"{(3*file_count-2):05}" + filename_suffix
        path_2 = upper_path + subfolder_projections + filename_prefix_projections + f"{(3*file_count-1):05}" + filename_suffix
        path_3 = upper_path + subfolder_projections + filename_prefix_projections + f"{3*file_count:05}" + filename_suffix

        if not np.mod(file_count,25):
            string_progress = 'Batch ' + str(i_batch+1) + ' out of ' + str(N_batches) + '. Loaded file ' + str(file_count) + ' out of ' + str(N_projections)
            print(string_progress)

        projection1 = fits.open(path_1)[0].data.astype(float)
        projection2 = fits.open(path_2)[0].data.astype(float)
        projection3 = fits.open(path_3)[0].data.astype(float)
        projection_mean = (projection1 + projection2 + projection3)/3
        D[i_proj] = np.median(projection_mean[OB_reg_TL[0]:OB_reg_BR[0], OB_reg_TL[1]:OB_reg_BR[1]])
        dose_filename = folder_filter + 'dose.npy'
        np.save(dose_filename, np.append(D,D0))
        filter_filename = folder_filter + 'filter_proj' + f"{file_count:05}" + '.npy'


        projection_mean_batch = projection_mean[:,i_from:i_to]

        #Case 1
        #projection_lognormalized = -(np.log(projection_mean_batch) - np.log(OB_batch) + np.log(D0) - np.log(D[i_proj]))
        #filtered_projection_lognormalized = iu.spotclean(projection_lognormalized, size=10)
        
        #Case 2
        projection_lognormalized = -(np.log(projection_mean) - np.log(OB_mean) + np.log(D0) - np.log(D[i_proj]))

        filtered_projection_lognormalized = iu.spotclean(projection_lognormalized, size=10)
        #np.save(filter_filename, filtered_projection_lognormalized)
        filtered_projection_lognormalized = filtered_projection_lognormalized[:,i_from:i_to]


        # Save the array
        A[i_proj] = filtered_projection_lognormalized

    A_slices = AcquisitionData(A, geometry=ag, deep_copy=False)
    A_slices.reorder(('vertical', 'angle','horizontal'))
    A_write = TIFFWriter(data=A_slices, file_name = folder_normalizations + 'sinogram_normalized', counter_offset=i_batch*batch_size+1)
    A_write.write()