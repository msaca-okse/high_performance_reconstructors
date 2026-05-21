# This script loads in a reconstruction, does preprocessing (morph_spot_clean + rotation)
# Then a reconstruction is made using fbp. Both preprocessing and reconstruction is done
# in parallel, however the preprocessing must be finished once reconstruction can start

import sys
import os
import time
import numpy as np
from cil.processors import Slicer


try:
    th = float(os.getenv("THRESHOLD"))
except:
    print('Using default threshold!')
    th = 0.5

try:
    size = float(os.getenv("SIZE"))
except:
    print('Using default size!')
    size = 7

try:
    decNum = float(os.getenv("DECNUM"))
except:
    print('Using default decnum!')
    decNum = 5

try:
    wname = float(os.getenv("WNAME"))
except:
    print('Using default wname!')
    wname = 10

try:
    sigma = float(os.getenv("SIGMA"))
except:
    print('Using default sigma!')
    sigma = 0.3

plot = False
skip = 100
fig_path =  '/dtu-compute/msaca/output/tilt_cor_corrector_slices/A_rec_slice'
save_folder_fbp ='/dtu-compute/msaca/output/fbp_recon/'
save_folder_tv = '/dtu-compute/msaca/output/tv_recon/'


# Add the desired directory to the sys.path
path_to_add = '/dtu-compute/msaca/muhrec_folder2/build-imagingsuite/Release/lib/'
sys.path.append(path_to_add)
path_to_add = '/dtu-compute/msaca/muhrec_folder2/amglib/CBCTCalibration/'
sys.path.append(path_to_add)
path_to_add = '/dtu-compute/msaca/muhrec_folder2/imagingsuite/package/'
sys.path.append(path_to_add)
path_to_add = '/zhome/71/c/146676/Desktop/msaca/main/'
sys.path.append(path_to_add)

import SimpleITK as sitk
import imgalg
from astropy.io import fits
import module_auxiliary as ma
import tifffile
from multiprocessing import Pool
import matplotlib.pyplot as plt
import image_utils as iu
import extended_data as ed
import CTcorrector as ctc
from cil.plugins.astra import FBP
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry, ImageData, BlockDataContainer
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.functions import L2NormSquared, L1Norm, BlockFunction, MixedL21Norm, IndicatorBox, TotalVariation, LeastSquares
from cil.optimisation.operators import BlockOperator, GradientOperator, IdentityOperator, FiniteDifferenceOperator
from cil.optimisation.algorithms import CGLS, SIRT, GD, FISTA, ISTA, PDHG, SPDHG
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, TotalVariation, \
                                       ZeroFunction
from cil.optimisation.operators import BlockOperator, GradientOperator,\
                                       GradientOperator
from cil.processors import PaganinProcessor

start_time = time.time()
##########################################################
# Part 1: Preprocessing


path = '/dtu-compute/msaca/sliceA_neutron_psi/OB/OB_start_#####.fits'
A = range(1,31)
ob_paths = ma.generate_paths(path, A)


def load_file(path):
    with fits.open(path) as hdul:
        # Assume the data is in the primary HDU
        return hdul[0].data
with Pool() as pool:
    ob_data = pool.map(load_file, ob_paths)

# Now compute the mean across all loaded files
# Convert list of arrays to a single array, assuming the arrays are of the same shape
ob = np.stack(ob_data)
ob = np.mean(ob, axis=0)
#ob = ob[50:1910,:2060]
ob = ob[50:1910,90:1970]

path = '/dtu-compute/msaca/sliceA_neutron_psi/DC/DC_#####.fits'
A = range(1,31)
dc_paths = ma.generate_paths(path, A)
with Pool() as pool:
    dc_data = pool.map(load_file, dc_paths)


# Now compute the mean across all loaded files
# Convert list of arrays to a single array, assuming the arrays are of the same shape
dc = np.stack(dc_data)
dc = np.mean(dc, axis=0)
#dc = dc[50:1910,:2060]
dc = dc[50:1910,90:1970]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Loading time: {elapsed_time:.2f} seconds")

path = '/dtu-compute/msaca/sliceA_neutron_psi/ct_3x1126_60s/ct_3x1126_60s_#####.fits'
path_cache = '/dtu-compute/msaca/output/cache/spot_cleaned_#####.tiff'

batch = np.linspace(1,1127,200).astype(np.uint16)

def preprocess_projection(batch_idx):
    mode = 'load'

    a = range(batch[batch_idx],batch[batch_idx+1])

    if mode == 'load':
        count = 0
        Data = []
        for i in a:
            A = [i]
            temp_path = ma.generate_paths(path_cache,A)[0]
            Data.append(tifffile.imread(temp_path))
        Data = np.stack(Data)
        return Data.astype(np.float32)

    Data = []
    for i in a:
        A = [3*i-2, 3*i-1, 3*i]
        data_paths = ma.generate_paths(path, A)
        data = ma.fits_loader(data_paths)
        #data = data[:,50:1910,:2060]
        data = data[:,50:1910,90:1970]
        data = np.median(data, axis=0)
        Data.append(data)

    Data = np.stack(Data)    
    Data = (-np.log((np.abs(Data-dc[np.newaxis])+0.0001)/(0.0001+np.abs(ob[np.newaxis]-dc[np.newaxis])))).astype(np.float32)
    roi = [1600,1700,1000,1200] # [y_start, y_end, x_start, x_end]
    Means = np.mean(Data[:,roi[0]:roi[1], roi[2]:roi[3]], axis=(1,2))
    Data = Data-Means[:,np.newaxis, np.newaxis]
    Data = iu.morph_spot_clean(Data,th_peaks=th,th_holes=th,method=0,size = size)

    if mode == 'save':
        count = 0
        for i in a:
            A = [i]
            temp_path = ma.generate_paths(path_cache, A)[0]
            tifffile.imwrite(temp_path, Data[count].astype(np.float32))
            count = count+1
        return Data.astype(np.float32)
    if mode is None:
        return Data.astype(np.float32)


with Pool() as pool:
   Data_ =  pool.map(preprocess_projection, range(len(batch)-1))

Data = np.vstack(Data_)
Data = np.transpose(Data, [1,0,2])
N_slices, N_angles, N_pixels = np.shape(Data)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Preprocess time: {elapsed_time:.2f} seconds")
################################################################################
#
# Part II, Beam padding,  Cor estimation, tilt-correction and sinogram based preprocessing
#
#################################################################################
# Add beam padding with gaussian blur
#subslices = np.arange(100, N_slices-100, skip)


angles = np.linspace(0, 360, N_angles, endpoint=True, dtype=np.float32)
slices = np.arange(0, N_slices)
ag = AcquisitionGeometry.create_Parallel3D(detector_position=[0,N_pixels//2,0])\
                            .set_angles(angles)\
                            .set_panel((N_pixels,N_slices), pixel_size=(1,1))\
                            .set_labels(labels=('vertical','angle','horizontal'))

sinograms = ed.ExtendedData()
sinograms.set_data(data = Data)
sinograms.set_slices(slices)
sinograms.pad_edges()
sinograms.acquisition_data(geometry=ag)
# Paganin filter
#print(sinograms.data.as_array()[200:210,200:210,200:210])
#sinograms.data.reorder('cil')
#sinograms.data.geometry.config.units = 'um'
#processor = PaganinProcessor(delta=delta,beta=beta, full_retrieval=False, energy=14, energy_units='keV')
#processor.set_input(sinograms.data)
#sinograms.data = processor.get_output()
#print(sinograms.data.as_array()[200:210,200:210,200:210])
input_data = sinograms.data*100
sinograms.data.reorder('astra')
end_time = time.time()
elapsed_time = end_time - start_time


############## Now do the cor and tilt correction printing the result

corrector = ctc.CTcorrector()
angles = np.linspace(0,360,num=N_angles)
corrector.set_angles(angles = angles)
corrector.load_data(input_data)
corrector.set_labels()

angle = 0.325 # 0.325 gives the right rotation
translation = -26 # -26 gives the right translation
angle_radians = np.deg2rad(angle)
rotation_matrix = [
                [np.cos(angle_radians),0,  -np.sin(angle_radians)],
                [0,1,0],
                [np.sin(angle_radians), 0,  np.cos(angle_radians)]
            ]
matrix = [elem for row in rotation_matrix for elem in row]
translation = [-translation, 0, 0]
transform = corrector.transformation(matrix = matrix, translation = translation)
data2 = corrector.resample(data=corrector.data, transform=transform)

sinograms.set_data(data2)
sinograms.acquisition_data(geometry=ag)
#sinograms.set_subdata(subslices=subslices)
sinograms.remove_ring(subdata = False, decNum = decNum, wname  = wname, sigma = sigma)

data = sinograms.data.as_array()
N_slices = np.shape(data)[0]
A = np.arange(N_slices)
path_cache_sino = '/dtu-compute/msaca/output/cache/sino_#####.tiff'
temp_path = ma.generate_paths(path_cache_sino, A)
def writer(i):
    tifffile.imwrite(temp_path[i], data[i].astype(np.float32))

with Pool() as pool:
   pool.map(writer, A)


#reconstruction1 = np.empty((len(sinograms.subslices), N_pixels,N_pixels))
#reconstruction2 = np.empty((len(sinograms.subslices), N_pixels,N_pixels))

#A = np.arange(len(sinograms.subslices))
#fig_paths = ma.generate_paths(fig_path + '_##.png',A)
