import sys
import os
import time
import numpy as np


alpha = float(os.getenv("ALPHA"))
num_proc = int(os.getenv("NUM_PROC"))
plot = False
skip = 1
fig_path =  '/dtu-compute/msaca/output/tilt_cor_corrector_slices/A_rec_slice'
save_folder_fbp ='/dtu-compute/msaca/output/fbp_recon/'
save_folder_tv = '/dtu-compute/msaca/output/tv_recon/'
path_to_add = '/zhome/71/c/146676/Desktop/msaca/main/'
sys.path.append(path_to_add)

import SimpleITK as sitk
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
from cil.processors import PaganinProcessor, Slicer


path_cache_sino = '/dtu-compute/msaca/output/cache/sino_#####.tiff'
N_slices = ma.find_largest_number(path_cache_sino)

subslices = np.arange(0,N_slices, skip)


read_path = ma.generate_paths(path_cache_sino, [0])
image = tifffile.imread(read_path[0])
N_pixels = np.shape(image)[1]
N_angles = 1126


angles = np.linspace(0, 360, N_angles, endpoint=True, dtype=np.float32)
ag = AcquisitionGeometry.create_Parallel3D(detector_position=[0,N_pixels//2,0])\
                            .set_angles(angles)\
                            .set_panel((N_pixels,N_slices), pixel_size=(1/N_pixels,1/N_pixels))\
                            .set_labels(labels=('vertical','angle','horizontal'))

base_size = len(subslices) // num_proc
remainder = len(subslices) % num_proc
# Split the array into num_proc parts
split_arr = []
start_idx = 0
for i in range(num_proc):
    # For the first N-1 parts, add an extra element if there's a remainder
    end_idx = start_idx + base_size + (1 if i < remainder else 0)
    split_arr.append(subslices[start_idx:end_idx])
    start_idx = end_idx


def crop(image):
    return image[820:1180,:]

def recon_FBP_single(batch_id):
    batch = split_arr[batch_id-1]
    read_path = ma.generate_paths(path_cache_sino, batch)
    write_path_FBP = ma.generate_paths(save_folder_fbp + 'slice_fbp_####.tiff',batch)

    for i in range(len(batch)):
        data2D = tifffile.imread(read_path[i])
        ag2D = AcquisitionGeometry.create_Parallel2D(detector_position=[0,N_pixels//2])\
                            .set_angles(angles)\
                            .set_panel((N_pixels), pixel_size=(1))\
                            .set_labels(labels=('angle','horizontal'))
        data2D = AcquisitionData(data2D, geometry=ag2D)
        data2D.reorder('astra')
        ag2D.set_angles(ag2D.angles, initial_angle=+10)
        ig2D = ag2D.get_ImageGeometry()
        device = 'gpu'
        fbp = FBP(ig2D,ag2D,device)
        recon_slice_FBP = fbp(data2D).as_array().astype(np.float32)
        tifffile.imwrite(write_path_FBP[i], crop(recon_slice_FBP))



def recon_FBP_multi(batch_id):
    print('Initiating batch FBP reconstruction from saved sinograms')
    batch = split_arr[batch_id-1]
    read_path = ma.generate_paths(path_cache_sino, batch)
    write_path_FBP = ma.generate_paths(save_folder_fbp + 'slice_fbp_####.tiff',batch)
    data_batch = np.empty((len(batch), N_angles, N_pixels), dtype = np.float32)

    start_time = time.time()
    for i in range(len(batch)):
        data_batch[i] = tifffile.imread(read_path[i])
    N_batch_slices = np.shape(data_batch)[0]


    print(f"Data loading time: {(time.time()-start_time):.2f} seconds")
    start_time = time.time()

    ag_batch = AcquisitionGeometry.create_Parallel3D(detector_position=[0,N_pixels//2,0])\
                            .set_angles(angles)\
                            .set_panel((N_pixels,N_batch_slices), pixel_size=(1,1))\
                            .set_labels(labels=('vertical','angle','horizontal'))
    data_batch = AcquisitionData(data_batch, geometry=ag_batch)
    data_batch.reorder('astra')

    ag_batch.set_angles(ag_batch.angles, initial_angle=+10)
    ig_batch = ag_batch.get_ImageGeometry()
    device = 'gpu'
    fbp = FBP(ig_batch,ag_batch,device)

    recon_slice_FBP = fbp(data_batch).as_array().astype(np.float32)

    print(f"Reconstruction time: {(time.time()-start_time):.2f} seconds")
    start_time = time.time()

    for i in range(len(batch)):
        tifffile.imwrite(write_path_FBP[i], crop(recon_slice_FBP[i]))

    print(f"Writing time: {(time.time()-start_time):.2f} seconds")







def recon_TV_single(batch_id):
    print('Initiating single slice TV reconstruction from saved sinograms')
    batch = split_arr[batch_id-1]
    read_path = ma.generate_paths(path_cache_sino, batch)
    write_path_TV = ma.generate_paths(save_folder_tv + 'slice_tv_####.tiff',batch)
    start_time_total = time.time()

    for i in range(len(batch)):
        start_time = time.time()
        data2D = tifffile.imread(read_path[i])
        ag2D = AcquisitionGeometry.create_Parallel2D(detector_position=[0,N_pixels//2])\
                            .set_angles(angles)\
                            .set_panel((N_pixels), pixel_size=(1))\
                            .set_labels(labels=('angle','horizontal'))
        data2D = AcquisitionData(data2D, geometry=ag2D)
        ag2D.set_angles(ag2D.angles, initial_angle=+10)
        ig2D = ag2D.get_ImageGeometry()
        device = 'gpu'

        N_iter = 200
        initial = ig2D.allocate(0)
        A = ProjectionOperator(ig2D,ag2D,device)
        b = data2D
        F = LeastSquares(A,b)
        G = alpha*FGP_TV(device='gpu', nonnegativity=True)
        reconstructor = FISTA(f=F, g=G, initial=initial)
        reconstructor.run(N_iter)
        recon_slice_TV = reconstructor.solution.copy().as_array().astype(np.float32)
        tifffile.imwrite(write_path_TV[i], crop(recon_slice_TV))
        print(f"Iteration time: {(time.time()-start_time):.2f} seconds")

    print(f"Total time: {(time.time()-start_time_total):.2f} seconds")



def recon_TV_multi(batch_id):
    print('Initiating batch TV reconstruction from saved sinograms')
    batch = split_arr[batch_id-1]
    read_path = ma.generate_paths(path_cache_sino, batch)
    write_path_TV = ma.generate_paths(save_folder_tv + 'slice_tv_####.tiff',batch)
    data_batch = np.empty((len(batch), N_angles, N_pixels), dtype = np.float32)

    start_time = time.time()
    for i in range(len(batch)):
        data_batch[i] = tifffile.imread(read_path[i])
    N_batch_slices = np.shape(data_batch)[0]

    print(f"Data loading time: {(time.time()-start_time):.2f} seconds")
    start_time = time.time()

    ag_batch = AcquisitionGeometry.create_Parallel3D(detector_position=[0,N_pixels//2,0])\
                            .set_angles(angles)\
                            .set_panel((N_pixels,N_batch_slices), pixel_size=(1,1))\
                            .set_labels(labels=('vertical','angle','horizontal'))

    
    data_batch = AcquisitionData(data_batch, geometry=ag_batch)
    data_batch.reorder('astra')

    ag_batch.set_angles(ag_batch.angles, initial_angle=+10)
    ig_batch = ag_batch.get_ImageGeometry()
    device = 'gpu'

    N_iter = 120
    initial = ig_batch.allocate(0)
    A = ProjectionOperator(ig_batch,ag_batch,device)
    b = data_batch
    F = LeastSquares(A,b)
    G = alpha*FGP_TV(device='gpu',nonnegativity=True)
    reconstructor = FISTA(f=F, g=G, initial=initial)
    reconstructor.run(N_iter)
    recon_slice_TV = reconstructor.solution.copy().as_array().astype(np.float32)

    print(f"Reconstruction time: {(time.time()-start_time):.2f} seconds")
    start_time = time.time()

    for i in range(len(batch)):
        tifffile.imwrite(write_path_TV[i], crop(recon_slice_TV[i]))

    print(f"Writing time: {(time.time()-start_time):.2f} seconds")