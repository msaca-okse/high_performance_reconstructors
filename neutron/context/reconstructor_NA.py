import sys
import os
import time
import numpy as np
os.chdir('/zhome/71/c/146676/main/')


plot = False
skip = 1
fig_path =  '/dtu-compute/msaca/sliceA_neutron_psi/output/tilt_cor_corrector_slices/A_rec_slice'
save_folder_fbp ='/dtu-compute/msaca/sliceA_neutron_psi/output/fbp_recon/'
save_folder_tv = '/dtu-compute/msaca/sliceA_neutron_psi/output/tv_recon/'
save_folder_dtv = '/dtu-compute/msaca/sliceA_neutron_psi/output/dtv_recon/'
path_to_add = '/zhome/71/c/146676/Desktop/msaca/main/'
sys.path.append(path_to_add)

import SimpleITK as sitk
from astropy.io import fits
from helpers import module_auxiliary as ma
import tifffile
from multiprocessing import Pool
import matplotlib.pyplot as plt
from helpers import image_utils as iu
from helpers import extended_data as ed
from registration import CTcorrector as ctc
#from cil.plugins.astra import FBP
from cil.recon import FBP
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry, ImageData, BlockDataContainer
from cil.plugins.ccpi_regularisation.functions import FGP_TV, FGP_dTV
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
from loaders import loader_XA_to_NA


path_cache_sino = '/dtu-compute/msaca/sliceA_neutron_psi/output/cache/sino_#####.tiff'
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

def find_split_arr(num_proc,subslices):
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
    return split_arr


def crop(image):
    return image[820:1180,:]

def recon_FBP_single(batch_id, num_proc = 4):
    split_arr = find_split_arr(num_proc,subslices)
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



def recon_FBP_multi(batch_id, num_proc = 4, output = False):
    print('Initiating batch FBP reconstruction from saved sinograms')
    split_arr = find_split_arr(num_proc,subslices)
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
    ig_batch_ = ag_batch.get_ImageGeometry()
    roi = {'horizontal_y':(820,1180,1), 'horizontal_x':(None,1600,1)}
    processor = Slicer(roi)
    processor.set_input(ig_batch_)
    ig_batch = processor.get_output()
    device = 'gpu'
    recon_slice_FBP = FBP(data_batch, image_geometry = ig_batch, filter='cosine', backend='astra').run().as_array().astype(np.float32)

    #recon_slice_FBP = fbp(data_batch).as_array().astype(np.float32)
    if output:
        return recon_slice_FBP, data_batch

    print(f"Reconstruction time: {(time.time()-start_time):.2f} seconds")
    start_time = time.time()

    for i in range(len(batch)):
        tifffile.imwrite(write_path_FBP[i], recon_slice_FBP[i])

    print(f"Writing time: {(time.time()-start_time):.2f} seconds")







def recon_TV_single(batch_id, num_proc = 4, N_iter = 50, alpha = 75.0):
    print('Initiating single slice TV reconstruction from saved sinograms')
    split_arr = find_split_arr(num_proc, subslices)
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



def recon_TV_multi(batch_id, num_proc = 4, N_iter = 50, alpha = 75.0):
    print('Initiating batch TV reconstruction from saved sinograms')
    split_arr = find_split_arr(num_proc, subslices)
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
    ig_batch_ = ag_batch.get_ImageGeometry()
    roi = {'horizontal_y':(820,1180,1), 'horizontal_x':(None,1600,1)}
    processor = Slicer(roi)
    processor.set_input(ig_batch_)
    ig_batch = processor.get_output()
    device = 'gpu'

    initial = ig_batch.allocate(0)
    A = ProjectionOperator(ig_batch,ag_batch,device)
    b = data_batch
    F = LeastSquares(A,b)
    G = alpha*FGP_TV(device='gpu',nonnegativity=False)
    reconstructor = FISTA(f=F, g=G, initial=initial)
    reconstructor.run(N_iter)
    recon_slice_TV = reconstructor.solution.copy().as_array().astype(np.float32)

    print(f"Reconstruction time: {(time.time()-start_time):.2f} seconds")
    start_time = time.time()

    for i in range(len(batch)):
        tifffile.imwrite(write_path_TV[i], recon_slice_TV[i])

    print(f"Writing time: {(time.time()-start_time):.2f} seconds")







def recon_dTV_multi(batch_id, num_proc = 4, xray_reference = 'tv', N_iter = 50, alpha = 75.0, eta = 0.01):
    ## 0. Load in neutron data for the batch
    subslices = range(0,1788)
    split_arr = find_split_arr(num_proc,subslices)
    batch = split_arr[batch_id-1]
    print('Batch id:'+ str(batch_id) + 'reconstructing slices:')
    print(batch)
    print('Loading data')
    read_path = ma.generate_paths(path_cache_sino, batch)
    write_path_dTV = ma.generate_paths(save_folder_dtv + 'slice_dtv_####.tiff',batch)
    data_batch = np.empty((len(batch), N_angles, N_pixels), dtype = np.float32)

    start_time = time.time()
    for i in range(len(batch)):
        data_batch[i] = tifffile.imread(read_path[i])
    N_batch_slices = np.shape(data_batch)[0]
    print(f"Neutron Data loading time: {(time.time()-start_time):.2f} seconds")
    start_time = time.time()

    ag_batch = AcquisitionGeometry.create_Parallel3D(detector_position=[0,N_pixels//2,0])\
                            .set_angles(angles)\
                            .set_panel((N_pixels,N_batch_slices), pixel_size=(1,1))\
                            .set_labels(labels=('vertical','angle','horizontal'))

    
    data_batch = AcquisitionData(data_batch, geometry=ag_batch)
    data_batch.reorder('astra')

    ag_batch.set_angles(ag_batch.angles, initial_angle=+10)
    ig_batch_ = ag_batch.get_ImageGeometry()
    roi = {'horizontal_y':(820,1180,1), 'horizontal_x':(None,1600,1)}
    processor = Slicer(roi)
    processor.set_input(ig_batch_)
    ig_batch = processor.get_output()
    device = 'gpu'
    xray_batch_start = int(max(2886-batch[-1]*2886/1788-300,0))
    xray_batch_end = int(min(2886 - batch[0]*2886/1788+300,2887))
    xray_batch = range(xray_batch_start,xray_batch_end)
    output_volume = [[batch[0],batch[-1]+1],[None], [None]]
    reg = loader_XA_to_NA.load_subset_of_registered_data(dataset_XA=xray_reference, 
                dataset_NA='tv' , h=1, xray_slices = xray_batch,
                neutron_slices = batch, output_volume = output_volume)
    print(f"Xray reference reconstruction loading time: {(time.time()-start_time):.2f} seconds")
    start_time = time.time()
    XA = sitk.GetArrayFromImage(reg.moving).astype(np.float32)
    XA_ = ImageData(XA, geometry=ig_batch)

    initial = ig_batch.allocate(0)
    A = ProjectionOperator(ig_batch,ag_batch,device)
    b = data_batch
    G = alpha * FGP_dTV(reference=XA_, eta=eta, device='gpu', nonnegativity=False)
    F = LeastSquares(A,b)
    reconstructor = FISTA(f=F, g=G, initial=initial)

    reconstructor.run(N_iter, verbose = 1)
    recon_slice_dTV = reconstructor.solution.copy().as_array().astype(np.float32)

    print(f"Reconstruction time: {(time.time()-start_time):.2f} seconds")
    start_time = time.time()

    for i in range(len(batch)):
        tifffile.imwrite(write_path_dTV[i], recon_slice_dTV[i])

    print(f"Writing time: {(time.time()-start_time):.2f} seconds")