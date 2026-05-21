import numpy as np
from cil.io import TIFFStackReader
import module_auxiliary as ma
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry, ImageData
from cil.processors import Slicer, RingRemover, CentreOfRotationCorrector
from numba import cuda
from cil.plugins.astra import FBP
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


class ExtendedData:

    
    def __init__(self, data = None, slices = None, subslices=None):
        self.data = data
        self.slices = slices
        self.subslices = subslices
        self.subdata = None
    
    def set_data(self,data):
        self.data = data
        if self.slices is None:
            self.slices = np.arange(np.shape(data)[0])

    def set_slices(self, slices):
        self.slices = slices

    def set_subslices(self,subslices):
        self.subslices = subslices
        self.subslices_indices = np.array([np.abs(self.slices - sub).argmin() for sub in self.subslices])
        

    def load_data(self,path, slices):
        paths = ma.generate_paths(path, slices)
        reader = TIFFStackReader(paths)
        self.data = reader.read()
        self.slices = slices

    def scale_data(self,scale,offset):
        self.data = self.data*scale + offset

    def acquisition_data(self, data=None, geometry=None):
        if (data is None) and (self.data is None):
            raise ValueError("Load data using .add_data(data) method, or by supplying it here")
        if (data is not None) and (self.data is not None):
            raise ValueError("Data has already been supplied. Overwrite it")
        if geometry is None:
            raise ValueError("Please supply Acquisition geometry")
        
        if (data is not None) and (self.data is None):
            self.data = AcquisitionData(data, geometry=geometry)
            if self.subslices is not None:
                self.set_subslices(subslices = self.slices)
                

        if (data is None) and (self.data is not None):
            self.data = AcquisitionData(self.data, geometry=geometry)



    def pad_edges(self):
        if self.slices is None:
            raise ValueError("Supply slices using add_slices")
        if isinstance(self.data, np.ndarray):
            mask, indices, data = ma.determine_edge2(self.data,bias = 20,slices=self.slices)
            self.data = ma.gaussian_padding(self.data,indices, sigma = 100, cutoff=10, pad_mean_window_size = 50)

        # Does not work ATM
        #if type(self.data) is AcquisitionData:
        #    data = self.data.as_array()
        #    mask, indices, data = ma.determine_edge2(data,bias = 15,slices=self.slices)
        #    data = ma.gaussian_padding(data,indices, sigma = 30, cutoff=4, pad_mean_window_size = 50)
        #    self.data.fill(data)

    def set_rotation_axis(self, position = [0,0,0], rotation = [0,0,1]):
        self.data.geometry.config.system.rotation_axis.direction = rotation
        self.data.geometry.config.system.rotation_axis.position = position

    def set_subdata(self,subslices = None):
        self.set_subslices(subslices=subslices)
        skip = self.subslices_indices[1] - self.subslices_indices[0]

        roi = {'vertical':(self.subslices_indices[0], None,skip)}
        processor = Slicer(roi)
        processor.set_input(self.data)
        self.subdata = processor.get_output()
        self.subdata.geometry.config.system.detector.position[2] = 0

    def remove_ring(self,subdata = False, decNum = 4, wname = 5,sigma = 0.1):
        wname = 'db'+str(wname)
        if not subdata:
            ringrmv = RingRemover(decNum=decNum, wname=wname, sigma=sigma, info=True)
            ringrmv.set_input(self.data)
            self.data = ringrmv.get_output()
        else:
            if self.subdata is None:
                raise ValueError('Specify subdata using .set_subdata()')
            ringrmv = RingRemover(decNum=decNum, wname=wname, sigma=sigma, info=True)
            ringrmv.set_input(self.subdata)
            self.subdata = ringrmv.get_output()

    def fbp(self, subdata = False):

        if not cuda.is_available():
            raise ValueError("GPU is not available.")
        
        if subdata:
            #print(np.shape(self.subdata.as_array()))
            #ig = self.subdata.geometry.get_ImageGeometry(resolution=1)
            #print(ig)
            #self.subdata.reorder('astra')
            #device = 'gpu'
            #fbp = FBP(ig,self.subdata.geometry,device)
            #reconstruction = fbp(self.subdata)
            reconstruction = np.empty((len(self.subslices), self.subdata.shape[2],self.subdata.shape[2]))
            for i in range(len(self.subslices)):
                data2D = self.subdata.get_slice(vertical=i)
                data2D.reorder('astra')
                ag2D = data2D.geometry
                ag2D.set_angles(ag2D.angles, initial_angle=0.0)
                ig2D = ag2D.get_ImageGeometry()
                device = 'gpu'
                fbp = FBP(ig2D,ag2D,device)
                reconstruction[i] = fbp(data2D)

        else:
            ig = self.data.geometry.get_ImageGeometry(resolution=1)
            self.data.reorder('astra')
            device = 'gpu'
            fbp = FBP(ig,self.data.geometry,device)
            reconstruction = fbp(self.data)

        return reconstruction


    def xcor_offset(self,slice_index, ang_tol = 0.2,n_projs = 10):
        projection_indices = np.arange(0,360, step=360//n_projs).astype(int)
        offsets = np.empty(len(projection_indices))
        for projection_index, iter in zip(projection_indices,range(len(projection_indices))):
            processor = CentreOfRotationCorrector.xcorrelation(slice_index,projection_index=int(projection_index),ang_tol=ang_tol)
            processor.set_input(self.subdata)
            temp = processor.get_output()
            offsets[iter] = temp.geometry.config.system.rotation_axis.position[0]
        return np.mean(offsets)

    def calculate_cor_axis(self,n_projs = 10):
        offsets = np.empty(len(self.subslices_indices))
        for iter in  range(len(self.subslices_indices)):
            offsets[iter] = self.xcor_offset(slice_index = iter, ang_tol = 0.2,n_projs = n_projs)
        
        return offsets

    def correct_axis(self, n_projs = 10):
        offsets = self.calculate_cor_axis(n_projs = n_projs)
        fit = np.polyfit(self.subslices, offsets,deg=1)
        return [fit[0], 0.0, 1], [fit[1], 0, 0.0]




    def tv(self, subdata = False):
        N_iter = 50
        if not cuda.is_available():
            raise ValueError("GPU is not available.")
        
        if subdata:
            #print(np.shape(self.subdata.as_array()))
            #ig = self.subdata.geometry.get_ImageGeometry(resolution=1)
            #print(ig)
            #self.subdata.reorder('astra')
            #device = 'gpu'
            #fbp = FBP(ig,self.subdata.geometry,device)
            #reconstruction = fbp(self.subdata)
            reconstruction = np.empty((len(self.subslices), self.subdata.shape[2],self.subdata.shape[2]))
            for i in range(len(self.subslices)):
                data2D = self.subdata.get_slice(vertical=i)
                data2D.reorder('astra')
                ag2D = data2D.geometry
                ag2D.set_angles(ag2D.angles, initial_angle=0.0)
                ig2D = ag2D.get_ImageGeometry()
                device = 'gpu'
                initial = ig2d.allocate(0)
                A = ProjectionOperator(ig2d, ag2d, device)
                b = data2D
                alpha = 0.9
                F = LeastSquares(A, b)
                G = alpha*FGP_TV(device='gpu')


                reconstructor = FISTA(f=F, g=G, initial=initial)
                reconstructor.run(N_iter)
                reconstruction[i] = reconstructor.solution.copy()

        else:
            ig = self.data.geometry.get_ImageGeometry(resolution=1)
            self.data.reorder('astra')
            device = 'gpu'
            fbp = FBP(ig,self.data.geometry,device)
            reconstruction = fbp(self.data)

        return reconstruction


















