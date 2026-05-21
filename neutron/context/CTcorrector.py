import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
previous_metric_value = None
import plotly.graph_objects as go
import random
from cil.framework.acquisition_data import AcquisitionData


class CTcorrector():
    def __init__(self,data=None, angles = None):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data,AcquisitionData):
            self.data = data.as_array()


        self.cor_data = []
        self.tilt_data = []
        self.k_angle = 0
        self.alpha = None
        self.t = None
        self.callback = 1
        self.labels = None
        if angles is not None:
            self.angles = np.array(angles)
            self.opposite_index_distance = self.set_opposite_index_distance
        else:
            self.angles = None


    def load_data(self,data):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data,AcquisitionData):
            self.data = data.as_array()


    def set_angles(self,angles):
        self.angles = np.array(angles)
        self.opposite_index_distance = self.set_opposite_index_distance()



    def set_opposite_index_distance(self, index_distance = None):
        if self.angles[0]>0.001:
            angles = self.angles
        elif self.angles[-1] <359.99:
            angles = self.angles
        else:
            angles = self.angles[:-1]

        if index_distance is None:
            index = np.argmin(np.abs(angles - 90))
            angle_at_index = angles[index]
            opposite_index = np.argmin(np.abs(angles -(angle_at_index + 180)))
            index_distance = opposite_index - index
            return index_distance
        else:
            self.opposite_index_distance = index_distance

    def set_labels(self,labels=None):
        if labels is None:
            (n0,n1,n2) = np.shape(self.data)
            if len(self.angles) == n0:
                labels = ['a', 'v', 'h']
            elif len(self.angles) == n1:
                labels = ['v','a', 'h']
            elif len(self.angles) == n2:
                labels = ['v', 'h', 'a']
            else:
                raise AttributeError('Please specify axis labels!!, ["v", "a", "h"]')
        else:
            if len(labelse) != 3:
                raise ValueError('Length of labels variable should be = 3')

        self.labels = labels

    def get_projection_and_opposite(self, k_angle=None):
        if k_angle is not None:
            p180 = self.opposite_index_distance
            k = np.argmin(np.abs(self.angles - k_angle))
        else:
            k = self.k_current

        if self.labels is None:
            self.labels = self.set_labels()

        if self.angles[-1] > 359.99:
            kp180 = (k+p180)%(len(self.angles)-1)
        else:
            kp180 = (k+p180)%len(self.angles)

        if self.labels == ['a', 'v', 'h']:
            img0 = self.data[k,:,:]
            img180 = self.data[kp180,:,:]
        elif self.labels == ['v','a','h']:
            img0 = self.data[:,k,:]
            img180 = self.data[:,kp180,:]
        elif self.labels == ['v', 'h', 'a']:
            img0 = self.data[:,:,k]
            img180 = self.data[:,:,kp180]
        else:
            raise AttributeError('Please specify labels correctly: ["v", "a", "h"]')

        self.fixed = sitk.GetImageFromArray(img0)
        self.fixed = self.set_image_details(self.fixed)
        self.moving = sitk.GetImageFromArray(img180)
        self.moving = self.set_image_details(self.moving)


    def register(self, learning_rate = 0.1, sampling_percentage = 0.1,
                     max_iter = 50,
                    metric_type = 'cor', optimizer_type = 'gd',
                     smoothing = 0,
                      shrinking = 1):
        """
        Perform the image registration using the provided transformation parameters.

        Args:
            transform_params: The transformation parameters (e.g., translation, rotation).
            metric_type (str): The metric used for the registration (e.g., 'MeanSquares', 'Mattes').
            optimizer_type (str): The optimizer used for the registration (e.g., 'GradientDescent').

        Returns:
            SimpleITK.Transform: The resulting transformation after registration.
        """

        if (self.alpha is None) or (self.t is None):
            initial_transform = get_composite_transform()
        else:
            initial_transform = get_composite_transform(alpha = self.alpha,t = self.t)


        registration = sitk.ImageRegistrationMethod()
        registration.SetInitialTransform(initial_transform, inPlace=True)
        fixed_d = self.fixed
        moving_d = self.moving


        if metric_type == 'cor':
            registration.SetMetricAsCorrelation()
        elif metric_type == 'ms':
            registration.SetMetricAsMeanSquares()
        else:
            raise ValueError('Specify metric type: "cor"')

        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(sampling_percentage)

        if optimizer_type == 'gd':
            registration.SetOptimizerAsGradientDescent(
            learningRate=learning_rate,
            numberOfIterations=max_iter,
            convergenceMinimumValue=-1e-16,
            convergenceWindowSize=1000
        )


        else:
            raise ValueError('Specify optimization algorithm: Only "gd" currently supported')

        registration.SetInterpolator(sitk.sitkLinear)


        self.metric_values = []
        self.iterations = []
        if self.callback==1:
            self.metric_values = []
            self.iterations = []
            registration.AddCommand(sitk.sitkIterationEvent, lambda: self.registration_callback(
            registration.GetOptimizerIteration(),
            registration.GetMetricValue(), every_N = max_iter//10,
            learning_rate =registration.GetOptimizerLearningRate(),
            method = registration))

        final_transform = registration.Execute(fixed_d, moving_d)
        optimal_parameters = final_transform.GetParameters()
        alpha = np.rad2deg(optimal_parameters[0])  # Convert to degrees
        t = optimal_parameters[1]

        self.alpha = alpha
        self.t = t

    def set_image_details(self,image):
        i_size = image.GetSize()
        max_i = i_size[0]
        i_space = image.GetSpacing()
        image.SetSpacing((1.5/max_i, 1.5/max_i))
        i_space = image.GetSpacing()
        new_origin = [-0.5 * (i_size[i] - 1) * i_space[i] for i in range(len(i_size))]
        image.SetOrigin(new_origin)
        return image


    def registration_callback(self,iteration, metric_value,every_N, learning_rate, method):
        global previous_metric_value
        initial_transform = method.GetInitialTransform()
        self.metric_value = metric_value

        if iteration == 2:
            self.first_metric_value = metric_value

        if not iteration % every_N:
            self.metric_values.append(metric_value)
            self.iterations.append(iteration)
            if previous_metric_value is not None:
                metric_difference = abs(previous_metric_value - metric_value)
                print(f"Iteration {iteration}: Metric Value = {metric_value:.4f}, Metric Difference = {metric_difference:.4f}")
            else:
                print(f"Iteration {iteration}: Metric Value = {metric_value}")

            #print('The current learning rate is', learning_rate)
            parameters = initial_transform.GetParameters()
            alpha = np.rad2deg(parameters[0])  # Convert to degrees
            t = parameters[1]
            scale = self.fixed.GetSpacing()[0]
            print('The parameters are: \n  t  = ', t/scale, ' pixels, \nalpha = ', alpha, ' degrees', t)

            # Update previous_metric_value for the next iteration
            previous_metric_value = metric_value

    def print_parameters(self,registration_method):
        # Get the current transform
        transform = registration_method.GetInitialTransform()
        # Print the parameters of the transform
        print(f"Transform Parameters: {transform.GetParameters()}")
        print()

    def transformation(self, type='Euler3DTransform', matrix=None,translation=(0,0,0)):
        """
        Returns a transformation object based on the provided transformation parameters.

        Args:
            transform_parameters: The parameters that define the transformation (e.g., rotation, translation).
            input: type='Similarity3Dtransform, Centroid, Euler3DTransform'

        Returns:
            SimpleITK.Transform: The transformation object.
        """
        if type == 'Similarity3DTransform':
            transform = sitk.Similarity3DTransform()
            transform.SetMatrix(matrix)
            transform.SetTranslation(translation)
        elif type == 'Euler3DTransform':
            transform = sitk.Euler3DTransform()
            transform.SetMatrix(matrix)
            transform.SetTranslation(translation)
        elif type == 'Affine':
            transform = sitk.AffineTransform(3)
            transform.SetMatrix(matrix)
            transform.SetTranslation(translation)

        return transform


    def resample(self, data = None, transform=None, interpolation_type: str = 'linear'):
        """
        Resample the moving image using the given transformation object.

        Args:
            transform (SimpleITK.Transform): The transformation object.
            interpolation_type (str, optional): The interpolation type ('linear' or 'nearest'). Default is 'linear'.
            inplace (bool, optional): Whether to overwrite the moving image or return a new resampled image. Default is True.

        Returns:
            SimpleITK.Image: The resampled image (if inplace=False).
        """
        if isinstance(data,np.ndarray):
            data = sitk.GetImageFromArray(data)
        elif data is None:
            raise ValueError('Data cannot be None')

        if interpolation_type == 'linear':
            interpolation_method = sitk.sitkLinear
        elif interpolation_type == 'nearest':
            interpolation_method = sitk.sitk.NearestNeighbor

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(data)  # Reference image (fixed)
        resampler.SetInterpolator(interpolation_method)   # Interpolation method
        resampler.SetTransform(transform)    # Apply the initial transform (aligned centroids)
        resampler.SetOutputPixelType(data.GetPixelID())
        resampler.SetOutputSpacing(data.GetSpacing())  # Ensure the spacing is preserved
        resampler.SetOutputOrigin(data.GetOrigin())  # Preserve origin
        resampler.SetOutputDirection(data.GetDirection())

        return sitk.GetArrayFromImage(resampler.Execute(data))  # Resample the moving image


def create_rotation_transform(angle_in_degrees, center):
    angle_in_radians = np.deg2rad(angle_in_degrees)
    rotation_transform = sitk.Euler2DTransform(center, angle_in_radians)
    return rotation_transform

# Define translation
def create_translation_transform(tx, ty):
    translation_transform = sitk.TranslationTransform(2)
    translation_transform.SetOffset((tx, ty))
    return translation_transform

# Define flip along the y-axis as an affine transform
def create_flip_transform(center):
    flip_transform = sitk.AffineTransform(2)
    matrix = [-1, 0, 0, 1]  # Flipping along y-axis
    flip_transform.SetMatrix(matrix)
    flip_transform.SetCenter(center)
    return flip_transform

def get_composite_transform(alpha = 0,t = 0):
    alpha = 0  # Rotation angle in degrees
    t = 0.00  # Translation in x-direction

    center = [0,0]

    # Create the sequence of transforms
    rotation1 = create_rotation_transform(alpha, center=center)
    translation1 = create_translation_transform(-t, 0)
    flip = create_flip_transform(center=center)
    translation2 = create_translation_transform(t, 0)
    rotation2 = create_rotation_transform(-alpha, center=center)

    # Combine into a composite transform
    composite_transform = sitk.CompositeTransform(2)
    composite_transform.AddTransform(rotation1)
    composite_transform.AddTransform(translation1)
    composite_transform.AddTransform(flip)
    composite_transform.AddTransform(translation2)
    composite_transform.AddTransform(rotation2)
    return composite_transform
