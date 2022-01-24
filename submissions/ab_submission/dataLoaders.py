import numpy as np
import pandas as pd
from sklearn.utils import check_array
from sklearn.tree import DecisionTreeClassifier
from scipy.ndimage import morphology
from types import FunctionType
import torch.nn as nn
from torch.optim import Adam
import torch
from torchvision import transforms
from torchvision.models import vgg16
import random
import imageio
import cv2
from joblib import dump, load
import os

class imageDataLoader():
    """imageDataLoader

    Class for dataset loading.
    This class provide numpy array given the image path and the image labels.
    It will be used to feed the different classifiers of the current project.
    """

    def __init__ (self, X, y=None):
        """imageDataLoader
        
        Initialisation of the image dataloader.

        Parameters:
        ----------
        X : np.array([str]) or [str], image paths
        y : np.array of size (n, 6), containing : filenames, xmin, xmax, ymin, ymax and labels. Can be None. 
        """


        self.X = check_array(X, ensure_2d=False, dtype=str)
        self.X_filenames = [os.path.basename(x) for x in X]
        
        if y is not None:
            # Checking the format of the input
            if y.shape[1] <= 5:
                raise ValueError("The input y should be of size 6.")

            self.has_y = True
            y = check_array(y, dtype=None)
            
            # Storing the y_box
            self.y_box = check_array(y[:,1:-1], dtype=int)

            # Storing the y_label
            self.y_label = check_array(y[:,-1], ensure_2d=False)
            
            # Storing the filenames
            self.y_filenames = check_array(y[:,0], ensure_2d=False, dtype=str)

            # Creating a list of mask for X-y correspondance
            self.xy_mask = [x == self.y_filenames for x in self.X_filenames]

        else:
            self.has_y = False 

    def __len__ (self):
        return len(self.X_filenames)

    def _load_image(self, image_path):
        """load_image

            Function that load the image given its path

            Parameters
            ----------
            image_path: str, path of the image

            Output
            ------
            Numpy array of size (w,h,3) with w the width of the image, h the height of the image
        """

        image_data = imageio.imread(image_path)

        return image_data

    def __generate_augmented_images(self, image_matrix, rotations=[0,90,180,270], flip=[-1,0]):
        """Generation of augmentated image dataset

        For a given image this function generate an augmented dataset of it.
        The generation is performed by doing rotation and flipping

        Parameters
        ----------
        image_matrix: Numpy object of dimension (w,h), w and h being the widt and the height of the image
        rotations: [int], angle in which the rotation should be performed, should contains 0 for keeping the original rotation
        flip: [int], axis in which to flip, should contains -1 for keeping the original flipping
        """

        rotations_code = {
            90:cv2.ROTATE_90_CLOCKWISE,
            180:cv2.ROTATE_180,
            270:cv2.ROTATE_90_COUNTERCLOCKWISE
        }

        augmented_image_matrix = []
        for x in rotations:
            for y in flip:
                if y in [0,1]:
                    flipped_image = np.flip(image_matrix, axis=y)
                if y == -1:
                    flipped_image = image_matrix
                
                if x != 0:
                    flipped_rotated_image = cv2.rotate(flipped_image, rotations_code[x])
                else:
                    flipped_rotated_image = flipped_image
                augmented_image_matrix.append(flipped_rotated_image)

        return augmented_image_matrix

    def get_crop(self, image_array, box_array, image_labels = None, data_augmentation=False):
        """Get crop data given an image and box array

        Parameters
        ----------
        image_array: numpy array of dimension (h,w)
        box_array: list of coordonates, xmin xmax ymin ymax for each box
        image_labels: list of labels, can be None if data_augmentation is False
        data_augmentation: boolean, default False, if true the function return an augmented dataset.

        Output
        ------
        Tuple with :
            list of cropped images
            list of image labels
        """

        cropped_images = []

        # Cropping the image
        for box in box_array:
            cropped_image = image_array[box[2]:box[3],box[0]:box[1]]

            # Dataset augmentation
            image_labels_repeat = [] # List of repetition of labels
            if data_augmentation == True:
                if image_labels is None:
                    raise Exception("image_labels are needed to generate the cropped data")

                augmented_images = self.__generate_augmented_images(cropped_image)
                cropped_images += augmented_images

                # Generating augmented labels
                image_labels_repeat.append(len(augmented_images))
            else:
                cropped_images.append(cropped_image)

        if len(image_labels_repeat):
            image_labels = np.repeat(image_labels, image_labels_repeat)

        return cropped_images, image_labels

    def get_sample(self, image_name, crop=False, pre_processing_fullimage_func=None, data_augmentation=False):
        """get_samples

            Get the data of an image

            Parameters
            ----------
            image_name: str, name of the image to get
            crop: boolean, default False, if true the function return samples of image cropped for each box
            data_augmentation: boolean, default False, if true the function return an augmented dataset. For computational reason, this operation is only performed when crop is set to true.
            pre_processing_fullimage_func: python function, which is applied to the full image. Set it to None if no function is applied.

            Output
            ------
            (image_data, image_box, image_labels, original_image)
            Tuple of numpy array containing :
            - image_data:
                Numpy array of size (w,h,3) with w the width of the image, h the height of the image if crop is false
                List of n numpy array of size (w, h, 3) with n the number of box if crop is true
            - image_box: Numpy array of size (n, 4) with n the number of box in the image and 4 representing : xmin, xmax, ymin and ymax. None if there is no label.
            - image_labels: Numpy array of size (n, ) with n the number of box in the image. None if there is no label.
            - image_shape : shape of the original image numpy array
        """

        if isinstance(image_name, str) is False or image_name not in self.X_filenames:
            raise ValueError("The image name is invalid, check the format (should be a string) of the name and the existence of the file.")

        # Parameters checks
        if crop is True and self.has_y is False:
            raise Exception("The box sampling is not permitted if the labels are not available.")
        if pre_processing_fullimage_func is not None and isinstance(pre_processing_fullimage_func, FunctionType) is False:
            raise ValueError("pre_processing_func is expected to be a function")

        # Loading the image
        i = self.X_filenames.index(image_name)
        image_path = self.X[i]
        image_data = self._load_image(image_path)

        # Loading labels and box
        if self.has_y:
            image_box = self.y_box[self.xy_mask[i]]
            image_labels = self.y_label[self.xy_mask[i]]
        else:
            image_box, image_labels = (None, None)

        # Applying pre_processing function to full image
        if pre_processing_fullimage_func is not None:
            image_data_preprocessed = pre_processing_fullimage_func(image_data)
        else:
            image_data_preprocessed = image_data

        # Cropping data
        if crop == True:
            # Cropping the image
            output_image, _ = self.get_crop(image_data_preprocessed, image_box)
        else:
            output_image = image_data

        image_shape = image_data.shape

        return output_image, image_box, image_labels, image_shape

    def get_samples(self, crop=False, pre_processing_fullimage_func=None, data_augmentation=False):
        """get_samples

            Get an iterator of the data.

            Parameters
            ----------
            crop: boolean, default False, if true the function return samples of images cropped for each box
            data_augmentation: boolean, default False, if true the function return an augmented dataset. For computational reason, this operation is only performed when crop is set to true.
            pre_processing_fullimage_func: python function, which is applied to the full image. Set it to None if no function is applied.

            Output
            ------
            (image_data, image_box, image_labels, original_image)
            Tuple of numpy array containing :
            - image_data:
                Numpy array of size (w,h,3) with w the width of the image, h the height of the image if crop is false
                List of n numpy array of size (w, h, 3) with n the number of box if crop is true
            - image_box: Numpy array of size (n, 4) with n the number of box in the image and 4 representing : xmin, xmax, ymin and ymax. None if there is no label.
            - image_labels: Numpy array of size (n, ) with n the number of box in the image. None if there is no label.
            - image_shape : shape of the original image numpy array
        """

        for i in range(len(self.X_filenames)):
            # Loading the data
            image_name = self.X_filenames[i]
            output_image, image_box, image_labels, image_shape = self.get_sample(image_name, crop=crop, pre_processing_fullimage_func=pre_processing_fullimage_func, data_augmentation=data_augmentation)

            yield output_image, image_box, image_labels, image_shape

    def _normalize_image(self, image):
        """_normalize_image

        This function normalize the image value between 0 and 1 for each layer.
        The idea behind this is to normalize the image according to itself, we want to make sure that it's pixel intensity is not "shift" and is always between 0 and 1.

        Parameters
        ----------
        image: numpy array of size (w, h, 3) containing the image. With w and h the image width and height.

        Output:
        -------
        Numpy array of size (w, h, 3) containing the normalized image.
        """

        # The idea behind this is to normalize the image according to itself, we want to make sure that it's pixel intensity is not "shift" and is always between 0 and 1.
        image_normalized = (image-image.mean(axis=(0,1)))/(image.max(axis=(0,1))-image.min(axis=(0,1)))
        image_normalized = (image_normalized-image_normalized.min(axis=(0,1)))

        return image_normalized

    def get_pixel_labels_sample(self, image_name, pre_processing_fullimage_func=None, normalize=True, random_pick=None, random_state=None, all = False):
        """get_pixel_samples
    
        For an image, it returns a pixel label dataset.
        Each pixel which is in a box is associated with its label.
        If the current dataset do not contains box, all the pixels of the image are returned

        Parameters
        ----------
        image_name: str, name of the image to get
        pre_processing_fullimage_func: python function, which is applied to the full image before generating de pixel - label sample. Set it to None if no function is applied.
        normalize: boolean, if true the pixel intensity are normalized between 0 and 1 for each color layer, the normalization is performed before appliance of the pre-processing function
        random_pick: float, is not None, is precise the percentage of pixels to randomly select for each image
        random_state: seed for random picking, not used if None
        all: boolean, if True, all the pixels of the image are returned even if it contains box

        Output
        ------
        (X, y, original_image) : tuple 
            X : numpy array of size (w*h, 3) with w and h the image width and height and y of size (w*h, 1)
            y : if there is not label, the y is None.
            image_shape : original image shape
        """

        # Input checks
        if isinstance(image_name, str) is False or image_name not in self.X_filenames:
            raise ValueError("The image name is invalid, check the format (should be a string) of the name and the existence of the file.")
        if random_state is not None and isinstance(random_state, int) is False:
            raise ValueError("random_state should be an integer or None")
        if random_pick is not None and (isinstance(random_pick, float) is False or not (0 < random_pick <= 1)):
            raise ValueError("random_state should be an float between 0 and 1 or None")

        if random_state is not None:
            random.seed(random_state)

        # Getting the data
        if self.has_y and all == False:
            crop = True
        else:
            crop = False

        image_data = self.get_sample(image_name, crop=crop, pre_processing_fullimage_func=pre_processing_fullimage_func, data_augmentation=False)
        
        # We need to perform it in a loop because the images are of different size
        images_data = []
        labels_data = []

        if crop:
            images_dataset = zip(image_data[0], image_data[2])
        else:
            images_dataset = [(image_data[0], None)]

        for (image_data_box, image_label_box) in images_dataset:
            
            # Normalization of the image
            image_data_normalized = self._normalize_image(image_data_box)

            # Then we flatten the array according to each color channel
            image_data_normalized_shape = image_data_normalized.shape
            image_data_normalized_flatten = image_data_normalized.reshape(image_data_normalized_shape[0]*image_data_normalized_shape[1], 3)

            # Randomly picking
            if random_pick is not None:
                random_idx = random.sample(
                    list(range(0, image_data_normalized_flatten.shape[0])),
                    k=int(random_pick*image_data_normalized_flatten.shape[0])
                )
                image_data_normalized_flatten = image_data_normalized_flatten[random_idx, :]

            # Getting the y array
            image_label_box_repeated = np.repeat(image_label_box, image_data_normalized_flatten.shape[0])

            # Storing data
            images_data.append(image_data_normalized_flatten)
            labels_data.append(image_label_box_repeated)

        X, y = (np.concatenate(x) for x in (images_data, labels_data))

        return X, y, image_data[-1]

    def get_pixel_labels_samples (self, pre_processing_fullimage_func=None, normalize=True, random_pick=None, random_state=None):
        """get_pixel_samples
        
            For each image, it returns a pixel label dataset.
            Each pixel which is in a box is associated with its label.
            If the current dataset do not contains box, all the pixels of the image are returned

            Parameters
            ----------
            pre_processing_fullimage_func: python function, which is applied to the full image before generating de pixel - label sample. Set it to None if no function is applied.
            normalize: boolean, if true the pixel intensity are normalized between 0 and 1 for each color layer, the normalization is performed before appliance of the pre-processing function
            random_pick: float, is not None, is precise the percentage of pixels to randomly select for each image
            random_state: seed for random picking, not used if None

            Output
            ------
            (X, y, original_image) : tuple 
                X : numpy array of size (w*h, 3) with w and h the image width and height and y of size (w*h, 1)
                y : if there is not label, the y is None.
                original_image : original image before cropping and pre-processing
        """

        for i in range(len(self.X_filenames)):
            image_name = self.X_filenames[i]
            X, y, original_data = self.get_pixel_labels_sample(image_name, pre_processing_fullimage_func=pre_processing_fullimage_func, normalize=normalize, random_pick=random_pick, random_state=random_state)

            yield X, y, original_data

