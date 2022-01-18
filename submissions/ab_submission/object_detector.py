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

from .follicleClassifier import follicleClassifier
from .dataLoaders import imageDataLoader

class ObjectDetector:
    def __init__(self, 
                preprocessing_boxPixelClassifier_convolve=30, 
                post_processing_boxPixelClassifier_convolve=10, 
                box_border = 1, 
                follicle_classifier_size=128,
                n_epochs=50,
                batch_size=16,
                ramp_mode=True
                ):
        """
        
        Parameters:
        -----------
        preprocessing_boxPixelClassifier_convolve: int, kernel size of the convolution performed before getting the pixel for classification
        post_processing_boxPixelClassifier_convolve: int, kernel size of the convolution performed after classification, used to estimate the density of prediction and thus weight the pixel use for random box generation
        box_border : float, proportion of the width and height for xmin, xmax, ymin and ymax determination of the border of the box
        follicle_classifier_size: int, heigh of the images sent to follicleClassifier
        n_epochs: int, number of epochs for follicle_classifier training
        batch_size: int, batch size for prediction, prevent cuda out of memory
        ramp_mode: boolean, if True execute code specific to ramp mode
        """
        # Class dictionnary
        self._class_dict = {
            0:"Negative",
            1:"Primordial",
            2:"Primary",
            3:"Secondary",
            4:"Tertiary"
        }

        # Device detection
        if torch.cuda.is_available():
            self.device="cuda:0"
        else:
            self.device="cpu"

        # Loading classifier
        self.boxPixelClassifier = DecisionTreeClassifier(max_depth=3, class_weight="balanced")
        self.follicleClassifier = follicleClassifier(device=self.device)

        # Parameters initialization
        self.preprocessing_boxPixelClassifier_convolve = preprocessing_boxPixelClassifier_convolve
        self.post_processing_boxPixelClassifier_convolve = post_processing_boxPixelClassifier_convolve
        self.box_border = box_border

        self.erosion_dilatation_kernel = np.ones((5,5))
        self.density_filter=np.ones((self.post_processing_boxPixelClassifier_convolve,self.post_processing_boxPixelClassifier_convolve))

        self.follicle_classifier_size = follicle_classifier_size
        self.n_epochs = n_epochs
        self.follicleClassifier_fitted_ = False
        self.batch_size = batch_size

        # Defining some pre-processing functions
        self.image_preprocessing = lambda x: cv2.filter2D(
            x, -1, np.ones(
                (preprocessing_boxPixelClassifier_convolve,preprocessing_boxPixelClassifier_convolve,1)
            )/(preprocessing_boxPixelClassifier_convolve**2)
        )

        self.ramp_mode = ramp_mode

    def save(self, boxPixelClassifier, follicleClassifier):
        """save_params

        Save the trained parameters of the model.

        Parameters
        ----------
        boxPixelClassifier: str, the path where to save the parameters of the boxPixel, if None, no export if performed
        follicleClassifier: str, the path where to save the parameters of the follicleClassifier, if None, no export if performed
        """
        
        if boxPixelClassifier is not None:
            dump(self.boxPixelClassifier, boxPixelClassifier)

        if follicleClassifier is not None:
            self.follicleClassifier.save_model(follicleClassifier)

    def load(self, boxPixelClassifier, follicleClassifier):
        """save_params

        Save the trained parameters of the model.

        Parameters
        ----------
        boxPixelClassifier: str, the path where to load the parameters of the boxPixel, if None, no loading if performed for boxPixelClassifier
        follicleClassifier: str, the path where to load the parameters of the follicleClassifier, if None, no loading if performed for follicleClassifier
        """

        if boxPixelClassifier is not None:
            self.boxPixelClassifier = load(boxPixelClassifier)
    
        if follicleClassifier is not None:
            self.follicleClassifier.load_model(follicleClassifier)

    def _get_box_list(self, image_loader, image_name):
        """Return the box list according to an image and to the box classifier

        Parameters
        ----------
        image_loader: image loader object
        image_name: str, filename of the image

        Output
        ------
        List of box in the format xmin, ymin, xmax, ymax
        """

        # Data for detection of pixel of interest
        pixel_data = image_loader.get_pixel_labels_sample(
            image_name,
            pre_processing_fullimage_func=self.image_preprocessing,
            all=True
        )
        X_pixel = pixel_data[0].mean(axis=1).reshape(-1,1)

        # Getting prediction in 2D format
        y_hat_pixel = self.boxPixelClassifier.predict(X_pixel)
        y_hat_pixel = y_hat_pixel.reshape(pixel_data[-1][0:2])

        # Post-processing

        ## Only keep the most dense area of pixels
        y_hat_pixel_morphology = y_hat_pixel.astype(np.uint8)
        erosion_threshold = y_hat_pixel_morphology.mean()/10
        y_hat_pixel_morphology_mean = y_hat_pixel_morphology.mean()
        while y_hat_pixel_morphology_mean > erosion_threshold and y_hat_pixel_morphology_mean > 0.01:
            y_hat_pixel_morphology = cv2.erode(y_hat_pixel_morphology, kernel=self.erosion_dilatation_kernel)
            y_hat_pixel_morphology_mean = y_hat_pixel_morphology.mean()

        ## Filtering pixel by density
        
        density_matrix = cv2.filter2D(y_hat_pixel_morphology, -1, self.density_filter)
        density_matrix = density_matrix/density_matrix.max()
        density_matrix[density_matrix < np.quantile(density_matrix[density_matrix != 0], 0.9)] = 0 # We cut the noise

        y_hat_pixel_density = density_matrix*y_hat_pixel_morphology
        y_hat_pixel_density = (y_hat_pixel_density == 1).astype(np.uint8)

        ## Erode again then dilate
        y_hat_pixel_density_erode = cv2.erode(y_hat_pixel_density, self.erosion_dilatation_kernel, iterations=3)
        y_hat_pixel_density_erode_dilate = cv2.dilate(y_hat_pixel_density_erode, self.erosion_dilatation_kernel, iterations=2)            

        # Getting box location
        y_hat_box = []
        box_ratio = self.follicleClassifier.box_ratio_
        box_size = self.follicleClassifier.box_size_
        contours, _ = cv2.findContours(y_hat_pixel_density_erode_dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            x_min, y_min, x_max, y_max = x-w*self.box_border, y-h*self.box_border, x+w*(1+self.box_border), y+h*(1+self.box_border)

            # Fixing the border conditions
            y_min, x_min = tuple([x if x > 0 else 0 for x in [y_min, x_min]])
            y_max, x_max = tuple([x if x < y else y for x, y in zip([y_max, x_max], list(y_hat_pixel_density_erode_dilate.shape))])
            
            nh, nw = y_max-y_min, x_max-x_min
            if (nh)/(nw) <= box_ratio and nh*nw >= box_size :
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                y_hat_box.append((x_min, y_min, x_max, y_max))

        return y_hat_box

    def fit(self, X, y):
        """fit function

        The main idea here is :
         - To train a classifier to detect pixel of interest for box generation
         - To Train a classifier to classify images from a box

        Parameters
        ----------
        X: list of file path
        y: pandas object containing the labels
        """

        # Loading data with the imageLoader

        ## We have to deal with problem input
        if self.ramp_mode:
            class_to_index = dict(zip(self._class_dict.values(), self._class_dict.keys()))
            y = pd.concat([pd.DataFrame(sample).assign(filename = file_path.split("/")[-1])
                for file_path, sample in zip(X,y)
            ])
            for lab, id in zip(["xmin","ymin","xmax","ymax"], list(range(4))):
                y[lab] = y["bbox"].apply(lambda x: x[id])
            y["label"] = y["class"].apply(lambda x: class_to_index[x])

        image_loader = imageDataLoader(X, y[["filename","xmin","xmax","ymin","ymax","label"]])

        # 1. Training the boxPixelClassifier
        print("Fitting boxPixelClassifier")

        # Skipping if already fitted

        if "tree_" not in dir(self.boxPixelClassifier):
            ## Getting the train dataset :
            train_dataset = list(
                image_loader.get_pixel_labels_samples(
                    random_pick=0.01, 
                    random_state=42, 
                    pre_processing_fullimage_func=self.image_preprocessing)
            )
            X, y = (np.concatenate([j[i] for j in train_dataset]) for i in [0,1])
            X = X.mean(axis=1).reshape(-1,1)
            y = (y != 0).astype("int")

            self.boxPixelClassifier.fit(X,y)

        # 2. Training image classifier
        print("Fitting follicleClassifier")

        # Skipping if already fitted
        if "box_ratio_" not in dir(self.follicleClassifier):

            # Constitution of the dataset : original cropped image, and box detected by the algorithm
            original_boxes = image_loader.y_box

            # Also learning the correct h/w box ratio
            box_ratio = np.quantile([(y[3]-y[2])/(y[1]-y[0]) for y in original_boxes], 0.95)*1.5
            box_size = np.quantile([(y[3]-y[2])*(y[1]-y[0]) for y in original_boxes], 0.01)/2

            # Storing the correct box ratio in the classifier
            self.follicleClassifier.box_ratio_ = box_ratio
            self.follicleClassifier.box_size_ = box_size

            dataset_for_follicle_classifier = []
            for filename in image_loader.X_filenames:

                # Original image
                original_data = image_loader.get_sample(filename)
                original_image_crop, _ = image_loader.get_crop(original_data[0], original_data[1])

                ## Dataset from original data
                dataset_for_follicle_classifier_original = [original_image_crop, original_data[2]]

                ##  Dataset from boxes
                ### Here we compute a matrix of zeros with the location of original labelled data
                ### We only keep box which intersect with theses
                boxes = self._get_box_list(image_loader=image_loader, image_name=filename)

                label_matrix = np.zeros(original_data[-1][0:2])
                for original_box, original_label in zip(original_data[1], original_data[2]):
                    label_matrix[original_box[2]:original_box[3],original_box[0]:original_box[1]] = original_label+1

                dataset_for_follicle_classifier_box_data = []
                dataset_for_follicle_classifier_box_label = []
                for box in boxes:
                    tmp_matrix = label_matrix[box[1]:box[3],box[0]:box[2]]
                    if np.max(tmp_matrix) != 0:        
                        area = tmp_matrix.shape[0]*tmp_matrix.shape[1]
                        n_pixels = (tmp_matrix != 0).sum()
                        
                        if area/n_pixels > 0.5:
                            box_label = np.argmax(np.bincount(tmp_matrix[tmp_matrix != 0].astype("int8")))-1
                            box_data = original_data[0][box[1]:box[3], box[0]:box[2]]

                            dataset_for_follicle_classifier_box_data.append(box_data)
                            dataset_for_follicle_classifier_box_label.append(box_label)

                if len(dataset_for_follicle_classifier) != 0:
                    dataset_for_follicle_classifier.append(
                        dataset_for_follicle_classifier_original
                    )

                if len(dataset_for_follicle_classifier_box_data) != 0:
                    dataset_for_follicle_classifier.append(
                        [dataset_for_follicle_classifier_box_data, dataset_for_follicle_classifier_box_label]
                    )

            tensors_for_follicle_classifier = []

            for data in dataset_for_follicle_classifier:
                x = [cv2.resize(
                        image, 
                        (self.follicle_classifier_size,
                        int(image.shape[0]*self.follicle_classifier_size/image.shape[1]))
                    ) for image in data[0] if len(image.shape) == 3]

                # Padding to get a dataset of same size everywhere
                if len(x) > 0:
                    x = nn.utils.rnn.pad_sequence([torch.tensor(data, dtype=torch.float32) for data in x], batch_first=True)
                    x = torch.moveaxis(x, 3, 1)
                    x = x/255 # VGG requires a normalized pixel intensity

                    # One hot encoding of the labels
                    y = nn.functional.one_hot(
                        torch.tensor(data[1], dtype=torch.int64), 
                        num_classes=5
                    ).float()

                    tensors_for_follicle_classifier.append((x,y))

            for i in range(self.n_epochs):
                for x,y in tensors_for_follicle_classifier:
                    n_batch = x.shape[0]//self.batch_size + int(x.shape[0]%self.batch_size)
                    for batch in range(n_batch):
                        x_temp = x[batch*self.batch_size:(batch+1)*self.batch_size].to(self.device)
                        y_temp = y[batch*self.batch_size:(batch+1)*self.batch_size].to(self.device)
                        
                        if x_temp.shape[0] > 1:
                            self.follicleClassifier.fit(x_temp,y_temp)

        return self

    def predict(self, X):
        """predict function

        The main idea here is :
         - Get the pixel of interest
         - Draw box around this pixel
         - Provide a classification of theses box

        Parameters
        ----------
        X: list of file path
        """

        # Loading data with the imageLoader
        image_loader = imageDataLoader(X)

        # Generating prediction
        predictions = []

        for image_name in image_loader.X_filenames:

            # Getting box location
            y_hat_box = self._get_box_list(image_loader, image_name)

            # Classification of the box
            # Get the full image

            full_image = image_loader.get_sample(image_name)[0]
            y_hat_box_image = [] # Array containing the images
            for box in y_hat_box:
                box_image = full_image[box[1]:box[3], box[0]:box[2]]
                y_hat_box_image.append(box_image)
            
            x = [cv2.resize(
                image, 
                (self.follicle_classifier_size,
                int(image.shape[0]*self.follicle_classifier_size/image.shape[1]))
            ) for image in y_hat_box_image]

            # Padding to get a dataset of same size everywhere
            x = nn.utils.rnn.pad_sequence([torch.tensor(data, dtype=torch.float32) for data in x], batch_first=True)
            x = torch.moveaxis(x, 3, 1)
            x = x/255 # VGG requires a normalized pixel intensity

            # Getting labels

            ## Getting prediction in batch
            n_batch = x.shape[0]//self.batch_size + int(x.shape[0]%self.batch_size)
            y_hat_proba_temp_array = []
            for batch in range(n_batch):
                x_temp = x[batch*self.batch_size:(batch+1)*self.batch_size].to(self.device)
                y_hat_proba_temp = self.follicleClassifier.predict(x_temp).cpu()
                y_hat_proba_temp_array.append(y_hat_proba_temp)
            y_hat_proba = torch.concat(y_hat_proba_temp_array, dim=0)

            y_hat_proba, y_hat_labels_id = torch.max(y_hat_proba, dim=1)
            y_hat_labels = np.vectorize(lambda x: self._class_dict[x])(y_hat_labels_id)

            predictions.append([
                dict(zip(
                    ["proba","bbox","class"],
                    pred
                )) for pred in zip(y_hat_proba.numpy().tolist(), 
                                    y_hat_box, 
                                    y_hat_labels
                                )
            ])
        
        y_pred = np.empty(len(X), dtype=object)
        y_pred[:] = predictions

        return y_pred