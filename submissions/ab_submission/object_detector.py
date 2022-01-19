import numpy as np
import pandas as pd
import cv2
import importlib
import os
import sys
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
import torch
from torch.utils.data import DataLoader

# Adding script folder to the PYHONPATH env variable
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path)

from follicleClassifier import follicleClassifier
from dataLoaders import imageDataLoader
from dataset import dataloader_collapse, folliclesDataset

class ObjectDetector:
    def __init__(self, 
                preprocessing_boxPixelClassifier_convolve=30, 
                post_processing_boxPixelClassifier_convolve=10, 
                box_border = 1.5, 
                follicle_classifier_size=128,
                n_epochs=50,
                batch_size=64,
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
        if ramp_mode == False:
            self.batch_size = batch_size
        else:
            self.batch_size = int(batch_size/4) # Ramp may do somes parallelization or something, but it goes easier in OOM

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

            # Also learning the correct h/w box ratio
            original_boxes = image_loader.y_box
            box_ratio = np.quantile([(y[3]-y[2])/(y[1]-y[0]) for y in original_boxes], 0.95)*1.5
            box_size = np.quantile([(y[3]-y[2])*(y[1]-y[0]) for y in original_boxes], 0.01)
            self.follicleClassifier.box_ratio_ = box_ratio
            self.follicleClassifier.box_size_ = box_size

            # Getting dataLoader
            train_dataset = folliclesDataset(
                image_loader,
                data_augmentation=False,
                local_path=None,
                box_classifier=self._get_box_list,
                in_memory=True,
                verbose=False,
                order="default",
                force_reload=False,
                mode="all"
            )
            
            dataloader_collapse_train = lambda x: dataloader_collapse(x, image_size_width=self.follicle_classifier_size, reducer="resize", random_flip=True)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=dataloader_collapse_train)

            for i in range(self.n_epochs):
                for x, y in train_dataloader:
                    
                    x = x.to(self.device)
                    y = y.to(self.device)

                    self.follicleClassifier.fit(x,y)

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

        # Getting dataloader
        predict_dataset = folliclesDataset(
            image_loader,
            data_augmentation=False,
            local_path=None,
            box_classifier=self._get_box_list,
            in_memory=True,
            verbose=False,
            order="box_ratio",
            force_reload=False,
            mode="all"
        )
        dataloader_collapse_predict = lambda x: dataloader_collapse(x, image_size_width=self.follicle_classifier_size, reducer="resize", random_flip=False)

        # Generating prediction
        predictions = []

        for image in image_loader.X_filenames:
            
            # Getting the images
            x = predict_dataset[image]
            x_metadata = [data[1] for data in x]
            
            y_hat_box = []
            y_hat_proba = []
            y_hat_labels = []

            # Looping over boxes, the idea is to not resize them for the prediction
            for i in range(len(x)):
                x_sample = [x[i]]
                x_sample_box = x_metadata[i]["bbox"]
                x_sample, _ = dataloader_collapse_predict(x_sample)

                # We need to check that the image is big enough to be processed
                if x_sample.shape[2] >= 40 and x_sample.shape[3] >= 40:
                    x_sample = x_sample.to(self.device)
                    y_hat_proba_sample = self.follicleClassifier.predict(x_sample).cpu()
                    y_hat_proba_sample, y_hat_labels_sample_id = torch.max(y_hat_proba_sample, dim=1)
                    y_hat_labels_sample = np.vectorize(lambda x: self._class_dict[x])(y_hat_labels_sample_id)

                    y_hat_box.append(x_sample_box)
                    y_hat_proba.append(y_hat_proba_sample[0].item())
                    y_hat_labels.append(y_hat_labels_sample[0])

            predictions.append([
                dict(zip(
                    ["proba","bbox","class"],
                    pred
                )) for pred in zip(y_hat_proba, 
                                    y_hat_box, 
                                    y_hat_labels
                                )
            ])
        
        y_pred = np.empty(len(X), dtype=object)
        y_pred[:] = predictions

        return y_pred