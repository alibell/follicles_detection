import numpy as np
import pandas as pd
import cv2
import importlib
import os
import sys
from functools import partial
import torch
from torch.utils.data import DataLoader

# Adding script folder to the PYHONPATH env variable
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path)

from follicleClassifier import follicleClassifier
from dataLoaders import imageDataLoader
from follicleDetector import follicleDetector, follicle_detector_lambda
from dataset import dataloader_collapse, folliclesDataset

class ObjectDetector:
    def __init__(self, 
                follicle_classifier_size=128,
                n_epochs=100,
                batch_size=64,
                ramp_mode=True,
                verbose_train=False
            ):
        """
        
        Parameters:
        -----------
        follicle_classifier_size: int, heigh of the images sent to follicleClassifier
        n_epochs: int, number of epochs for follicle_classifier training
        batch_size: int, batch size for prediction, prevent cuda out of memory
        ramp_mode: boolean, if True execute code specific to ramp mode
        verbose_train: boolean, if True, a log of the train is displayed
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
        self.follicleDetector = follicleDetector()
        self.follicleClassifier = follicleClassifier(device=self.device)

        # Parameters for neural network training
        self.follicle_classifier_size = follicle_classifier_size
        self.n_epochs = n_epochs
        self.verbose_train = verbose_train

        self.follicleClassifier_fitted_ = False

        if ramp_mode == False:
            self.batch_size = batch_size
        else:
            self.batch_size = int(batch_size/4) # Ramp may do somes parallelization or something, but it goes easier in OOM

        self.ramp_mode = ramp_mode

    def save(self, follicleDetector, follicleClassifier):
        """save_params

        Save the trained parameters of the model.

        Parameters
        ----------
        follicleDetector: str, the path where to save the parameters of the boxPixel, if None, no export if performed
        follicleClassifier: str, the path where to save the parameters of the follicleClassifier, if None, no export if performed
        """
        
        if follicleDetector is not None:
            self.follicleDetector.save_model(follicleDetector)

        if follicleClassifier is not None:
            self.follicleClassifier.save_model(follicleClassifier)

    def load(self, follicleDetector, follicleClassifier):
        """save_params

        Save the trained parameters of the model.

        Parameters
        ----------
        follicleDetector: str, the path where to load the parameters of the boxPixel, if None, no loading if performed for follicleDetector
        follicleClassifier: str, the path where to load the parameters of the follicleClassifier, if None, no loading if performed for follicleClassifier
        """

        if follicleDetector is not None:
            self.follicleDetector.load_model(follicleDetector)
    
        if follicleClassifier is not None:
            self.follicleClassifier.load_model(follicleClassifier)

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

        # 1. Training the follicleDetector
        print("Fitting follicleDetector")

        # Skipping if already fitted
        if self.follicleDetector.fitted_ == False:
            ## Fitting the classifier
            self.follicleDetector.fit(X,y[["filename","xmin","xmax","ymin","ymax","label"]])

        # 2. Training image classifier
        print("Fitting follicleClassifier")

        # Skipping if already fitted
        if self.follicleClassifier.fitted_ == False:
        
            # Getting dataLoader
            ## The in_memory parameter means that all the cropped sample will be stored in ram instead of in disk
            follicle_detector_lambda_partial = partial(follicle_detector_lambda, boxDetector=self.follicleDetector)
            train_dataset = folliclesDataset(
                image_loader,
                label_ratio_threshold=0.2,
                data_augmentation=False,
                local_path=None,
                box_classifier=follicle_detector_lambda_partial,
                in_memory=True,
                verbose=False,
                order="box_ratio",
                force_reload=False,
                mode="crop"
            )

            dataloader_collapse_train = lambda x: dataloader_collapse(
                x, 
                image_size_width=self.follicle_classifier_size, 
                reducer="random_crop", 
                random_flip=True
            )

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=dataloader_collapse_train)

            for i in range(self.n_epochs):

                if self.verbose_train:
                    if i%3 == 0 and len(self.follicleClassifier.losses) > 0:
                        mean_loss = np.array(self.follicleClassifier.losses).mean()
                        print(f"Epoch n°{i} - Train loss : {mean_loss}")
                    self.follicleClassifier.losses = []

                for x, y in train_dataloader:
                    
                    # x is a tuple, we have to take it into account
                    x = tuple([z.to(self.device) for z in x])
                    y = y.to(self.device)

                    self.follicleClassifier.fit(x,y)
            
            self.follicleClassifier.fitted_ = True

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

        follicle_detector_lambda_partial = partial(follicle_detector_lambda, boxDetector=self.follicleDetector)
        predict_dataset = folliclesDataset(
            image_loader,
            label_ratio_threshold=0.2,
            data_augmentation=False,
            local_path=None,
            box_classifier=follicle_detector_lambda_partial,
            in_memory=True,
            verbose=False,
            order="default",
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
                x_sample = [z.to(self.device) for z in x_sample]
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