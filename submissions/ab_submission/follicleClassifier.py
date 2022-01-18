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

class follicleClassifier(nn.Module):

    def __init__ (self, device="cpu"):

        super(follicleClassifier, self).__init__()

        # Device detection for NN
        self.device = device

        # Creation of the neural network

        ## Creating pre-processing layer
        preprocessing_layer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        ## Loading VGG16 and freezing the parameters
        vgg_model = vgg16(pretrained=True)
        for param in vgg_model.parameters():
            param.requires_grad = False

        #vgg_features = vgg_model.features[0:12]
        vgg_features = vgg_model.features
        
        ## Full network
        self.network = nn.Sequential(*[
            preprocessing_layer,
            vgg_features,
            nn.Conv2d(512, 512, padding="same", kernel_size=(3,3), stride=(1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(
                in_features=512,
                out_features=125,
                bias=True
            ),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.BatchNorm1d(num_features=125),
            nn.Linear(
                in_features=125,
                out_features=25,
                bias=True
            ),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.BatchNorm1d(num_features=25),
            nn.Linear(
                in_features=25,
                out_features=5,
                bias=True
            ),
            nn.Softmax(dim=1)
        ])

        # Setting optimizer and loss
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.parameters(), lr=0.001)

        # Storing in device
        self.to(device)

        # Loss history
        self.losses = []
        
    def forward(self, X):
        """forward
        Required to perform the forward pass of the network.

        Parameters
        ----------
        X : torch tensor of size (n, 3, w, h) with n the number of samples, w the image width and h the image height

        Output
        ------
        Torch tensor of sice (n, 5) with n the number of samples and 5 the number of features
        """

        y_hat = self.network(X)

        return y_hat

    def forward_intermediate(self, X):
        """forward_intermediate
        Required to perform the forward pass of the intermediate layers of the network.

        Parameters
        ----------
        X : torch tensor of size (n, 3, w, h) with n the number of samples, w the image width and h the image height

        Output
        ------
        Torch tensor of sice (n, 125) with 125 the representation of the data
        """

        y_hat = self.network[0:9](X)

        return y_hat

    def fit(self, X, y, verbose=True):
        """fit
        Train the neural network

        Parameters
        ----------
        X : torch tensor of size (n, 3, w, h) with n the number of samples, w the image width and h the image height,
        y : torch tensor of size (n, 5) with n the number of samples and the number of features
        """

        # Training mode
        self.train()
        self.optimizer.zero_grad()

        y_hat = self.forward(X)
        loss = self.loss(y, y_hat)

        # Back propagation
        loss.backward()

        # Gradient descient step
        self.optimizer.step()

        # Keeping track of loss
        self.losses.append(loss.item())

    def predict(self, X):
        """predict
        Get the neural network prediction

        Parameters
        ----------
        X : torch tensor of size (n, 3, w, h) with n the number of samples, w the image width and h the image height,
        """

        self.eval()

        with torch.no_grad():
            y_hat = self.forward(X)
            
        return y_hat

    def predict_intermediate(self, X):
        """predict_intermediate
        Get the neural network prediction of intermediate layer of the network

        Parameters
        ----------
        X : torch tensor of size (n, 3, w, h) with n the number of samples, w the image width and h the image height,
        """

        self.eval()

        with torch.no_grad():
            y_hat = self.forward_intermediate(X)
            
        return y_hat

    def save_model(self, path):
        """Save a serialized version of the model

        Parameters
        ----------
        path: str, output path for the model
        """

        if "box_ratio_" in dir(self):
            box_ratio = self.box_ratio_
        else:
            box_ratio = None

        if "box_size_" in dir(self):
            box_size = self.box_size_
        else:
            box_size = None

        state = {
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'box_ratio': box_ratio,
            'box_size': box_size
        }

        torch.save(state, path)

        print(f"Model save in {path}")

    def load_model(self, path):
        """Load a serialized version of the model

        Parameters
        ----------
        path: str, output path for the model
        device: str, device in which to load the model
        """

        state = torch.load(path, map_location = self.device)
        self.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

        if state["box_ratio"] is not None:
            self.box_ratio_ = state["box_ratio"]

        if state["box_size"] is not None:
            self.box_size_ = state["box_size"]

        print(f"Model loaded from {path}")