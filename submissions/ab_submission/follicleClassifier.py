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
from torchvision.models import resnet18
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
        
        ## Loading resnet and freezing all but the last layer : the litterature identified resnet as an effective NN for histologic classification
        rn18 = resnet18(pretrained=True)
        rn18_layer = nn.Sequential(*list(rn18.children())[0:-1])
        for param in rn18_layer.parameters():
            param.requires_grad = False
        for param in rn18_layer[-2].parameters():
            param.requires_grad = True

        self.network_features1 = nn.Sequential(*[
            preprocessing_layer,
            rn18_layer,
            nn.Conv2d(in_channels=512, out_channels=100, padding="same", kernel_size=(3,3)),
            nn.Dropout(0.2),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(in_features=100, out_features=25),
            nn.Dropout(0.5),
            nn.LeakyReLU()
        ])

        self.network_features2 = nn.Sequential(*[
            nn.BatchNorm1d(29),
            nn.Linear(in_features=29, out_features=10),
            nn.LeakyReLU(),
            nn.Linear(in_features=10, out_features=5),
            nn.Softmax(dim=1)
        ])

        # Setting optimizer and loss
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.parameters(), lr=2e-4, betas=(0.9, 0.99))

        # Storing in device
        self.to(device)

        # Fitted state flag
        self.fitted_ = False

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

        x, x_ = X
        y_hat_ = self.network_features1(x)
        y_hat = self.network_features2(
            torch.concat([y_hat_, x_], axis=1)
        )

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

    def save_model(self, path):
        """Save a serialized version of the model

        Parameters
        ----------
        path: str, output path for the model
        """

        state = {
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'fitted_': self.fitted_
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
        self.fitted_ = state["fitted_"]

        print(f"Model loaded from {path}")