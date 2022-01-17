import numpy as np
import pandas as pd
from sklearn.utils import check_array
from sklearn.tree import DecisionTreeClassifier
from scipy.ndimage import rotate, morphology
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

        augmented_image_matrix = []
        for x in rotations:
            for y in flip:
                if y in [0,1]:
                    flipped_image = np.flip(image_matrix, axis=y)
                if y == -1:
                    flipped_image = image_matrix
                
                flipped_rotated_image = rotate(flipped_image, angle=x, order=0)
                augmented_image_matrix.append(flipped_rotated_image)

        return augmented_image_matrix

    def get_crop(self, image_array, box_array, data_augmentation=False):
        """Get crop data given an image and box array

        Parameters
        ----------
        image_array: numpy array of dimension (h,w)
        box_array: list of coordonates, xmin xmax ymin ymax for each box
        data_augmentation: boolean, default False, if true the function return an augmented dataset.

        Output
        ------
        list of cropped images
        """

        cropped_images = []

        # Cropping the image
        for box in box_array:
            cropped_image = image_array[box[2]:box[3],box[0]:box[1]]

            # Dataset augmentation
            if data_augmentation == True:
                augmented_images = self.__generate_augmented_images(cropped_image)
                cropped_images += augmented_images

                # Generating augmented labels
                image_labels = np.repeat(image_labels, len(augmented_images))
            else:
                cropped_images.append(cropped_image)

        return cropped_images

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
            output_image = self.get_crop(image_data_preprocessed, image_box)
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

        ## Loading Inception3 and freezing the parameters
        vgg_model = vgg16(pretrained=True)
        for param in vgg_model.parameters():
            param.requires_grad = False

        #vgg_features = vgg_model.features[0:12]
        vgg_features = vgg_model.features
        
        ## Full network
        self.network = nn.Sequential(*[
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            preprocessing_layer,
            vgg_features,
            nn.Conv2d(512, 512, padding="same", kernel_size=(3,3), stride=(1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(
                in_features=512,
                out_features=125,
                bias=True
            ),
            nn.Dropout(0.3),
            nn.BatchNorm1d(num_features=125),
            nn.ReLU(),
            nn.Linear(
                in_features=125,
                out_features=25,
                bias=True
            ),
            nn.Dropout(0.3),
            nn.BatchNorm1d(num_features=25),
            nn.ReLU(),
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
        """fit
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
                y_hat_box.append([x_min, y_min, x_max, y_max])

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
                original_image_crop = image_loader.get_crop(original_data[0], original_data[1])

                ## Dataset from original data
                dataset_for_follicle_classifier_original = [original_image_crop, original_data[2]]

                ##  Dataset from boxes
                ### Here we compute a matrix of zeros with the location of original labelled data, we only keep box which intersect with theses
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
                dict(
                    ("bbox",pred[0]),("proba",pred[1]),("class",pred[2])
                ) for pred in zip(y_hat_box, y_hat_proba.numpy().tolist(), y_hat_labels)
            ])

        return predictions