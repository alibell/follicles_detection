from PIL import Image
from PIL import ImageDraw
from sklearn.utils import check_array
import cv2
import pickle
import numpy as np

class follicleDetector():
    """Identification of follicles candidate according to an image.
    """

    def __init__(self, 
                 threshold=0.7, 
                 scale_factor=10, 
                 erosion_kernel_size=(3,3), 
                 gaussian_blur_kernel_size=(25,25), 
                 box_padding=0.5,
                 box_ratio_min_tolerance=1,
                 box_ratio_max_tolerance=1,
                 box_relative_size_min_tolerance=1,
                 box_relative_size_max_tolerance=1
        ):
        """follicleDetector initialisation

        Parameters
        ----------
        threshold: float, threshold for image binarisation
        scale_factor: int, factor of image resizing, needed to reduce computational time
        erosion_kernel_size: tuple of int, size of the erosion kernel
        gaussian_blur_kernel_size: tuple of odd int, size of the gaussian blur kernel
        box_padding: float, % of the box width/height of box padding
        box_ratio_min_tolerance: float, percentage of variation tolerance of lower bound in the box ratio during the box filtering phase
        box_ratio_max_tolerance: float, percentage of variation tolerance of the upper bound in the box ratio during the box filtering phase
        box_relative_size_min_tolerance: float, percentage of variation tolerance of lower bound in the box_relative_size during the box filtering phase
        box_relative_size_max_tolerance: float, percentage of variation tolerance of the upper bound in the box_relative_size during the box filtering phase
        """

        self.params = {
            "threshold": float(threshold),
            "scale_factor": int(scale_factor),
            "erosion_kernel_size": erosion_kernel_size,
            "gaussian_blur_kernel_size": gaussian_blur_kernel_size,
            "box_padding": float(box_padding),
            "box_ratio_min_tolerance": box_ratio_min_tolerance,
            "box_ratio_max_tolerance": box_ratio_max_tolerance,
            "box_relative_size_min_tolerance": box_relative_size_min_tolerance,
            "box_relative_size_max_tolerance": box_relative_size_max_tolerance
        }

        self.coefs_ = {}

        # Set the fitted flag
        self.fitted_ = False

    def load_image(self, image_path, resize=True, standardize=True):
        """load the image givent its path

        Here we use opencv which relies on the libjpeg-turbo lib which seems to be faster

        Parameters
        ----------
        image_path: str, path of the image
        resize: boolean, if true the image is resized according to the scale factor
        standardize: boolean, if true the image is standardize between 0 and 1
        get_native: boolean, in case of resizing, if this parameter is 

        Output
        ------
        Tuple :
            - Numpy array of size (h,w,3) uint8 or float32 in BGR color format after processing (resize, normalization ...)
            - Numpy array of size (H,W,3) corresponding to the native image file
        """

        image = cv2.imread(image_path)

        if resize==True:
            nw = int(image.shape[1]/self.params["scale_factor"])
            nh = int(image.shape[0]/self.params["scale_factor"])
            
            image_processed = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_AREA)
        else:
            image_processed = image

        if standardize == True:
            image_processed = image_processed/255.

        return image_processed, image

    def _get_image_size(self, image_path):
        """Get the image size according to its path

        Lazy evaluation (thanks to PIL) of the image size without loading it in memory

        Parameters
        ----------
        image_path: str, path of the image

        Output
        ------
        (h, w) tuple of int containing image width and height
        """

        # PIL for lazy image evaluation, 10 times faster than a full loading
        image_file = Image.open(image_path)
        w,h = image_file.size

        return h, w

    def fit(self, X, y):
        """fit

        Training of the algorithm

        Parameters:
        -----------
        X : np.array([str]) or [str], image paths
        y : np.array of size (n, 6), containing : filenames, xmin, xmax, ymin, ymax and labels. Can be None. 
        """

        # Parsing data
        X = check_array(X, ensure_2d=False, dtype=str)
        y = check_array(y, dtype=None)

        # Get train dataset properties : height, weight and mask of positive values
        image_h_w = np.array([self._get_image_size(x) for x in X])
        image_size = image_h_w[:,0]*image_h_w[:,1]
        _, image_number_of_box = np.unique(y[:,0], return_counts=True)
        
        box_w = y[:,2]-y[:,1]
        box_h = y[:,4]-y[:,3]
        box_ratio = (box_h/box_w)
        box_relative_size = (box_h*box_w)/np.repeat(image_size, image_number_of_box)
        positive_mask = (y[:,5] != 0)
        
        # Computing coefs
        self.coefs_ = {
            'box_relative_size_min': np.quantile(box_relative_size[positive_mask], 0.01),
            'box_relative_size_max': np.quantile(box_relative_size[positive_mask], 0.99),
            'box_ratio_min': np.quantile(box_ratio[positive_mask], 0.01),
            'box_ratio_max': np.quantile(box_ratio[positive_mask], 0.99)
        }

        # Set the fitted flag
        self.fitted_ = True

    def save_model(self, model_path):
        """Function for model consistency
        Export the model parameters to a pickle file

        Parameters
        ----------
        model_path: str, path where to save the model
        """

        model_params = {
            "coefs_":self.coefs_,
            "fitted_":self.fitted_
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_params, f)

    def load_model(self, model_path):
        """Function for model consistency
        Load the model parameters from a pickle file

        Parameters
        ----------
        model_path: str, path where the model parameters are saved
        """

        with open(model_path, "rb") as f:
            model_params = pickle.load(f)

        for key, value in model_params.items():
            self.__setattr__(key, value)

    def plot_image_processing(self, image_path, fig_size=(20,10)):
        """Plot the image processing
        
        This function is used for debugging purpose, it plot the differents steps of the image processing.

        Parameters
        ----------
        image_path: str, path of an image for which we plot the processing
        fig_size: tuple of int, size of the output plot
        """

        # Getting image history
        box_list, image_history, _ = self.get_box_candidates(image_path)

        # Filter and adding filter to history
        box_list_positive, _ = self.filter_box_candidates(image_path, box_list)
        image_box_filter = image_history[list(image_history.keys())[-1]].copy()
        image_box_filter_PIL = Image.fromarray(image_box_filter)
        image_box_filter_PIL_draw = ImageDraw.Draw(image_box_filter_PIL)

        for box in box_list_positive:
            box_scaled = [int(x/self.params["scale_factor"]) for x in box]
            x1, y1, x2, y2 = box_scaled
            image_box_filter_PIL_draw.rounded_rectangle(
                ((x1, y1), (x2,y2))
            , fill=None, outline="black", width=5)
        image_box_filter = np.array(image_box_filter_PIL)

        image_history["99_image_box_filter"] = image_box_filter

        n_images = len(list(image_history.keys()))
        n_col = 3
        n_row = (n_images // n_col) + int((n_images % n_col) != 0)

        images = list(image_history.values())
        titles = list(image_history.keys())
        fig = plt.figure(figsize=fig_size)

        for i in range(len(images)):
            fig.add_subplot(n_row,n_col,i+1)
            plt.title(titles[i])
            plt.imshow(images[i], cmap="gray")


    def get_box_candidates(self, image_path):
        """get_box_candidates

        Return the box candidates

        Parameters
        ----------
        image_path: str, path of the image

        Output
        ------
        tuple :
            - List of boxs locations in xmin, ymin, xmax, ymax format
            - List of images during the preprocessing, can be useful for debugging purpose
            - Native image before any processing
        """

        image_history = {}

        # Getting the image
        image_history["0_native_image"], raw_image = self.load_image(image_path, resize=True, standardize=True)

        # Binarisation of the image
        image_history["1_binary_image"] = 1-(image_history["0_native_image"].mean(axis=2) <= self.params["threshold"]).astype(np.uint8)

        # Erosion of the image
        image_history["2_binary_image_erode"] = 1-cv2.erode(image_history["1_binary_image"], np.ones(self.params["erosion_kernel_size"]))

        # Appliance of gaussian blur
        image_history["3_binary_image_blurred"] = cv2.GaussianBlur(image_history["2_binary_image_erode"], self.params["gaussian_blur_kernel_size"], 0)

        # Erosion / dilatation
        image_history["4_binary_image_blurred_erode_dilate"] = cv2.erode(
            cv2.dilate(
                image_history["3_binary_image_blurred"], np.ones(self.params["erosion_kernel_size"])
            , iterations=2)
        , np.ones(self.params["erosion_kernel_size"]), iterations=2)

        # Getting the box
        final_image = image_history["4_binary_image_blurred_erode_dilate"].astype(np.uint8)
        image_small_shape = image_history["0_native_image"].shape
        image_shape = self._get_image_size(image_path=image_path)

        x_ratio = image_shape[1]/image_small_shape[1]
        y_ratio = image_shape[0]/image_small_shape[0]

        contours, _ = cv2.findContours(final_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        box_list = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            w_padding = w*self.params["box_padding"]
            h_padding = h*self.params["box_padding"]

            # Getting x_min, y_min, x_max, y_max
            x_min, y_min, x_max, y_max = x-w_padding, y-h_padding, x+w+w_padding, y+h+h_padding
            y_min, x_min = tuple([int(i) if i > 0 else 0 for i in [y_min, x_min]])
            y_max, x_max = tuple([int(i) if i < j else j for i,j in zip([y_max, x_max], image_small_shape)])

            # Finally, getting the coordonates for real image
            x_min, x_max = tuple([int(x_ratio*i) for i in [x_min, x_max]])
            y_min, y_max = tuple([int(y_ratio*i) for i in [y_min, y_max]])

            box_list.append([x_min, y_min, x_max, y_max])

        return box_list, image_history, raw_image

    def filter_box_candidates(self, image_path, box_list):
        """Given a box list and an image path, this function return a restricted amount of box

        Parameters
        ----------
        image_path: str, path of the image
        box_list: list, containing list of coordonates in format xmin, ymin, xmax, ymax

        Output
        ------
        Tuple :
            - List of correct boxs locations in xmin, ymin, xmax, ymax format
            - List of filtered box locations in xmin, ymin, xmax, ymax format
        """

        if self.fitted_ == True:
            # Getting image parameters
            image_shape = self._get_image_size(image_path=image_path)
            
            # Simple rule : we filter box for which the box ratio and box relative size is unexpected
            box_list_np = np.array(box_list)
            box_w = box_list_np[:,2]-box_list_np[:,0]
            box_h = box_list_np[:,3]-box_list_np[:,1]

            box_ratio = (box_h/box_w)
            box_relative_size = (box_h*box_w)/(image_shape[0]*image_shape[1])

            relative_size_min_limit = self.coefs_["box_relative_size_min"]*self.params["box_relative_size_min_tolerance"]
            relative_size_max_limit = self.coefs_["box_relative_size_max"]*self.params["box_relative_size_max_tolerance"]
            ratio_min_limit = self.coefs_["box_ratio_min"]*self.params["box_ratio_min_tolerance"]
            ratio_max_limit = self.coefs_["box_ratio_max"]*self.params["box_ratio_max_tolerance"]

            # Filter candidates
            box_filter_mask = (box_ratio > ratio_min_limit) & \
                            (box_ratio < ratio_max_limit) & \
                            (box_relative_size > relative_size_min_limit) & \
                            (box_relative_size < relative_size_max_limit)

            box_list_positive = box_list_np[box_filter_mask,:].tolist()
            box_list_filtered = box_list_np[(box_filter_mask==False),:].tolist()
        else:
            box_list_positive = box_list
            box_list_filtered = []

        return box_list_positive, box_list_filtered

    def predict(self, X, get_images=False):
        """ Get a list of box candidate given of images path
        
        Parameters
        ----------
        X: list of image paths
        get_images: boolean, if True, the loaded image is return with the label, this can be useful to reduce a pipeline IO

        Output
        ------
        Tuple :
            List of list of coordinates in format : (xmin, ymin, xmax, ymax)
            List of the native images (empty if get_images set to false)
        """

        # Parsing data
        X = check_array(X, ensure_2d=False, dtype=str)

        output_box_list = []
        native_images = []

        for x in X:
            # Getting the image box
            box_list, _, native_image = self.get_box_candidates(x)

            # Getting the filtered image box
            box_list_correct, _ = self.filter_box_candidates(x, box_list)

            if get_images == True:
                native_images.append(native_image)

            output_box_list.append(box_list_correct)

        return output_box_list, native_images

# Lambda function for follicle detector
## Here for code compatibility purpose
def follicle_detector_lambda (boxDetector, image_name, image_loader):
    # Get image path
    i = image_loader.X_filenames.index(image_name)
    image_path = [image_loader.X[i]]

    prediction = boxDetector.predict(image_path)[0][0]

    return prediction