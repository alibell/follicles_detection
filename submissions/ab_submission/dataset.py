import numpy as np
import os
import pickle
import torch
from torch import nn
from torchvision.transforms import Resize, Pad, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, RandomRotation
from torch.utils.data import Dataset

class folliclesDataset(Dataset):
    """folliclesDataset
    
    This class provide a dataset for follicles algorithm training.
    The aim is to perform all the data transform and augmentation at the same place.
    This class provide an iterate, either it provide data in live, either it stores them in hard drive and provide them from memory 
    """

    def __init__ (self, 
        image_loader, 
        data_augmentation, 
        local_path, 
        in_memory=False,
        box_classifier = None, 
        verbose=True, 
        force_reload=False,
        label_ratio_threshold=0.7,
        order="default",
        mode="all"
    ):
        """Parameters
            ----------
            image_loader: object from the image loader class
            data_augmentation: boolean, if True a data augmentation is performed
            local_path: str, local path for data storage which are kept in memory as pickle in the local_path folder
            in_memory: boolean, if True the data are store in memory, otherwise they are written
            box_classifier: object from the box classifier class, if None no box are generated from the classifier
            verbose: boolean, informations about current operations are displayed
            force_reload: boolean, if the data have already been load for a given folder, they are not loaded again if force_reload is set at False
            order: str, ('default','box_ratio') : order in which the data would be iterate with __get_item___, by default it is the order of the data, if box_ratio it is the ratio of each box
            label_ratio_threshold: float, threshold for keeping the label of a box
            mode: str, type of data to return, "crop", "original" or "all"
        """

        # Storing the image loader
        self.image_loader = image_loader
        self.box_classifier = box_classifier

        # Storing the parameters
        if local_path is not None and os.path.exists(local_path):
            self.local_path = local_path
        else:
            self.local_path = None
            if in_memory == False:
                raise Exception("The provided path doesn't exist.")

        self.data_augmentation = data_augmentation
        self.verbose = verbose
        self.in_memory = in_memory
        self.memory = {}

        # Recording metadata
        ## Contains the dataset metadata
        ## files metadata, files location
        if self.local_path is not None:
            self.metadata_path = "/".join([
                self.local_path,
                "metadata.pickle"
            ])
        else:
            self.metadata_path = None

        load_data = True
        self.metadata = []

        # Checking is metadata have been already loaded
        if self.metadata_path is not None and os.path.isfile(self.metadata_path):
            if force_reload == False:
                with open(self.metadata_path, "rb") as f:
                        self.metadata = pickle.load(f)
                load_data = False                    

        # Generating data
        if load_data:
            if self.verbose:
                print("Generating data")

            self._generate_all_data(label_ratio_threshold=label_ratio_threshold)

            if self.metadata_path is not None:
                self._write_metadata(self.metadata_path)

        self.set_order_and_mode(order=order, mode=mode)

    def set_order_and_mode(self, order, mode):
        """set_order

        Calling this function change temporary the order of the data

        Parameters
        ----------
        order: str, ('default','box_ratio') : order in which the data would be iterate with __get_item___, by default it is the order of the data, if box_ratio it is the ratio of each box
        mode: str, type of data to return, "crop", "original" or "all"
        """

        if order in ("default",'box_ratio'):
            metadata_range = list(range(len(self.metadata)))
            if order=="default":
                self.metadata_mask = dict(zip(
                    metadata_range,
                    metadata_range
                ))
            elif order=='box_ratio':
                ratio_list = [x["bbox_ratio"] for x in self.metadata]
                ratio_sorted_list = np.argsort(ratio_list).tolist()
                self.metadata_mask = dict(zip(
                    metadata_range,
                    ratio_sorted_list
                ))
        else:
            raise Exception("order should be default or box_ratio")

        if mode in ("crop","original"):
            allowed_status = [mode]
            metadata = self.metadata

            metadata_mask_values = [x for x in self.metadata_mask.values() if metadata[x]["status"] in allowed_status]
            self.metadata_mask = dict(zip(range(len(metadata_mask_values)), metadata_mask_values))
                

    def _generate_all_data(self, label_ratio_threshold=0.7):
        """Function that generate and write all the data

        Parameters
        ----------
        label_ratio_threshold: threshold of percentage of box intersection for keeping it

        Output
        ------
        No output. It writes all the data.
        """

        for filename in self.image_loader.X_filenames:
            output_data = self._generate_data(filename, label_ratio_threshold)
            ids = list(range(
                        len(self.metadata), 
                        len(self.metadata)+len(output_data)
            ))

            if self.local_path is not None:
                output_filenames = [
                    "/".join([
                        self.local_path,
                        str(x)
                    ])+".pickle" for x in ids]
            else:
                output_filenames = [
                    None for x in ids
                ]

            for data, filename, id in zip(output_data, output_filenames, ids):
                if self.verbose:
                    print(f"Writting {filename}")

                output_dict = dict([(key,value) for key, value in data.items() if key not in ["data"]])
                output_dict["path"] = filename
                output_dict["id"] = id

                # Keeping the data in the internal metadata list
                self.metadata.append(output_dict)

                # Writting file
                if filename is not None:
                    with open(filename,"wb") as f:
                        pickle.dump(data["data"], f)

                if self.in_memory:
                    self.memory[id] = data["data"]


    def _generate_data(self, filename, label_ratio_threshold=0.7):
        """Generate the data from a sample

        Parameters
        ----------
        filename: str, name of the file from which we generate the data
        label_ratio_threshold: threshold of percentage of box intersection for keeping it

        Output
        ------
        List of dict, containing :
            filename: name of the original file
            status: if data from original crop or not
            width: width of the image
            height: height of the image
            ratio: ratio h/w of the image
            bbox: xmin, ymin, xmax, ymax of the box
            bbox_width:  widht of the box
            bbox_height: height of the box
            bbox_ratio : ratio h/w of the box
            data: box content
            label: label of the box 
        """

        # Getting original data and cropped data
        original_data = self.image_loader.get_sample(filename)
        original_image_shape = original_data[-1]
        if self.image_loader.has_y:
            original_image, original_boxes, original_labels = original_data[0], original_data[1], original_data[2]
            original_image_crop, original_image_crop_labels = self.image_loader.get_crop(original_image, original_boxes, image_labels=original_labels, data_augmentation=self.data_augmentation)
            original_boxes = [[x[0],x[2],x[1], x[3]] for x in original_boxes] # Converting original box to xmin, ymin, xmax, ymax
        else:
            original_image = original_data[0]

        # Getting the box
        detected_box = self.box_classifier(image_loader = self.image_loader, image_name = filename)

        # Filter boxs and get labels
        if self.image_loader.has_y:
            new_box_coordonates, new_box_data, new_box_labels = self._filter_box(original_image=original_image, 
                                                                                original_boxes=original_boxes, 
                                                                                original_labels=original_labels,
                                                                                detected_box=detected_box,
                                                                                label_ratio_threshold=label_ratio_threshold
                                                            )
        else:
            new_box_data = [original_image[x[1]:x[3], x[0]:x[2]] for x in detected_box]
            new_box_coordonates, new_box_data, new_box_labels = detected_box, new_box_data, [None for x in range(len(detected_box))]

        # Generating output data

        if self.image_loader.has_y:
            output_data = [
                ['original', zip(original_boxes, original_image_crop, original_image_crop_labels)],
                ['crop', zip(new_box_coordonates, new_box_data, new_box_labels)]
            ]
        else:
            output_data = [
                ['crop', zip(new_box_coordonates, new_box_data, new_box_labels)]
            ]

        output_dict = [{
            "filename":filename,
            "status":data[0],
            "height":original_image_shape[0],
            "width":original_image_shape[1],
            "ratio":original_image_shape[0]/original_image_shape[1],
            "bbox_width":x[0][2]-x[0][0],
            "bbox_height":x[0][3]-x[0][1],
            "bbox_ratio":(x[0][3]-x[0][1])/(x[0][2]-x[0][0]),
            "bbox":x[0],
            "data":x[1],
            "label":x[2]
        } for data in output_data for x in data[1]]

        return output_dict
        

    def _filter_box(self, original_image, original_boxes, original_labels, detected_box, label_ratio_threshold=0.7):
        """Given a box list, return a filtered list and its labels

        Parameters
        ----------
        original_image: numpy array of size (h, w, 3) of the original image
        original_boxes: list of original box locations in format xmin, ymin, xmax, ymax
        original_labels: list integer corresponding of the labels of the original box
        detected_box: list of detected box in formay xmin, ymin, xmax, ymax
        label_ratio_threshold: threshold of percentage of box intersection for keeping it

        Output
        ------
        Tuple new_box_coordonates, new_box_data, new_box_label :
        - new_box_coordonates: list of xmin, ymin, xmax and ymax coordonates
        - new_box_data: numpy array of size (h,w) which contains the content of the box
        - new_box_label: int of the box class
        """

        # We create a reference matrix, which contains the true labels
        label_matrix = np.ones(original_image.shape[0:2])*-1
        for original_box, original_label in zip(original_boxes, original_labels):
            label_matrix[original_box[1]:original_box[3],original_box[0]:original_box[2]] = original_label

        new_box_coordonates = []
        new_box_data = []
        new_box_label = []

        for box in detected_box:
            # Create a temporary matrix for working on data
            working_matrix = label_matrix[box[1]:box[3],box[0]:box[2]].copy()
            if np.max(working_matrix) != -1:      
                # Compute the proportion of pixels with a label
                label_ratio = (working_matrix != -1).astype("int").mean()
                
                if label_ratio > label_ratio_threshold:
                    labels_list = np.bincount(working_matrix[working_matrix != -1].astype("int8")).flatten()
                    box_label = np.argmax(labels_list)
                    box_data = original_image[box[1]:box[3], box[0]:box[2]]

                    new_box_coordonates.append(box)
                    new_box_data.append(box_data)
                    new_box_label.append(box_label)

        return new_box_coordonates, new_box_data, new_box_label

    def _write_metadata(self, path):
        """Write the metadata in a pickle file

        Parameters
        ----------
        path: str, path where to write the metadata pickle file
        """

        if self.verbose:
            print(f"Writting metadata in {path}")

        with open(path, "wb") as f:
            pickle.dump(self.metadata, f)

    def __getitem__(self, idx):
        """For a given id, return a data

        Parameters
        ----------
        idx: int, id of the data to get or str to get list of data of a specific file
        """

        if isinstance(idx, int):

            # Getting the metadata
            metadata = self.metadata[self.metadata_mask[idx]]

            # Loading the data
            if self.in_memory:
                data = self.memory[metadata["id"]]
            else:
                with open(metadata["path"], "rb") as f:
                    data = pickle.load(f)

            return data, metadata

        if isinstance(idx, str):
            # Getting the metadata and data
            metadata = [x for x in self.metadata if x["filename"] == idx]
            if self.in_memory:
                data = [self.memory[x["id"]] for x in metadata]
            else:
                data = []
                for x in metadata:
                    with open(x["path"], "rb") as f:
                        data.append(pickle.load(f))

            output = list(zip(data, metadata))

            return output

    def __len__(self):
        return len(list(self.metadata_mask.keys()))

def dataloader_collapse (x, image_size_width=128, reducer="random_crop", random_flip=True):
    """dataloader_collate
    
    Function of collate for the DataLoader class use with the datasetFollicles class.
    This function is used to provide images from the datasetFollicles in a tensor for training task.

    Parameters
    ----------
    x: input data to collate, list containing tuple of a numpy array of size (h,w,3) and a dictionnary is expected
    image_size: int, width of the output image
    reducer: str, operation used to fit all the samples in the same tensor : "resize", "random_crop"
    random_flip: boolean, if True, random flip of the images are performed

    Output
    ------
    Tuple containing a tensor of size :
        (batch_size, *features) with features the size of the features data
        (batch_size, 5) with 5 the one-hot encoding of the label
    """

    Xs = []
    Xs_ = [] # Extra features
    ys = []
    min_ratio = np.array([data[1]["bbox_ratio"] for data in x]).mean()

    # Getting the images target size
    ## We try to respect the 128 but adapt to the image size
    if np.array([data[1]["bbox_width"] for data in x]).min() <= image_size_width:
        image_size_width = np.array([data[1]["bbox_width"] for data in x]).min()
    width = image_size_width
    height = int(min_ratio*image_size_width)

    # Getting the operators
    if reducer == "resize":
        reducer_operator = Resize((height, width))

    elif reducer == "random_crop":
        # Recomputing height
        reducer_operator = RandomResizedCrop(size=(height, width))

    if random_flip:
        random_hflip = RandomHorizontalFlip()
        random_vflip = RandomVerticalFlip()

    for data in x:
        # Getting data in tensor
        data_tensor = torch.tensor(data[0], dtype=torch.float32)
        data_tensor /= 255.
        data_tensor = torch.moveaxis(data_tensor, 2, 0)

        # Cropping or padding it
        data_tensor = reducer_operator(data_tensor)

        # Random transformations
        if random_flip:
            data_tensor = random_hflip(data_tensor)
            data_tensor = random_vflip(data_tensor)

        # extra features
        bbox_ratio = data[1]["bbox_ratio"]
        bbox_relative_size = (data[1]["bbox_width"]*data[1]["bbox_height"])/(data[1]["width"]*data[1]["height"])
        bbox_relative_x_position = data[1]["bbox"][0]/data[1]["width"]
        bbox_relative_y_position = data[1]["bbox"][1]/data[1]["height"]

        # Getting label
        if data[1]["label"] is not None:
            y = nn.functional.one_hot(
                torch.tensor(data[1]["label"].astype("int"), dtype=torch.int64), 
                num_classes=5
            ).float()

            ys.append(y)
        else:
            y = None
        
        Xs.append(data_tensor)
        Xs_.append(torch.tensor([bbox_ratio, bbox_relative_size, bbox_relative_x_position, bbox_relative_y_position], dtype=torch.float32))

    X = torch.stack(Xs)
    X_ = torch.stack(Xs_)
    if y is not None:
        y = torch.stack(ys)

    return (X, X_), y