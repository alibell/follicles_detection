import numpy as np
import os
import pickle
import torch
from torch import nn
from torchvision.transforms import Resize, Pad, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, RandomRotation
from torch.utils.data import Dataset
from PIL import Image

class folliclesDataset(Dataset):
    """folliclesDataset
    
    This class provide a dataset for follicles algorithm training.
    The aim is to perform all the data transform and augmentation at the same place.
    This class provide an iterate, either it provide data in live, either it stores them in hard drive and provide them from memory 
    """

    def __init__ (self, 
        image_loader, 
        local_path, 
        in_memory=False,
        verbose=True, 
        force_reload=False,
        window_size=(400,400),
        border_condition="translate"
    ):
        """Parameters
            ----------
            image_loader: object from the image loader class
            local_path: str, local path for data storage which are kept in memory as pickle in the local_path folder
            in_memory: boolean, if True the data are store in memory, otherwise they are written
            verbose: boolean, informations about current operations are displayed
            force_reload: boolean, if the data have already been load for a given folder, they are not loaded again if force_reload is set at False
            window_size: tuple of two int, size of the windows moving on the picture
            border_condition: str, how to manage when the window is greater than the limit of the image, values : ignore : return a cropped version of the border, translate : perform a translation of the window on the image, padding : pad the missing values with 0
        """

        # Storing the image loader
        self.image_loader = image_loader

        # Storing the parameters
        if local_path is not None and os.path.exists(local_path):
            self.local_path = local_path
        else:
            self.local_path = None
            if in_memory == False:
                raise Exception("The provided path doesn't exist.")

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

            self._generate_all_data(window_size=window_size, border_condition=border_condition)

            if self.metadata_path is not None:
                self._write_metadata(self.metadata_path)

    def _generate_all_data(self, window_size=(400,400), border_condition="translate"):
        """Function that generate and write all the data

        Parameters
        ----------
        window_size: size of the windows moving on the picture
        border_condition: how to manage when the window is greater than the limit of the image, values : ignore : return a cropped version of the border, translate : perform a translation of the window on the image, padding : pad the missing values with 0

        Output
        ------
        No output. It writes all the data.
        """

        for filename in self.image_loader.X_filenames:
            output_data = self._generate_data(filename, window_size=window_size, border_condition=border_condition)

            if self.verbose == True:
                n_images = len(output_data)
                print(f"Generated {n_images} data from {filename}")

            ids = list(range(
                        len(self.metadata), 
                        len(self.metadata)+len(output_data)
            ))

            if self.local_path is not None:
                output_filenames = [
                    "/".join([
                        self.local_path,
                        str(x)
                    ])+".jpg" for x in ids]
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
                    Image.fromarray(data["data"].numpy()).save(filename)

                if self.in_memory:
                    self.memory[id] = data["data"].numpy()

    def _fix_window_border (self, offsets, image_border, border_condition):
        """_fix_window_border

        This function fix the border limit when generating windows of the image

        Parameters
        ----------
        offsets: tuple of min and max of the offset (xmin, xmax) or (ymin, ymax)
        image_shape: int of the border limit of the the image xmax or ymax 
        border_condition: str, how to manage when the window is greater than the limit of the image, values : ignore : return a cropped version of the border, translate : perform a translation of the window on the image, padding : pad the missing values with 0

        Output
        ------
        tuple of offset with fixed border
        """
        
        offset_min, offset_max = offsets

        if offset_max > image_border:
            if border_condition == "translate":
                delta = offset_max-image_border
                offset_min -= delta
                offset_max = image_border
        
        return offset_min, offset_max

    def _generate_data(self, filename, window_size=(400,400), border_condition="translate"):
        """Generate the data from a sample

        Parameters
        ----------
        filename: str, name of the file from which we generate the data
        window_size: size of the windows moving on the picture
        border_condition: how to manage when the window is greater than the limit of the image, values : ignore : return a cropped version of the border, translate : perform a translation of the window on the image, padding : pad the missing values with 0

        Output
        ------
        List of dict, containing :
            filename: name of the original file
            width: width of the image
            height: height of the image
            window_offsets: offset of the current window
            bbox: list of bbox with xmin, ymin, xmax, ymax contained in the image
            bbox_label: list of labels of the bbox contained in the image
            data: box content
        """

        # Getting original data and cropped data
        original_data = self.image_loader.get_sample(filename)
        original_image_shape = original_data[-1]
        if self.image_loader.has_y:
            original_image, original_boxes, original_labels = original_data[0], original_data[1], original_data[2]
            original_image = torch.tensor(original_image)
            original_boxes = [[x[0],x[2],x[1], x[3]] for x in original_boxes] # Converting original box to xmin, ymin, xmax, ymax
        else:
            original_image = original_data[0]

        # Generating the new windows on the image
        windows = []

        if self.image_loader.has_y:
            bbox_array = np.array(original_boxes)
            label_array = np.array(original_labels)

        # Number of box to produce
        i = original_image_shape[0]//window_size[0] + int(original_image_shape[0]%window_size[0] != 0)
        j = original_image_shape[1]//window_size[1] + int(original_image_shape[1]%window_size[1] != 0)

        for j_ in range(j):
            for i_ in range(i):
                # Getting the offset
                window_xmin_, window_ymin_ = j_*window_size[1], i_*window_size[0]
                window_xmax_, window_ymax_ = (window_xmin_+window_size[1]), (window_ymin_+window_size[0])

                window_xmin, window_xmax = self._fix_window_border(
                                                (window_xmin_, window_xmax_), 
                                                original_image_shape[1], 
                                                border_condition
                                            )
                window_ymin, window_ymax = self._fix_window_border(
                                                (window_ymin_, window_ymax_), 
                                                original_image_shape[0], 
                                                border_condition
                                            )
                
                window_offsets = (window_xmin, window_ymin, window_xmax, window_ymax)

                # Getting the image
                window_data = original_image[window_ymin:window_ymax, window_xmin:window_xmax]
                if border_condition == "pad":
                    delta_x = window_offsets[2]-original_image_shape[1]
                    delta_y = window_offsets[3]-original_image_shape[0]

                    window_data = Pad((delta_x, delta_y))(window_data)

                # Getting the bbox and labels
                if self.image_loader.has_y:
                    bbox_window = bbox_array.copy()
                    bbox_window -= np.repeat([[window_xmin, window_ymin]], 2, axis=0).flatten()
                    bbox_window = np.clip(bbox_window, a_min = 0, a_max=np.repeat(np.array([[window_size[1], window_size[0]]]),2, axis=0).flatten())
                    bbox_window_mask = (bbox_window[:,2]-bbox_window[:,0])*(bbox_window[:,3]-bbox_window[:,1]) != 0
                    
                    bbox_window = bbox_window[bbox_window_mask]
                    bbox_label = label_array[bbox_window_mask]

                windows.append({
                    "filename":filename,
                    "width": window_size[1],
                    "height": window_size[0],
                    "window_offsets": window_offsets, 
                    "data": window_data, 
                    "bbox": bbox_window, 
                    "bbox_label": bbox_label
                })

        return windows
        
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
            metadata = self.metadata[idx]

            # Loading the data
            if self.in_memory:
                data = self.memory[metadata["id"]]
            else:
                data = np.array(Image.open(metadata["path"])).astype(np.uint8)

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
                        data.append(np.array(Image.open(f)).astype(np.uint8))

            output = list(zip(data, metadata))

            return output

    def __len__(self):
        return len(self.metadata)