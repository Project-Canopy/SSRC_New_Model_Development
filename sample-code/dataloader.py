import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import tensorflow as tf


class DataLoader:
    def __init__(self,
                 training_dir="./",
                 label_file_path_train="labels_test_v1.csv",
                 label_file_path_val="labels_val.csv",
                 bucket_name='canopy-production-ml',
                 data_extension_type='.tif',
                 bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 augment=False,
                 enable_shuffle=False,
                 training_data_shuffle_buffer_size=1000,
                 training_data_batch_size=32,
                 enable_data_prefetch=False,
                 data_prefetch_size=tf.data.experimental.AUTOTUNE,
                 num_parallel_calls=tf.data.AUTOTUNE,
                 use_nbr=False,
                 output_shape=(tf.float32, tf.float32)):

        self.bucket_name = bucket_name
        self.use_nbr = use_nbr

        self.label_file_path_train = label_file_path_train
        self.label_file_path_val = label_file_path_val

        print(f"label_file_path_train: {self.label_file_path_train}")
        print(f"labels_file_val: {self.label_file_path_val}")
        self.labels_file_train = pd.read_csv(self.label_file_path_train)
        self.labels_file_val = pd.read_csv(self.label_file_path_val)
        self.training_filenames = self.labels_file_train.paths.to_list()

        self.labels_file_val = pd.read_csv(self.label_file_path_val)
        self.validation_filenames = self.labels_file_val.paths.to_list()

        self.bands = bands
        self.augment = augment
        self.local_path_train = training_dir

        self.enable_shuffle = enable_shuffle
        if training_data_shuffle_buffer_size is not None:
            self.training_data_shuffle_buffer_size = training_data_shuffle_buffer_size

        self.num_parallel_calls = num_parallel_calls
        self.enable_data_prefetch = enable_data_prefetch
        self.data_prefetch_size = data_prefetch_size
        self.data_extension_type = data_extension_type
        self.output_shape = output_shape

        self.training_data_batch_size = training_data_batch_size

        self.build_training_dataset()
        self.build_validation_dataset()

    def build_training_dataset(self):
        # Tensor of all the paths to the images
        # https://stackoverflow.com/questions/52582275/tf-data-with-multiple-inputs-outputs-in-keras
        self.training_dataset = tf.data.Dataset.from_tensor_slices(self.training_filenames)
        # self.training_dataset = tf.data.Dataset.from_tensor_slices({'input_RGB': self.training_filenames, 'input_x': self.training_filenames})

        # If data augmentation
        if self.augment is True:
            # https://stackoverflow.com/questions/61760235/data-augmentation-on-tf-dataset-dataset
            print("Data augmentation enabled")
            
            self.training_dataset = self.training_dataset.map((
                lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
                num_parallel_calls=self.num_parallel_calls).map(
            lambda image, label: (tf.image.random_flip_left_right(image), label)
            ).map(
            lambda image, label: (tf.image.random_flip_up_down(image), label)
            ).repeat(3)

            ### add blur augmentation
            
            self.length_training_dataset = len(self.training_filenames) * 3
            print(f"Training on {self.length_training_dataset} images")
        else:
            print("No data augmentation. Please set augment to True if you want to augment training dataset")
            self.training_dataset = self.training_dataset.map((
                lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
                num_parallel_calls=self.num_parallel_calls)
            self.length_training_dataset = len(self.training_filenames)
            print(f"Training on {self.length_training_dataset} images")

        # Randomly shuffles the elements of this dataset.
        # This dataset fills a buffer with `buffer_size` elements, then randomly
        # samples elements from this buffer, replacing the selected elements with new
        # elements. For perfect shuffling, a buffer size greater than or equal to the
        # full size of the dataset is required.
        if self.enable_shuffle is True:
            if self.training_data_shuffle_buffer_size is None:
                self.training_data_shuffle_buffer_size = len(self.length_training_dataset)
            self.training_dataset = self.training_dataset.shuffle(self.training_data_shuffle_buffer_size,
                                                                  reshuffle_each_iteration=True
                                                                  # controls whether the shuffle order should be different for each epoch
                                                                  )

        if self.training_data_batch_size is not None:
            # Combines consecutive elements of this dataset into batches
            self.training_dataset = self.training_dataset.batch(self.training_data_batch_size)

        # Most dataset input pipelines should end with a call to `prefetch`. This
        # allows later elements to be prepared while the current element is being
        # processed. This often improves latency and throughput, at the cost of
        # using additional memory to store prefetched elements.
        if self.enable_data_prefetch:
            self.training_dataset = self.training_dataset.prefetch(self.data_prefetch_size)

    def build_validation_dataset(self):
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(list(self.validation_filenames))
        self.validation_dataset = self.validation_dataset.map(
            (lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
            num_parallel_calls=self.num_parallel_calls)

        self.validation_dataset = self.validation_dataset.batch(self.training_data_batch_size)
        print(f"Validation on {len(self.validation_filenames)} images ")

#     def read_image(self, path_img):
#         path_to_img = self.local_path_train + "/" + path_img.numpy().decode()
#         train_img = np.transpose(rasterio.open(path_to_img).read(self.bands), (1, 2, 0))
#         # Normalize image
#         train_img = tf.image.convert_image_dtype(train_img, tf.float32)
#         return train_img
    
    def add_ndvi(self, img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        img = img.addBands(ndvi)
        img = img.float()
        return img
    
    def _rast_read_chip(self, path_img, bands):
        """
        Ensures that each raster read has the same shape (100x100 in this instance)
        """
        with rasterio.open(path_img) as src:
            chip = src.read(bands, window=Window(0, 0, 100, 100))
        
        return chip
    
    def read_image(self, path_img):
        path_to_img = self.local_path_train + "/" + path_img.numpy().decode()
        
        if 18 in self.bands: # if it has an NDVI band, remove the band before conducting normalization
            
            #create copy of bands list, remove ndvi band from copy 
            bands_copy = self.bands.copy()
            bands_copy.remove(18)
            train_img_no_ndvi = self._rast_read_chip(path_to_img, bands_copy)
            #normalize non_ndvi and ndvi bands separately, then combine as a single tensor (numpy) array
            train_img_no_ndvi = tf.image.convert_image_dtype(train_img_no_ndvi, tf.float32)
            ndvi_band = rasterio.open(path_to_img).read(18)
            train_img_ndvi = tf.image.convert_image_dtype(ndvi_band, tf.float32)
            train_img = tf.concat([train_img_no_ndvi,[train_img_ndvi]],axis=0)
            train_img = tf.transpose(train_img,perm=[1, 2, 0])
            
        elif 13 in self.bands: # if it has an NBR band, remove the band before conducting normalization
            
            #create copy of bands list, remove ndvi band from copy 
            bands_copy = self.bands.copy()
            bands_copy.remove(13)
            train_img_no_nbr = self._rast_read_chip(path_to_img, bands_copy)
            #normalize non_ndvi and ndvi bands separately, then combine as a single tensor (numpy) array
            train_img_no_nbr = tf.image.convert_image_dtype(train_img_no_nbr, tf.float32)
            nbr_band = self._rast_read_chip(path_to_img, 13)
            train_img_nbr = tf.image.convert_image_dtype(nbr_band, tf.float32)
            train_img = tf.concat([train_img_no_nbr,[train_img_nbr]],axis=0)
            train_img = tf.transpose(train_img,perm=[1, 2, 0])
        
        else:

            train_img = self._rast_read_chip(path_to_img, self.bands)
            
            if self.use_nbr:
                nbr_band = es.normalized_diff(train_img[9], train_img[12])
                train_img_nbr = tf.image.convert_image_dtype(nbr_band, tf.float32)
                train_img = tf.concat([train_img,[train_img_nbr]],axis=0)
            
            train_img = np.transpose(train_img, (1, 2, 0))
            # Normalize image
            train_img = tf.image.convert_image_dtype(train_img, tf.float32)
            

            
        return train_img 
    

    def get_label_from_csv(self, path_img):
        # testing if path in the training csv file or in the val one
        if path_img.numpy().decode() in self.labels_file_train.paths.to_list():
            ### Training csv
            # path_img is a tf.string and needs to be converted into a string using .numpy().decode()
            id = int(self.labels_file_train[self.labels_file_train.paths == path_img.numpy().decode()].index.values[0])
            # The list of labels (e.g [0,1,0,0,0,0,0,0,0,0] is grabbed from the csv file on the row where the s3 path is
            label = self.labels_file_train.drop('paths', axis=1).iloc[int(id)].to_list()
        else:
            ### Validation csv
            # path_img is a tf.string and needs to be converted into a string using .numpy().decode()
            id = int(self.labels_file_val[self.labels_file_val.paths == path_img.numpy().decode()].index.values[0])
            # The list of labels (e.g [0,1,0,0,0,0,0,0,0,0] is grabbed from the csv file on the row where the s3 path is
            label = self.labels_file_val.drop('paths', axis=1).iloc[int(id)].to_list()
        return label

    # Function used in the map() and returns the image and label corresponding to the file_path input
    def process_path(self, file_path):
        label = self.get_label_from_csv(file_path)
        #path_prefix = "s3://canopy-production-ml/chips/model2_s2cloudless/training_v1/"
        img = self.read_image(file_path)
        return img, label