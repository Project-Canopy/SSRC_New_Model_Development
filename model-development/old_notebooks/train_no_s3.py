import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy

# tf.keras.losses.CategoricalCrossentropy

import numpy as np
import argparse, os, subprocess, sys
import tensorflow.keras as keras
import pandas as pd
import io
import json
import random
import h5py
import multiprocessing
import time
import pickle

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Script mode doesn't support requirements.txt
# Here's the workaround:
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


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
                 training_data_batch_size=20,
                 enable_data_prefetch=False,
                 data_prefetch_size=tf.data.experimental.AUTOTUNE,
                 num_parallel_calls=tf.data.AUTOTUNE,
                 output_shape=(tf.float32, tf.float32)):

        self.bucket_name = bucket_name

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
    
    def read_image(self, path_img):
        path_to_img = self.local_path_train + "/" + path_img.numpy().decode()
        
        if 18 in self.bands:
            
            #create copy of bands list, remove ndvi band from copy 
            bands_copy = self.bands.copy()
            bands_copy.remove(18)
            train_img_no_ndvi = rasterio.open(path_to_img).read(bands_copy)
            #normalize non_ndvi and ndvi bands separately, then combine as a single tensor (numpy) array
            train_img_no_ndvi = tf.image.convert_image_dtype(train_img_no_ndvi, tf.float32)
            ndvi_band = rasterio.open(path_to_img).read(18)
            train_img_ndvi = tf.image.convert_image_dtype(ndvi_band, tf.float32)
            train_img = tf.concat([train_img_no_ndvi,[train_img_ndvi]],axis=0)
            train_img = tf.transpose(train_img,perm=[1, 2, 0])
        
        else:
            
            train_img = np.transpose(rasterio.open(path_to_img).read(self.bands), (1, 2, 0))
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
            label = self.labels_file_train.drop('paths', 1).iloc[int(id)].to_list()
        else:
            ### Validation csv
            # path_img is a tf.string and needs to be converted into a string using .numpy().decode()
            id = int(self.labels_file_val[self.labels_file_val.paths == path_img.numpy().decode()].index.values[0])
            # The list of labels (e.g [0,1,0,0,0,0,0,0,0,0] is grabbed from the csv file on the row where the s3 path is
            label = self.labels_file_val.drop('paths', 1).iloc[int(id)].to_list()
        return label

    # Function used in the map() and returns the image and label corresponding to the file_path input
    def process_path(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        return img, label


class SaveCheckpoints(keras.callbacks.Callback):
    def __init__(self, 
                base_name_checkpoint=None,
                lcl_chkpt_dir=None,
                s3_chkpt_dir=None):

        self.base_name_checkpoint = base_name_checkpoint
        self.lcl_chkpt_dir = lcl_chkpt_dir 
        self.s3_chkpt_dir = s3_chkpt_dir

    def on_epoch_end(self, epoch, logs={}):
        epoch = epoch + 1 
        print(f'\nEpoch {epoch} saving checkpoint')
        model_name = f'{self.base_name_checkpoint}_epoch_{epoch}.h5'
        local_path =  self.lcl_chkpt_dir + "/" + model_name
        s3_path = self.s3_chkpt_dir + "/" + model_name
        self.model.save_weights(local_path, save_format='h5')
        s3 = boto3.resource('s3')
        BUCKET = "canopy-production-ml-output"
        s3.Bucket(BUCKET).upload_file(local_path, s3_path)
        last_chkpt_filename = "last_chkpt.h5"
        last_chkpt_path = self.lcl_chkpt_dir + "/" + last_chkpt_filename
        self.model.save_weights(last_chkpt_path, save_format='h5')
        s3_path = self.s3_chkpt_dir + "/" + last_chkpt_filename
        s3.Bucket(BUCKET).upload_file(last_chkpt_path, s3_path)
        
class LRFinder(keras.callbacks.Callback):
    """Callback that exponentially adjusts the learning rate after each training batch between start_lr and
    end_lr for a maximum number of batches: max_step. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the plot method.
    """

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 100, smoothing=0.9,                 lcl_chkpt_dir=None,
                s3_chkpt_dir=None):
        super(LRFinder, self).__init__()
        self.lcl_chkpt_dir = lcl_chkpt_dir 
        self.s3_chkpt_dir = s3_chkpt_dir
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        print('Current batch learning rate:',tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print('New learning rate:', self.lr)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        print('Batch end')
        logs = logs or {}
        loss = logs.get('loss')
        print('Loss:', loss)
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)
            
            print('if/then statement 1')

            if step == 0 or loss < self.best_loss:
                print('New best loss found!')
                self.best_loss = loss
                
            print('if/then statement 2')

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True
                
        print('if/then statement 3')

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses)
        
        
    def on_epoch_end(self, epoch, logs={}):
        epoch = epoch + 1 
        print(f'\nEpoch {epoch} saving lr_finder_object')
        loss = self.losses
        lr = self.lrs
        df = pd.DataFrame(data={"loss":loss,"lr":lr})
        filename = "lr_finder.csv"
        local_path =  self.lcl_chkpt_dir + "/" + filename
        s3_path = self.s3_chkpt_dir + "/" + filename
        df.to_csv(local_path,index=False)
        s3 = boto3.resource('s3')
        BUCKET = "canopy-production-ml-output"
        s3.Bucket(BUCKET).upload_file(local_path, s3_path)


if __name__ == '__main__':
    print("TensorFlow version", tf.__version__)
    print("Keras version", keras.__version__)
    # Keras-metrics brings additional metrics: precision, recall, f1
    install('keras-metrics')
    import keras_metrics

    install('tensorflow-addons')
    from tensorflow_addons.metrics import F1Score, HammingLoss
    from tensorflow_addons.losses import SigmoidFocalCrossEntropy
    from tensorflow_addons.optimizers import CyclicalLearningRate

    install('wandb')
    import wandb
    from wandb.keras import WandbCallback

    install('rasterio')
    import rasterio
    from rasterio.session import AWSSession

    install('boto3')
    import boto3

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default="resnet")
    parser.add_argument('--callback', type=str, default="lrplateau")
    parser.add_argument('--early_stop', type=str, default="True")
    parser.add_argument('--clr_initial', type=float, default=.00001)
    parser.add_argument('--clr_max', type=float, default=.0005)
    parser.add_argument('--clr_step', type=int, default=2500)
    parser.add_argument('--lr_reduce_min', type=float, default=.00001)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--augment', type=str, default="False")
    parser.add_argument('--flip_left_right', type=str, default="False")
    parser.add_argument('--flip_up_down', type=str, default="False")
    parser.add_argument('--rot90', type=str, default="False")
    parser.add_argument('--numclasses', type=float, default=10)
    parser.add_argument('--bands', required=True)
    parser.add_argument('--bucket', type=str, default="margaux-bucket-us-east-1")
    parser.add_argument('--training_file', type=str, default="labels_test_v1.csv")
    parser.add_argument('--validation_file', type=str, default="val_labels.csv")
    parser.add_argument('--wandb_key', type=str, default=None)
    # data directories
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    parser.add_argument('--s3_chkpt_dir', type=str)
    parser.add_argument('--output', type=str, default="/opt/ml/output")
    parser.add_argument('--job_name', 
                        type=str, default=json.loads(os.environ.get('SM_TRAINING_ENV'))["job_name"])
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--starting_checkpoint', type=str, default=None)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    parser.add_argument('--output_type', type=str, default='multiclass')

    args, _ = parser.parse_known_args()

    os.environ["WANDB_API_KEY"] = args.wandb_key   # "6607ed7a49b452c2f3494ce60f9514f6c9e3b4e6"
    wandb.init(project='project-canopy')  # , sync_tensorboard=True
#     wandb.init(project='project-canopy')  # , sync_tensorboard=True
    config = wandb.config
    
    job_name = args.job_name
    data_dir = args.data
    lcl_chkpt_dir = "/opt/ml/model/checkpoints"
    
    s3_chkpt_base_dir = args.s3_chkpt_dir
    starting_checkpoint = args.starting_checkpoint
    print('Starting checkpoint:', starting_checkpoint)
    
    if not os.path.exists(lcl_chkpt_dir):
        os.mkdir(lcl_chkpt_dir)

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    bucket = args.bucket
    training_file = args.training_file
    validation_file = args.validation_file

    # Had to use strings because of the way sagemaker script was passing the args values 
    if args.augment.lower() == "false":
        augmentation_data = False
    else:
        augmentation_data = True
        
    if args.early_stop.lower() == "false":
        early_stop = False
    else:
        early_stop = True
        
        
    lr_callback = args.callback.lower() 

    print(f"lr: {lr}, batch_size: {batch_size}, augmentation_data: {augmentation_data}")
    
    bands = args.bands.split(" ")
    bands = list(map(int, bands))   # convert to int
    print(f"bands {bands}")
    input_shape = (100, 100, int(len(bands)))
    print(f"Input shape: {input_shape}")

    numclasses = int(args.numclasses)

    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    
    print(os.system(f"ls {training_dir}"))
    
    pre_trained_model = args.model

    output_type = args.output_type

    #default s3_checkpoint definition
    s3_chkpt_dir = s3_chkpt_base_dir + "/" + job_name
    
    #testing_s3_spot_restart_functionality
#     s3_chkpt_dir = "ckpt/pc-tf-custom-container-2021-04-20-05-19-50-592"

    def define_model(numclasses, xbands_input_shape, output_type, starting_checkpoint=None, lcl_chkpt_dir=None,s3_chkpt_dir=s3_chkpt_dir):
        
        print(f"Using Pre-trained {pre_trained_model} model")
        
        # parameters for CNN
        input_tensor = Input(shape=xbands_input_shape, name='input_x')

        # introduce a additional layer to get from bands to 3 input channels
        input_tensor = Conv2D(3, (1, 1))(input_tensor)
        
        if pre_trained_model == "resnet50":

            base_model_pre_trained = keras.applications.ResNet50(include_top=False,
                                                              weights='imagenet',
                                                              input_shape=(100, 100, 3))
            base_model = keras.applications.ResNet50(include_top=False,
                                                     weights=None,
                                                     input_tensor=input_tensor)

            # RGB_base_model = keras.applications.ResNet50(include_top=False,
            #                                                   weights='imagenet',
            #                                                   input_shape=(100, 100, 3))
            
            # xbands_base_model = keras.applications.ResNet50(include_top=False,
            #                                          weights=None,
            #                                          input_tensor=input_tensor)  #Ideally, this would be a resnet50 trained on the extra bands

            # RGB_base_model.layers[0]._name = 'input_RGB'

            # for layer in xbands_base_model.layers :
            #     layer._name = layer.name + str('_x')

            # premerge_RGB_model =  Model(inputs=RGB_base_model.input, outputs=RGB_base_model.get_layer('conv2_block3_out').output)
            # premerge_xbands_model = Model(inputs=xbands_base_model.input, outputs=xbands_base_model.get_layer('conv2_block3_out_x').output)
            # merged_features = Add()([premerge_RGB_model.output, premerge_xbands_model.output])
            # premerge_model = Model(inputs= [premerge_RGB_model.input, premerge_xbands_model.input],outputs = merged_features)
            # postmerge_model = Model(inputs = xbands_base_model.get_layer('conv3_block1_1_conv_x').input, outputs= xbands_base_model.output)
            # full_output = postmerge_model(premerge_model.output)
            # full_model = Model(inputs= [premerge_RGB_model.input, premerge_xbands_model.input], outputs= full_output)
                                                     
        if pre_trained_model == "efficientnetb0":
            
            base_model_pre_trained = keras.applications.EfficientNetB0(include_top=False,
                                                          weights="imagenet",
                                                          input_shape=(100, 100, 3))
            
            base_model = keras.applications.EfficientNetB0(include_top=False,
                                         weights=None,
                                         input_tensor=input_tensor)
            
            

        for i, layer in enumerate(base_model_pre_trained.layers):
            # we must skip input layer, which has no weights
            if i == 0:
                continue
            base_model.layers[i + 1].set_weights(layer.get_weights())

        # add a global spatial average pooling layer
        top_model = base_model.output
        # top_model = full_model.output
        top_model = GlobalAveragePooling2D()(top_model)

        # let's add a fully-connected layer
        top_model = Dense(2048, activation='relu')(top_model)
        top_model = Dense(2048, activation='relu')(top_model)
        # and a logistic layer
        if output_type == 'multilabel':
            predictions = Dense(numclasses, activation='softmax')(top_model)
        elif output_type == 'multiclass':
            predictions = Dense(numclasses + 1, activation='softmax')(top_model)
        else:
            raise ValueError(f'output_type is not multilabel or multiclass but {output_type}')

        # this is the model we will train
        # model = Model(inputs=base_model.input, outputs=predictions)
        model = Model(inputs=full_model.input, outputs=predictions)
#         model.summary()
#         last_chkpt_path = lcl_chkpt_dir + 'last_chkpt.h5'
        
        s3 = boto3.resource('s3')
        bucket = "canopy-production-ml-output"
        my_bucket = s3.Bucket(bucket)
        key = s3_chkpt_dir
        h5_files = [obj.key for obj in my_bucket.objects.filter(Prefix=key) if "h5" == obj.key[-2:]]
        
        h5_files_dict = {}
        
        for file in h5_files:
            try:
                match_key = int(file.split('_')[-1].split('.')[0])
                h5_files_dict[match_key] = file
            except:
                continue
    
        if h5_files_dict:

            max_epoch = max(h5_files_dict.keys())
            last_chkpt_s3_path = h5_files_dict[max_epoch]
            
            print('Spot instance restarting; loading previous checkpoint from',last_chkpt_s3_path)
            
            last_chkpt_filename = last_chkpt_s3_path.split("/")[-1]
            last_chkpt_local_path = lcl_chkpt_dir + '/' + last_chkpt_filename
            my_bucket.download_file(last_chkpt_s3_path, last_chkpt_local_path)
            model.load_weights(last_chkpt_local_path)
            start_epoch = max_epoch
            
        elif starting_checkpoint:
            start_epoch = 0 
            print('No previous checkpoint found in opt/ml/checkpoints directory; loading checkpoint from', starting_checkpoint)
            chkpt_name = lcl_chkpt_dir + '/' + 'start_chkpt.h5'
            my_bucket.download_file(starting_checkpoint, chkpt_name)
            model.load_weights(chkpt_name)
        else:
            start_epoch = 0 
            print('No previous checkpoint found in opt/ml/checkpoints directory; start training from scratch')
    
        return model,start_epoch
    
    
    base_name_checkpoint = "model_resnet"
    save_checkpoint_s3 = SaveCheckpoints(base_name_checkpoint, lcl_chkpt_dir, s3_chkpt_dir)
    save_lr_finder_s3 = LRFinder(lcl_chkpt_dir=lcl_chkpt_dir, s3_chkpt_dir=s3_chkpt_dir)
    
    


    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_recall', mode='max', patience=20, verbose=1)
    
    if lr_callback == "clr":
        
        clr_initial = args.clr_initial
        clr_max = args.clr_max
        clr_step = args.clr_step
        
        clr = CyclicalLearningRate(initial_learning_rate=clr_initial,
                                        maximal_learning_rate=clr_max,
                                        step_size=clr_step,
                                       scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
                                        scale_mode='cycle',
                                        name='CyclicalLearningRate')
    
        lrs = tf.keras.callbacks.LearningRateScheduler(clr, verbose=1)
        
        lr_callback = lrs
        
        
    if lr_callback == "lrplateau":
        
        lr_reduce_min = args.lr_reduce_min

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                         mode='min', 
                                                         factor=0.1,
                                                         patience=5, 
                                                         min_lr=lr_reduce_min, 
                                                         verbose=1)
        lr_callback = reduce_lr
    

    callbacks_list = [save_checkpoint_s3, early_stop_callback,lr_callback, WandbCallback()]
    
    
    if not early_stop:
        
        callbacks_list = [save_checkpoint_s3,lr_callback, WandbCallback()]
        
    
            
    if lr_callback == "lrfinder":
        
        lr_callback = save_lr_finder_s3
        
        callbacks_list = [lr_callback, WandbCallback()]
        

    ######## WIP: multi GPUs ###########
    # if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    #     print("Running on GPU using mirrored_strategy")
    #     list_gpus = []
    #     for gpu in range(0, gpu_count):
    #         list_gpus.append(f"/gpu:{gpu}")
    #
    #     print(f"list_gpus: {list_gpus}")
    #     # https://towardsdatascience.com/quick-start-to-multi-gpu-deep-learning-on-aws-sagemaker-using-tf-distribute-9ee08bc9612b
    #     mirrored_strategy = tf.distribute.MirroredStrategy(devices=list_gpus)
    #     batch_size = batch_size * mirrored_strategy.num_replicas_in_sync
    #     with mirrored_strategy.scope():
    #         model = define_model(numclasses, input_shape, starting_checkpoint, lcl_chkpt_dir)
    #     with mirrored_strategy.scope():
    #       # Set reduction to `none` so we can do the reduction afterwards and divide by
    #       # global batch size.
    #       loss_object = SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.NONE)
    #       def compute_loss(labels, predictions):
    #         per_example_loss = loss_object(labels, predictions)
    #         return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    #         model.compile(loss=SigmoidFocalCrossEntropy(),
    #                       # https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
    #                       optimizer=keras.optimizers.Adam(lr),
    #                       metrics=[tf.metrics.BinaryAccuracy(name='accuracy'),
    #                                tf.keras.metrics.Precision(name='precision'),
    #                                # Computes the precision of the predictions with respect to the labels.
    #                                tf.keras.metrics.Recall(name='recall'),
    #                                # Computes the recall of the predictions with respect to the labels.
    #                                F1Score(num_classes=numclasses, name="f1_score")
    #                                # https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
    #                                ]
    #                       )
    # else:
    #     print("Running on CPU")
    model,start_epoch = define_model(numclasses, input_shape, output_type, starting_checkpoint, lcl_chkpt_dir)

    if output_type == 'multilabel':
        model_loss = BinaryCrossentropy()
    elif output_type == 'multiclass':
        model_loss = CategoricalCrossentropy()
    else:
        raise ValueError(f'output type must be multilabel or multiclass, not {output_type}')


    model.compile(loss=model_loss,
                  # https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
                  optimizer=keras.optimizers.Adam(lr),
                  metrics=[tf.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Precision(class_id=1,name='ISL_precision'),
                           # Computes the precision of the predictions with respect to the labels.
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.Recall(class_id=1,name='ISL_recall'),
                           # Computes the recall of the predictions with respect to the labels.
                           F1Score(num_classes=numclasses, name="f1_score")
                           # https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
                           ]
                  )

    print(f"training_dir {os.path.join(training_dir, training_file)}")
    print(f"val_dir {os.path.join(training_dir, validation_file)}")

    print("Dataloader initialization...")
    gen = DataLoader(training_dir=training_dir,
                     label_file_path_train=os.path.join(training_dir, training_file),
                     label_file_path_val=os.path.join(training_dir, validation_file),
                     bucket_name=bucket,
                     data_extension_type='.tif',
                     bands=bands,
                     augment=augmentation_data,
                     enable_shuffle=True,
                     training_data_batch_size=batch_size,
                     enable_data_prefetch=True)
    
    history = model.fit(gen.training_dataset,
                        validation_data=gen.validation_dataset,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        initial_epoch=start_epoch 
                        )

